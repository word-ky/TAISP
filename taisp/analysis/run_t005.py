"""Paired frozen-detector self-supervision screen and clean control."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions, combine_support
from taisp.tta import AdaptConfig, adapt
from .coco import COCOSubset, prediction_records, subset_ap
from .corruptions import CASES, corrupt
from .oracle import detector_task_loss
from .run_t002 import environment_metadata, physical_vector
from .spatial_diagnostics import norm_match


def serializable(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def prepare_support(detector, x, config):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        prediction = detector(x)[0]
        base = select_predictions(prediction, config['support_threshold'], config['support_topk'])
    torch.cuda.synchronize()
    base_seconds = time.perf_counter()-start
    base_peak = torch.cuda.max_memory_allocated()/1024**2
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        flipped = select_predictions(detector(x.flip(-1))[0], config['support_threshold'], config['support_topk'])
        support = combine_support(base, flipped, x.shape[-1], config['match_iou'])
    torch.cuda.synchronize()
    extra_seconds = time.perf_counter()-start
    return support, prediction, {'base_seconds': base_seconds, 'flip_match_seconds': extra_seconds,
                                'base_peak_mb': base_peak, 'flip_match_peak_mb': torch.cuda.max_memory_allocated()/1024**2}


def run(config, data_root, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False
    device = torch.device(config['device'])
    data = COCOSubset(data_root)
    ids = data.ids[:limit] if limit is not None else data.ids[:config['count']]
    generic = load_clip_guidance(device, local_files_only=True)
    detector, isp = load_detector(device), DifferentiableISP().to(device)
    cfg = AdaptConfig(steps=config['semantic_steps'], lr=config['semantic_lr'],
                      consistency_weight=0, regularization_weight=0)
    cases = list(CASES)+[('clean', 0)]
    meta = environment_metadata(config)
    meta.update(evaluated_image_ids=ids, smoke_limit=limit, annotation_sha256=data.manifest['annotation_sha256'],
                support='original base/flip score>=0.5 top20; same-class IoU>=0.5; greedy descending IoU then base/flip index; fixed view-specific boxes',
                object_weights='base original confidence for pseudo; arithmetic mean of original base/flip confidence for stable; normalize sum1',
                logits='model.transform -> backbone -> fixed box scale -> box_roi_pool -> box_head -> box_predictor; full91 classes incl background, bypass RPN/NMS',
                losses='weighted CE; stable0.5*(CEbase+CEflip); stable_JS plus JS coefficient1; no box regression; empty support zero loss/update',
                timing='adapt plus variant-required original support setup; separate setup/adapt peaks, fixed order; analysis excluded',
                norm_match_reference='same-image/case global_generic gradient; zero stays zero; labels not used in scaling',
                cosine='raw undefined cosine=null; zero-coded cosine used only in labeled all-observation paired summaries',
                clean_control='same200 clean images; before predictions contemporaneous; reported separately from corrupted overall')
    (output/'environment.json').write_text(json.dumps(meta, indent=2)+'\n')
    shutil.copyfile(data.root/'subset.json', output/'subset.json')
    predictions = {f'{f}_s{s}_{v}{k}': [] for f, s in cases for v in config['variants'] for k in (1, 3)}
    predictions.update({f'{f}_s{s}_before': [] for f, s in cases})
    rows = []
    initial_physical = physical_vector(isp, torch.zeros(8, device=device))
    started = time.time()
    with (output/'samples.jsonl').open('w', encoding='utf-8') as samples:
        for index, image_id in enumerate(ids):
            clean, targets = data.load(image_id, device)
            for family, severity in cases:
                x = clean if family == 'clean' else corrupt(clean, family, severity)
                seed = config['seed']+image_id
                support, original_prediction, setup = prepare_support(detector, x, config)
                predictions[f'{family}_s{severity}_before'].extend(prediction_records(image_id, original_prediction))
                support_record = serializable(support)
                p0 = torch.zeros(8, device=device, requires_grad=True)
                before, _ = detector_task_loss(detector, isp(x, p0), targets, seed=seed)
                g_det = torch.autograd.grad(before, p0)[0].detach()
                before_value = before.item()
                del before
                global_gradient = None
                for variant in config['variants']:
                    native = variant != 'global_generic'
                    guidance = DetectorNativeLoss(detector, support, variant, config['js_coefficient']) if native else generic
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    result = adapt(x, isp, guidance, config=cfg)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    peak = torch.cuda.max_memory_allocated()/1024**2
                    g = x.new_tensor(result.diagnostics[0]['gradient_per_coordinate'])
                    if not native:
                        global_gradient = g
                    matched = norm_match(g, global_gradient)
                    phi1 = x.new_tensor(result.diagnostics[1]['phi'])
                    y1 = isp(x, phi1).detach()
                    pm = -cfg.lr*matched
                    ym = isp(x, pm).detach()
                    with torch.no_grad():
                        loss1, _ = detector_task_loss(detector, y1, targets, seed=seed)
                        loss3, _ = detector_task_loss(detector, result.enhanced, targets, seed=seed)
                        lossm, _ = detector_task_loss(detector, ym, targets, seed=seed)
                        components = {str(k): [v.item() for v in guidance.components(y)]
                                      for k, y in ((0, isp(x, p0)), (1, y1), (3, result.enhanced))} if native else {}
                        for k, y in ((1, y1), (3, result.enhanced)):
                            predictions[f'{family}_s{severity}_{variant}{k}'].extend(prediction_records(image_id, detector(y)[0]))
                    norm = (g.norm()*g_det.norm()).item()
                    setup_time = (setup['base_seconds']+(setup['flip_match_seconds'] if variant != 'det_pseudo' else 0)) if native else 0
                    n = len(guidance.boxes) if native else None
                    row = {'image_id': image_id, 'family': family, 'severity': severity, 'variant': variant,
                           'g_sem': g.tolist(), 'g_det': g_det.tolist(), 'g_sem_norm': g.norm().item(), 'g_det_norm': g_det.norm().item(),
                           'gradient_cosine': torch.dot(g, g_det).item()/norm if norm > 0 else None,
                           'linear_prediction': (-cfg.lr*torch.dot(g, g_det)).item(),
                           'det_loss_before': before_value, 'det_loss_delta_sem1': loss1.item()-before_value,
                           'det_loss_delta_sem3': loss3.item()-before_value,
                           'semantic_loss_before': result.diagnostics[0]['semantic'],
                           'semantic_loss_1': result.diagnostics[1]['semantic'], 'semantic_loss_3': result.diagnostics[-1]['semantic'],
                           'native_ce_js_0_1_3': components, 'diagnostics': result.diagnostics,
                           'saturation_before': result.diagnostics[0]['saturation_rate'],
                           'saturation_1': result.diagnostics[1]['saturation_rate'], 'saturation_3': result.diagnostics[-1]['saturation_rate'],
                           'phi_norm_1': phi1.norm().item(), 'phi_norm_3': result.phi.norm().item(),
                           'physical_change_1': (physical_vector(isp, phi1)-initial_physical).tolist(),
                           'physical_change_3': (physical_vector(isp, result.phi)-initial_physical).tolist(),
                           'adapt_seconds_3': elapsed, 'deploy_seconds_3': elapsed+setup_time,
                           'peak_allocated_mb': peak, 'setup': setup,
                           'support': support_record, 'support_count': n, 'no_update_fallback': n == 0,
                           'object_weights': guidance.weights.tolist() if native else None,
                           'norm_matched': {'gradient': matched.tolist(), 'gradient_norm': matched.norm().item(),
                               'target_norm': global_gradient.norm().item(), 'zero_gradient': g.norm().item() == 0,
                               'phi': pm.tolist(), 'det_loss_delta': lossm.item()-before_value,
                               'linear_prediction': (-cfg.lr*torch.dot(matched, g_det)).item(),
                               'saturation': ((ym <= 1e-4) | (ym >= 1-1e-4)).float().mean().item()}}
                    samples.write(json.dumps(row, allow_nan=False)+'\n')
                    samples.flush()
                    rows.append(row)
            print(f'completed {index+1}/{len(ids)} image_id={image_id} elapsed={time.time()-started:.1f}s', flush=True)
    pd = output/'predictions'
    pd.mkdir(exist_ok=True)
    metrics = {}
    for name, records in predictions.items():
        (pd/f'{name}.json').write_text(json.dumps(records)+'\n')
        print(f'Evaluating {name}', flush=True)
        metrics[name] = subset_ap(data.coco, ids, records)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps({v: {f'{f}_s{s}': {'count': len(ids)} for f, s in cases} for v in config['variants']}, indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'images': len(ids), 'samples': len(rows), 'elapsed_seconds': time.time()-started})+'\n')
    print(f'Finished {len(ids)} images / {len(rows)} variant observations', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/t005.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.data_root, args.output, args.limit)


if __name__ == '__main__':
    main()
