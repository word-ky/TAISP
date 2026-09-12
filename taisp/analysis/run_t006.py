"""T006: source-only ISP adaptation, paired source/independent-target analysis."""
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
from taisp.models.detector_signal import select_predictions
from taisp.tta import AdaptConfig, adapt
from .coco import COCOSubset, prediction_records, subset_ap
from .corruptions import CASES, corrupt
from .fcos_target import load_fcos_target, target_task_loss, target_metadata
from .oracle import detector_task_loss
from .run_t002 import environment_metadata, physical_vector
from .run_t005 import serializable
from .spatial_diagnostics import norm_match


def cosine(a, b):
    denominator = (a.norm()*b.norm()).item()
    return torch.dot(a, b).item()/denominator if denominator > 0 else None


def detector_diagnostics(gradient, oracle_gradient, before, after, matched, lr):
    return {'g_det': oracle_gradient.tolist(), 'g_det_norm': oracle_gradient.norm().item(),
            'gradient_cosine': cosine(gradient, oracle_gradient),
            'coordinate_dot': (gradient*oracle_gradient).tolist(),
            'linear_prediction': (-lr*torch.dot(gradient, oracle_gradient)).item(),
            'det_loss_before': before, 'det_loss_delta_sem1': after['1']-before,
            'det_loss_delta_sem3': after['3']-before,
            'norm_matched': {'det_loss_delta': after['matched']-before,
                             'linear_prediction': (-lr*torch.dot(matched, oracle_gradient)).item(),
                             'coordinate_dot': (matched*oracle_gradient).tolist()}}


def run(config, data_root, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False
    device = torch.device(config['device'])
    data = COCOSubset(data_root)
    ids = data.ids[:limit] if limit is not None else data.ids[:config['count']]
    generic = load_clip_guidance(device, local_files_only=True)
    source, target, isp = load_detector(device), load_fcos_target(device), DifferentiableISP().to(device)
    detectors = {'source': source, 'target': target}
    cfg = AdaptConfig(steps=config['semantic_steps'], lr=config['semantic_lr'], consistency_weight=0, regularization_weight=0)
    cases = list(CASES)+[('clean', 0)]
    meta = environment_metadata(config)
    meta.update(evaluated_image_ids=ids, smoke_limit=limit, target=target_metadata(target),
                annotation_sha256=data.manifest['annotation_sha256'],
                support='source original base only; score>=0.5 stable descending top20; detached fixed boxes/classes/confidence',
                loss='T005 det_pseudo score-normalized fixed-ROI CE; no target, flip or stable/JS',
                target_oracle='analysis-only native FCOS classification+bbox_regression+bbox_ctrness; top-level training flag only; children eval',
                norm_match='same-case contemporaneous CLIP gradient norm; zero stays zero; no oracle in scaling',
                timing='adapt+source original setup for det_pseudo; target analysis excluded; shared-process source,target,CLIP resident',
                severity='s1 and s2 each pool3families perimage; clean separate; no severity used in deployment')
    (output/'environment.json').write_text(json.dumps(meta, indent=2)+'\n')
    shutil.copyfile(data.root/'subset.json', output/'subset.json')
    predictions = {f'{d}_{f}_s{s}_{v}{k}': [] for d in detectors for f, s in cases for v in config['variants'] for k in (1, 3)}
    predictions.update({f'{d}_{f}_s{s}_before': [] for d in detectors for f, s in cases})
    count = 0
    started = time.time()
    initial_physical = physical_vector(isp, torch.zeros(8, device=device))
    with (output/'samples.jsonl').open('w', encoding='utf-8') as samples:
        for index, image_id in enumerate(ids):
            clean, annotations = data.load(image_id, device)
            for family, severity in cases:
                x = clean if family == 'clean' else corrupt(clean, family, severity)
                seed = config['seed']+image_id
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.no_grad():
                    original_source = source(x)[0]
                    selected = select_predictions(original_source, config['support_threshold'], config['support_topk'])
                torch.cuda.synchronize()
                setup_seconds = time.perf_counter()-start
                setup_peak = torch.cuda.max_memory_allocated()/1024**2
                support = {'base': selected}
                predictions[f'source_{family}_s{severity}_before'].extend(prediction_records(image_id, original_source))
                with torch.no_grad():
                    predictions[f'target_{family}_s{severity}_before'].extend(prediction_records(image_id, target(x)[0]))
                # These closures and gradients exist only in this annotated driver.
                losses = {'source': lambda y: detector_task_loss(source, y, annotations, seed=seed)[0],
                          'target': lambda y: target_task_loss(target, y, annotations)[0]}
                gradients, before = {}, {}
                for name in detectors:
                    p0 = torch.zeros(8, device=device, requires_grad=True)
                    value = losses[name](isp(x, p0))
                    gradients[name] = torch.autograd.grad(value, p0)[0].detach()
                    before[name] = value.item()
                    del value
                global_gradient = None
                for variant in config['variants']:
                    native = variant == 'det_pseudo'
                    guidance = DetectorNativeLoss(source, support, variant) if native else generic
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    # Deployment receives neither target nor annotations.
                    result = adapt(x, isp, guidance, config=cfg)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    peak = torch.cuda.max_memory_allocated()/1024**2
                    gradient = x.new_tensor(result.diagnostics[0]['gradient_per_coordinate'])
                    if not native:
                        global_gradient = gradient
                    matched = norm_match(gradient, global_gradient)
                    phi1 = x.new_tensor(result.diagnostics[1]['phi'])
                    pm = -cfg.lr*matched
                    with torch.no_grad():
                        images = {'1': isp(x, phi1), '3': result.enhanced, 'matched': isp(x, pm)}
                        after = {d: {k: losses[d](y).item() for k, y in images.items()} for d in detectors}
                        for d, detector in detectors.items():
                            for k in ('1', '3'):
                                predictions[f'{d}_{family}_s{severity}_{variant}{k}'].extend(prediction_records(image_id, detector(images[k])[0]))
                    diagnostics = {d: detector_diagnostics(gradient, gradients[d], before[d], after[d], matched, cfg.lr) for d in detectors}
                    row = {'image_id': image_id, 'family': family, 'severity': severity, 'variant': variant,
                           'g_sem': gradient.tolist(), 'g_sem_norm': gradient.norm().item(),
                           'source_target_cosine': cosine(gradients['source'], gradients['target']),
                           'source_target_coordinate_dot': (gradients['source']*gradients['target']).tolist(),
                           'source': diagnostics['source'], 'target': diagnostics['target'],
                           'diagnostics': result.diagnostics,
                           'semantic_loss_before': result.diagnostics[0]['semantic'],
                           'semantic_loss_1': result.diagnostics[1]['semantic'], 'semantic_loss_3': result.diagnostics[-1]['semantic'],
                           'saturation_before': result.diagnostics[0]['saturation_rate'],
                           'saturation_1': result.diagnostics[1]['saturation_rate'], 'saturation_3': result.diagnostics[-1]['saturation_rate'],
                           'phi_norm_1': phi1.norm().item(), 'phi_norm_3': result.phi.norm().item(),
                           'physical_change_1': (physical_vector(isp, phi1)-initial_physical).tolist(),
                           'physical_change_3': (physical_vector(isp, result.phi)-initial_physical).tolist(),
                           'adapt_seconds_3': elapsed, 'deploy_seconds_3': elapsed+(setup_seconds if native else 0),
                           'peak_allocated_mb': peak, 'source_setup_seconds': setup_seconds, 'source_setup_peak_mb': setup_peak,
                           'support': serializable(selected), 'support_count': len(selected['boxes']),
                           'no_update_fallback': native and len(selected['boxes']) == 0,
                           'object_weights': guidance.weights.tolist() if native else None,
                           'norm_matched': {'gradient': matched.tolist(), 'gradient_norm': matched.norm().item(),
                               'target_norm': global_gradient.norm().item(), 'zero_gradient': gradient.norm().item() == 0,
                               'phi': pm.tolist(), 'saturation': ((images['matched'] <= 1e-4) | (images['matched'] >= 1-1e-4)).float().mean().item()}}
                    samples.write(json.dumps(row, allow_nan=False)+'\n')
                    samples.flush()
                    count += 1
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
    (output/'completion.json').write_text(json.dumps({'images': len(ids), 'samples': count, 'elapsed_seconds': time.time()-started})+'\n')
    print(f'Finished {len(ids)} images / {count} variant observations', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/t006.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.data_root, args.output, args.limit)


if __name__ == '__main__':
    main()
