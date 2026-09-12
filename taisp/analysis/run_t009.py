"""T009: accepted T007 adaptation, K3-only evaluation on three frozen detectors."""
import argparse
import hashlib
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
from taisp.tta.trust_radius import adapt_clip_radius
from .coco import COCOSubset, prediction_records
from .corruptions import CASES, corrupt
from .fcos_target import load_fcos_target, target_metadata
from .replication import replication_ap
from .run_t002 import environment_metadata
from .run_t005 import serializable
from .ssd_target import load_ssd_target, ssd_metadata


def run(config, data_root, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False
    device = torch.device(config['device'])
    data = COCOSubset(data_root)
    ids = data.ids[:limit] if limit is not None else data.ids[:config['count']]
    generic = load_clip_guidance(device, local_files_only=True)
    source = load_detector(device)
    detectors = {'source': source, 'target': load_fcos_target(device), 'ssd': load_ssd_target(device)}
    isp = DifferentiableISP().to(device)
    cfg = AdaptConfig(steps=config['semantic_steps'], lr=config['semantic_lr'], consistency_weight=0, regularization_weight=0)
    cases = list(CASES)+[('clean', 0)]
    names = [f'{f}_s{s}' for f, s in cases]
    meta = environment_metadata(config)
    meta.update(evaluated_image_ids=ids, smoke_limit=limit, cases=names, detectors=list(detectors),
                target=target_metadata(detectors['target']), ssd=ssd_metadata(detectors['ssd']),
                annotation_sha256=data.manifest['annotation_sha256'],
                subset_sha256=hashlib.sha256((data.root/'subset.json').read_bytes()).hexdigest(),
                support='unchanged source original detached score>=0.5 top20',
                endpoint='K3; no K selection, target gradients or oracle-loss study',
                adaptation='unchanged T007 adapt/adapt_clip_radius; source direction and current-phi CLIP norm',
                timing='synchronized adaptation; native includes source original support setup; target evaluation excluded; all4models resident',
                interpretation='frozen-method1000 disjoint images plus5 predeclared blocks; no AP confidence intervals')
    assert meta['ssd'] == json.loads(Path('research_log/T009_ssd_pin.json').read_text())
    (output/'environment.json').write_text(json.dumps(meta, indent=2)+'\n')
    shutil.copyfile(data.root/'subset.json', output/'subset.json')
    variants = ['no_adapt']+config['variants']
    predictions = {f'{d}_{case}_{v}': [] for d in detectors for case in names for v in variants}
    count, started = 0, time.time()
    with (output/'samples.jsonl').open('w', encoding='utf-8') as samples:
        for index, image_id in enumerate(ids):
            clean, _ = data.load(image_id, device)
            for family, severity in cases:
                case = f'{family}_s{severity}'
                x = clean if family == 'clean' else corrupt(clean, family, severity)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.no_grad():
                    original_source = source(x)[0]
                    selected = select_predictions(original_source, config['support_threshold'], config['support_topk'])
                torch.cuda.synchronize()
                setup_seconds = time.perf_counter()-start
                support = {'base': selected}
                predictions[f'source_{case}_no_adapt'].extend(prediction_records(image_id, original_source))
                with torch.no_grad():
                    for name in ('target', 'ssd'):
                        predictions[f'{name}_{case}_no_adapt'].extend(prediction_records(image_id, detectors[name](x)[0]))
                for variant in config['variants']:
                    native = variant != 'global_generic'
                    guidance = DetectorNativeLoss(source, support, 'det_pseudo') if native else generic
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    # Same accepted T007 calls; neither targets nor labels are arguments.
                    result = (adapt_clip_radius(x, isp, guidance, generic, steps=cfg.steps, lr=cfg.lr, eps=config['radius_eps'])
                              if variant == 'det_pseudo_clip_radius' else adapt(x, isp, guidance, config=cfg))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    peak = torch.cuda.max_memory_allocated()/1024**2
                    with torch.no_grad():
                        for name, detector in detectors.items():
                            predictions[f'{name}_{case}_{variant}'].extend(prediction_records(image_id, detector(result.enhanced)[0]))
                    row = {'image_id': image_id, 'family': family, 'severity': severity, 'variant': variant,
                           'diagnostics': result.diagnostics,
                           **{f'phi_norm_{k}': x.new_tensor(result.diagnostics[k]['phi']).norm().item() for k in (1, 2, 3)},
                           'saturation_before': result.diagnostics[0]['saturation_rate'],
                           'saturation_3': result.diagnostics[-1]['saturation_rate'],
                           'adapt_seconds_3': elapsed, 'deploy_seconds_3': elapsed+(setup_seconds if native else 0),
                           'peak_allocated_mb': peak, 'source_setup_seconds': setup_seconds,
                           'support': serializable(selected), 'support_count': len(selected['boxes']),
                           'no_update_fallback': native and len(selected['boxes']) == 0,
                           'object_weights': guidance.weights.tolist() if native else None}
                    samples.write(json.dumps(row, allow_nan=False)+'\n')
                    samples.flush()
                    count += 1
            print(f'completed {index+1}/{len(ids)} image_id={image_id} elapsed={time.time()-started:.1f}s', flush=True)
    pd = output/'predictions'
    pd.mkdir(exist_ok=True)
    # Persist all raw predictions before aggregate/block AP evaluation.
    for name, records in predictions.items():
        (pd/f'{name}.json').write_text(json.dumps(records)+'\n')
    metrics = {}
    for name, records in predictions.items():
        print(f'Evaluating aggregate/blocks {name}', flush=True)
        for group, values in replication_ap(data.coco, ids, records, data.manifest['replication_blocks']).items():
            metrics.setdefault(group, {})[name] = values
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'images': len(ids), 'samples': count, 'elapsed_seconds': time.time()-started,
                                                     'evaluations': sum(len(v) for v in metrics.values())})+'\n')
    print(f'Finished {len(ids)} images / {count} variant observations', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/t009.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.data_root, args.output, args.limit)


if __name__ == '__main__':
    main()
