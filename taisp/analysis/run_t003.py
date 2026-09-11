"""T003 fixed-subset diagnostic. All oracle-family choices are analysis-only."""

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
from taisp.losses.conditioned_clip import CONCEPTS, NEGATIVE_BANKS, COORDINATE_MASKS, load_conditioner
from taisp.losses.semantic import SemanticDirectionLoss
from taisp.models.detector import load_detector
from taisp.tta import AdaptConfig, adapt
from .coco import COCOSubset, prediction_records, subset_ap
from .corruptions import CASES, corrupt
from .oracle import detector_task_loss
from .run_t002 import environment_metadata, physical_vector, summarize

FAMILY_INDEX = {'gamma': 0, 'contrast': 1, 'color_cast': 2}


def choose_variant(name, episode, conditioner, family):
    """Oracle IDs are confined here; deployment conditioner never sees them."""
    guidance, gate = episode.guidance, None
    index = FAMILY_INDEX[family]
    if name in ('oracle_prompt', 'oracle_both'):
        guidance = SemanticDirectionLoss(conditioner.encoder, conditioner.positive - conditioner.negatives[index])
    if name == 'soft_gate':
        gate = episode.gate
    elif name in ('soft_oracle_gate', 'oracle_both'):
        gate = conditioner.masks[index]
    return guidance, gate


def run(config, data_root, baseline, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False
    device = torch.device(config['device'])
    data = COCOSubset(data_root)
    ids = data.ids[:limit] if limit is not None else data.ids[:config['count']]
    baseline_rows = [json.loads(line) for line in (baseline / 'samples.jsonl').read_text().splitlines()]
    saved = {(r['image_id'], r['family'], r['severity']): r for r in baseline_rows}
    generic = load_clip_guidance(device, local_files_only=True)
    conditioner = load_conditioner(generic, config['temperature'], local_files_only=True)
    detector, isp = load_detector(device), DifferentiableISP().to(device)
    settings = AdaptConfig(steps=config['semantic_steps'], lr=config['semantic_lr'],
                           consistency_weight=config['consistency_weight'],
                           regularization_weight=config['regularization_weight'])
    meta = environment_metadata(config)
    meta.update(evaluated_image_ids=ids, smoke_limit=limit,
                annotation_sha256=data.manifest['annotation_sha256'],
                negative_concept_banks=NEGATIVE_BANKS, concept_order=CONCEPTS,
                masks=COORDINATE_MASKS, baseline_study=str(baseline),
                baseline_source=json.loads((baseline / 'environment.json').read_text())['source_revision'],
                baseline_samples_sha256=hashlib.sha256((baseline / 'samples.jsonl').read_bytes()).hexdigest(),
                baseline_metrics_sha256=hashlib.sha256((baseline / 'metrics.json').read_bytes()).hexdigest(),
                oracle='family choices only in analysis; no annotated gradients in conditioning/gating',
                gradient_definition='g_sem is effective gated update; raw objective gradient saved separately',
                baseline_reuse='T002 clean/corrupted AP only; fresh g_det shared across all six variants; generic rerun',
                timing='condition inference + variant setup + three-step adapt including diagnostics; excludes evaluation')
    (output / 'environment.json').write_text(json.dumps(meta, indent=2) + '\n')
    shutil.copyfile(data.root / 'subset.json', output / 'subset.json')
    predictions = {f'{f}_s{s}_{v}{k}': [] for f, s in CASES for v in config['variants'] for k in (1, 3)}
    rows = []
    baseline_checks = []
    initial_physical = physical_vector(isp, torch.zeros(8, device=device))
    started = time.time()
    with (output / 'samples.jsonl').open('w', encoding='utf-8') as samples:
        for index, image_id in enumerate(ids):
            clean, targets = data.load(image_id, device)
            for family, severity in CASES:
                x = corrupt(clean, family, severity)
                base = saved[image_id, family, severity]
                seed = config['seed'] + image_id
                phi0 = torch.zeros(8, device=device, requires_grad=True)
                initial_loss, _ = detector_task_loss(detector, isp(x, phi0), targets, seed=seed)
                g_det = torch.autograd.grad(initial_loss, phi0)[0].detach()
                loss0 = initial_loss.item()
                del initial_loss
                baseline_check = {'image_id': image_id, 'family': family, 'severity': severity,
                    'loss_abs_error': abs(loss0-base['det_loss_before']),
                    'gradient_max_abs_error': (g_det-x.new_tensor(base['g_det'])).abs().max().item()}
                # Repeated native detector backward was observed to vary while
                # its forward loss was exact. Share one fresh gradient across
                # variants; do not widen equality tolerances for cached gradients.
                if limit is not None:
                    torch.testing.assert_close(x.new_tensor(loss0), x.new_tensor(base['det_loss_before']), atol=1e-6, rtol=1e-5)
                    repeat = adapt(x, isp, generic, config=settings)
                    torch.testing.assert_close(repeat.phi, x.new_tensor(base['diagnostics'][-1]['phi']), atol=1e-7, rtol=1e-5)
                    baseline_check['generic_phi3_max_abs_error'] = (repeat.phi-x.new_tensor(base['diagnostics'][-1]['phi'])).abs().max().item()
                    del repeat
                baseline_checks.append(baseline_check)
                for variant in config['variants']:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    episode = None if variant == 'generic' else conditioner.prepare(x)
                    guidance, gate = (generic, None) if episode is None else choose_variant(variant, episode, conditioner, family)
                    result = adapt(x, isp, guidance, config=settings, coordinate_gate=gate)
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated() / 1024**2
                    effective = x.new_tensor(result.diagnostics[0]['gradient_per_coordinate'])
                    raw = result.diagnostics[0].get('raw_gradient_per_coordinate', result.diagnostics[0]['gradient_per_coordinate'])
                    phi1 = x.new_tensor(result.diagnostics[1]['phi'])
                    sem1 = isp(x, phi1).detach()
                    norm = (effective.norm() * g_det.norm()).item()
                    with torch.no_grad():
                        loss1, _ = detector_task_loss(detector, sem1, targets, seed=seed)
                        loss3, _ = detector_task_loss(detector, result.enhanced, targets, seed=seed)
                        common1 = generic(x, sem1).item()
                        common3 = generic(x, result.enhanced).item()
                        for k, enhanced in ((1, sem1), (3, result.enhanced)):
                            predictions[f'{family}_s{severity}_{variant}{k}'].extend(prediction_records(image_id, detector(enhanced)[0]))
                    row = {**base, 'variant': variant,
                        'g_det': g_det.cpu().tolist(), 'g_det_norm': g_det.norm().item(), 'det_loss_before': loss0,
                        'gradient_cosine': (torch.dot(effective, g_det) / norm).item() if norm > 0 else None,
                        'g_sem': effective.cpu().tolist(), 'g_sem_raw': raw,
                        'g_sem_norm': effective.norm().item(),
                        'det_loss_delta_sem1': loss1.item()-loss0, 'det_loss_delta_sem3': loss3.item()-loss0,
                        'semantic_loss_before': result.diagnostics[0]['semantic'],
                        'semantic_loss_1': result.diagnostics[1]['semantic'],
                        'semantic_loss_3': result.diagnostics[-1]['semantic'],
                        'generic_semantic_loss_1': common1, 'generic_semantic_loss_3': common3,
                        'physical_change_1': (physical_vector(isp, phi1)-initial_physical).cpu().tolist(),
                        'physical_change_3': (physical_vector(isp, result.phi)-initial_physical).cpu().tolist(),
                        'saturation_before': result.diagnostics[0]['saturation_rate'],
                        'saturation_1': result.diagnostics[1]['saturation_rate'],
                        'saturation_3': result.diagnostics[-1]['saturation_rate'],
                        'adapt_seconds_3': duration, 'peak_allocated_mb': peak,
                        'condition_weights': episode.weights.flatten().cpu().tolist() if episode else None,
                        'condition_similarities': episode.similarities.flatten().cpu().tolist() if episode else None,
                        'coordinate_gate': gate.flatten().cpu().tolist() if gate is not None else [1.]*8,
                        'direction': guidance.direction.flatten().cpu().tolist(),
                        'diagnostics': result.diagnostics}
                    samples.write(json.dumps(row, allow_nan=False)+'\n')
                    samples.flush()
                    rows.append(row)
            print(f'completed {index+1}/{len(ids)} image_id={image_id} elapsed={time.time()-started:.1f}s', flush=True)
    # Baseline AP is copied only for the full identical subset. A smoke uses the
    # original baseline predictions filtered to its IDs with the same evaluator.
    baseline_metrics = json.loads((baseline / 'metrics.json').read_text())
    if limit is not None:
        baseline_metrics = {name: subset_ap(data.coco, ids, [p for p in json.loads((baseline / 'predictions' / f'{name}.json').read_text()) if p['image_id'] in ids])
                            for name in baseline_metrics}
    (output / 'baseline_metrics.json').write_text(json.dumps(baseline_metrics, indent=2)+'\n')
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    metrics = {}
    for name, records in predictions.items():
        (prediction_dir / f'{name}.json').write_text(json.dumps(records)+'\n')
        print(f'Evaluating {name}', flush=True)
        metrics[name] = subset_ap(data.coco, ids, records)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    summary = {v: summarize([r for r in rows if r['variant'] == v]) for v in config['variants']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output / 'baseline_checks.json').write_text(json.dumps(baseline_checks, indent=2)+'\n')
    (output / 'completion.json').write_text(json.dumps({'images': len(ids), 'samples': len(rows),
                  'variants': config['variants'], 'elapsed_seconds': time.time()-started})+'\n')
    print(f'Finished {len(ids)} images / {len(rows)} variant observations', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/t003.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Implementation smoke only')
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.data_root, args.baseline, args.output, args.limit)


if __name__ == '__main__':
    main()
