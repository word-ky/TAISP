"""Frozen T009 aggregate/block AP and fixed safety/runtime panel; no proxy search."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_t005 import distribution
from taisp.analysis.replication import ap_contrasts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    root = args.study
    read = lambda name: json.loads((root/name).read_text())
    env, metrics, manifest = read('environment.json'), read('metrics.json'), read('subset.json')
    samples = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    cases, detectors, variants = env['cases'], env['detectors'], ['no_adapt']+env['config']['variants']
    contrasts = {group: ap_contrasts(values, cases, detectors, variants) for group, values in metrics.items()}
    safety = {}
    for group in ['corrupted_overall', *cases]:
        chosen = [r for r in samples if (r['family'] != 'clean' if group == 'corrupted_overall' else f"{r['family']}_s{r['severity']}" == group)]
        safety[group] = {}
        for variant in env['config']['variants']:
            rr = [r for r in chosen if r['variant'] == variant]
            q = {field: distribution([r[field] for r in rr]) for field in
                 ('phi_norm_1', 'phi_norm_2', 'phi_norm_3', 'saturation_before', 'saturation_3',
                  'support_count', 'source_setup_seconds', 'adapt_seconds_3', 'deploy_seconds_3', 'peak_allocated_mb')}
            q['fallback_fraction'] = float(np.mean([r['no_update_fallback'] for r in rr]))
            q['nonzero_phi3_fraction'] = float(np.mean([r['phi_norm_3'] > 0 for r in rr]))
            q['phi_trajectory_mean'] = np.mean([[d['phi'] for d in r['diagnostics']] for r in rr], axis=0).tolist()
            if variant == 'det_pseudo_clip_radius':
                q['steps'] = {}
                for step in range(4):
                    dd = [r['diagnostics'][step] for r in rr]
                    q['steps'][str(step)] = {field: distribution([d[field] for d in dd if d[field] is not None])
                        for field in ('detector_gradient_norm', 'clip_gradient_norm', 'gradient_norm', 'detector_clip_ratio', 'scale_factor')}
                    q['steps'][str(step)]['scale_above_one_fraction'] = float(np.mean([d['scale_factor'] > 1 for d in dd]))
            safety[group][variant] = q
    replication = {}
    H = 'det_pseudo_clip_radius'
    for detector in ('target', 'ssd'):
        replication[detector] = {}
        for reference in ('no_adapt', 'det_pseudo'):
            block_values = {group: value[detector][H][reference]['macro_corruption_AP_delta']
                            for group, value in contrasts.items() if group != 'aggregate'}
            replication[detector][reference] = {
                'aggregate': contrasts['aggregate'][detector][H][reference]['macro_corruption_AP_delta'],
                'blocks': block_values, 'positive_blocks': sum(x > 0 for x in block_values.values())}
    full_five_blocks = len(contrasts) == 6 and len(env['evaluated_image_ids']) == 1000
    criterion = None if not full_five_blocks else all(replication[d]['no_adapt']['aggregate'] > 0 and
        replication[d]['det_pseudo']['aggregate'] > 0 and replication[d]['no_adapt']['positive_blocks'] >= 4
        for d in ('target', 'ssd'))
    analysis = {'contrasts': contrasts, 'safety': safety, 'replication': replication,
                'full_cohort_complete': full_five_blocks, 'external_AP_criterion_met': criterion,
                'clean_caveat': 'AP criterion does not establish clean neutrality or a complete method',
                'AP_uncertainty': 'five fixed replication blocks; no formal AP confidence interval'}
    (root/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    rows = []
    for group, values in metrics.items():
        for detector in detectors:
            for variant in variants:
                for case in cases:
                    m = values[f'{detector}_{case}_{variant}']
                    row = {'group': group, 'detector': detector, 'variant': variant, 'case': case,
                           **{k: m[k]*100 for k in ('AP', 'AP50', 'AP75')}}
                    for ref in ('no_adapt', 'global_generic', 'det_pseudo'):
                        row.update({f'{k}_delta_{ref}': v for k, v in contrasts[group][detector][variant][ref]['cases'][case].items()})
                    rows.append(row)
    with (root/'AP_tables.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# T009 fixed-method external validation', '',
             f"Images: {len(env['evaluated_image_ids'])}; variant rows: {len(samples)}. K=3 only. No AP confidence intervals.", '',
             '## Independent-target replication', '',
             '| Detector | Reference | Aggregate macro AP delta | Block deltas (AP points) | Positive blocks |',
             '| --- | --- | ---: | --- | ---: |']
    for detector, refs in replication.items():
        for reference, r in refs.items():
            lines.append(f"| {detector} | {reference} | {r['aggregate']:+.4f} | "+' / '.join(f'{k}: {v:+.4f}' for k, v in r['blocks'].items())+f" | {r['positive_blocks']}/{len(r['blocks'])} |")
    criterion_text = str(criterion) if full_five_blocks else 'not assessed on a partial/smoke cohort'
    lines.extend(['', f'Full 1,000-image/five-block coverage: {full_five_blocks}. External AP criterion: {criterion_text}. Clean outcomes still require review.', '',
                  '## Aggregate AP / AP50 / AP75', '', '| Detector | Condition | Method | AP / AP50 / AP75 | AP delta no-adapt / CLIP / raw |', '| --- | --- | --- | ---: | ---: |'])
    for r in rows:
        if r['group'] == 'aggregate':
            lines.append(f"| {r['detector']} | {r['case']} | {r['variant']} | "+' / '.join(f"{r[m]:.3f}" for m in ('AP', 'AP50', 'AP75'))+' | '+
                         ' / '.join(f"{r['AP_delta_'+m]:+.3f}" for m in ('no_adapt', 'global_generic', 'det_pseudo'))+' |')
    lines.extend(['', '## Every method and replication block: macro AP deltas', '',
                  '| Group | Detector | Method | Macro delta no-adapt / CLIP / raw | Positive conditions vs no-adapt / raw | Clean delta vs no-adapt |', '| --- | --- | --- | ---: | ---: | ---: |'])
    for group, ds in contrasts.items():
        for detector, vs in ds.items():
            for variant, refs in vs.items():
                lines.append(f'| {group} | {detector} | {variant} | '+
                    ' / '.join(f"{refs[r]['macro_corruption_AP_delta']:+.4f}" for r in ('no_adapt', 'global_generic', 'det_pseudo'))+
                    f" | {refs['no_adapt']['positive_corruption_count']}/6 / {refs['det_pseudo']['positive_corruption_count']}/6 | {refs['no_adapt']['clean_AP_delta']:+.4f} |")
    lines.extend(['', '## Fixed safety/runtime panel', '',
                  'Shared-process peak allocated GPU memory includes source, FCOS, SSD and CLIP residency. Adaptation latency excludes target evaluation; native variants include original source/support setup. Existing terminal diagnostics retained.', '',
                  '| Group | Method | Mean phi3 | Mean saturation before / after % | Support mean | Fallback % | Deploy sec | Peak MiB |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for group, vs in safety.items():
        for variant, r in vs.items():
            lines.append(f"| {group} | {variant} | {r['phi_norm_3']['mean']:.6f} | {100*r['saturation_before']['mean']:.3f} / {100*r['saturation_3']['mean']:.3f} | {r['support_count']['mean']:.2f} | {100*r['fallback_fraction']:.2f} | {r['deploy_seconds_3']['mean']:.4f} | {r['peak_allocated_mb']['mean']:.1f} |")
    lines.extend(['', 'Full per-block AP/AP50/AP75 and all paired deltas: AP_tables.csv. All phi/ratio/scale distributions and step diagnostics: analysis.json and samples.jsonl.', ''])
    (root/'results.md').write_text('\n'.join(lines))
    (root/'receipt.json').write_text(json.dumps({'source_revision': env['source_revision'],
        'images': len(env['evaluated_image_ids']), 'samples': len(samples), 'evaluations': sum(len(v) for v in metrics.values()),
        'prediction_files': len(list((root/'predictions').glob('*.json'))),
        'subset_sha256': hashlib.sha256((root/'subset.json').read_bytes()).hexdigest(),
        'sample_sha256': hashlib.sha256((root/'samples.jsonl').read_bytes()).hexdigest(),
        'replication_blocks': manifest['replication_blocks']}, indent=2)+'\n')
    print(json.dumps({'full_cohort_complete': full_five_blocks, 'external_AP_criterion_met': criterion, 'replication': replication}, indent=2))


if __name__ == '__main__':
    main()
