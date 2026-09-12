"""Fixed half-dose outcomes against accepted endpoints and frozen random controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_t005 import distribution
from taisp.analysis.dose import half_dose_criteria, ratio_summary
from taisp.analysis.random_controls import randomization
from taisp.analysis.replication import ap_contrasts

HALF, FULL = 'det_pseudo_half_dose', 'det_pseudo_clip_radius'


def report(study, reference, controls):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    env, metrics, manifest = (read(study/n) for n in ('environment.json', 'metrics.json', 'subset.json'))
    full = len(env['evaluated_image_ids']) == 1000 and len(metrics) == 6
    pins = read(Path('research_log/T012_references.json'))
    for directory, expected in [(reference, pins['full' if full else 'smoke']),
                                (controls, pins['T011' if full else 'T011_smoke'])]:
        for name, digest in expected.items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, name
    old_metrics = read(reference/'metrics.json')
    anchor_checks = 0
    for group, values in metrics.items():
        for key, value in values.items():
            if key.endswith(('_no_adapt', '_'+FULL)):
                assert value == old_metrics[group][key]
                anchor_checks += 1
    samples = [json.loads(s) for s in (study/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    old = {}
    with (reference/'samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['variant'] == FULL:
                old[row['image_id'], row['family'], row['severity']] = row
    cases, detectors = env['cases'], env['detectors']
    assert len(samples) == len(env['evaluated_image_ids'])*len(cases)
    support_matches, paired = 0, []
    for row in samples:
        previous = old[row['image_id'], row['family'], row['severity']]
        support_matches += row['support'] == previous['support']
        p = {'image_id': row['image_id'], 'case': f"{row['family']}_s{row['severity']}",
             'half_phi3': row['phi_norm_3'], 'full_phi3': previous['phi_norm_3'],
             'phi3_ratio': row['phi_norm_3']/previous['phi_norm_3'] if previous['phi_norm_3'] > 0 else None,
             'same_original_support': row['support'] == previous['support']}
        for k in range(4):
            d = row['diagnostics'][k]
            gd, gc = np.array(d['detector_gradient']), np.array(d['clip_gradient'])
            h = gd*np.linalg.norm(gc)/(np.linalg.norm(gd)+1e-12)
            np.testing.assert_allclose(d['gradient_per_coordinate'], .5*h, rtol=2e-5, atol=1e-8)
            assert d['dose_coefficient'] == .5
            if k < 3:
                np.testing.assert_allclose(row['diagnostics'][k+1]['phi'], np.array(d['phi'])-.1*np.array(d['gradient_per_coordinate']), rtol=2e-5, atol=1e-8)
                p[f'half_update_norm_{k}'] = d['gradient_norm']
                p[f'full_update_norm_{k}'] = previous['diagnostics'][k]['gradient_norm']
        assert row['diagnostics'][0]['phi'] == [0.]*8
        if row['no_update_fallback']:
            assert row['phi_norm_3'] == 0
        paired.append(p)
    assert support_matches == len(samples)
    contrasts = {g: ap_contrasts(v, cases, detectors, ['no_adapt', FULL, HALF], references=('no_adapt', FULL)) for g, v in metrics.items()}
    ap_rows, macro_rows = [], []
    for group, ds in contrasts.items():
        for detector, variants in ds.items():
            for variant, refs in variants.items():
                macro_rows.append({'group': group, 'detector': detector, 'variant': variant,
                    **{f'{ref}_{field}': refs[ref][field] for ref in refs for field in ('macro_corruption_AP_delta', 'positive_corruption_count', 'clean_AP_delta')}})
                for case in cases:
                    values = metrics[group][f'{detector}_{case}_{variant}']
                    ap_rows.append({'group': group, 'detector': detector, 'variant': variant, 'case': case,
                        **{m: 100*values[m] for m in ('AP', 'AP50', 'AP75')},
                        **{f'{m}_delta_{ref}': v for ref in refs for m, v in refs[ref]['cases'][case].items()}})
    selected_ids = set(env['evaluated_image_ids'])
    groups = {'aggregate': selected_ids, **{g: selected_ids.intersection(ids) for g, ids in manifest['replication_blocks'].items()}}
    safety = {}
    for group, ids in groups.items():
        if not ids:
            continue
        safety[group] = {}
        for condition in ['corrupted_overall', *cases]:
            chosen = [j for j, r in enumerate(samples) if r['image_id'] in ids and
                (r['family'] != 'clean' if condition == 'corrupted_overall' else f"{r['family']}_s{r['severity']}" == condition)]
            rows = [samples[j] for j in chosen]
            q = {field: distribution([r[field] for r in rows]) for field in ('phi_norm_1', 'phi_norm_2', 'phi_norm_3',
                'saturation_before', 'saturation_3', 'adapt_seconds_3', 'deploy_seconds_3', 'source_setup_seconds', 'peak_allocated_mb')}
            q['nonzero_update_fraction'] = float(np.mean([r['phi_norm_3'] > 0 for r in rows]))
            q['fallback_fraction'] = float(np.mean([r['no_update_fallback'] for r in rows]))
            pp = [paired[j] for j in chosen]
            q['full_phi3'] = distribution([p['full_phi3'] for p in pp])
            q['phi3_ratios'] = ratio_summary([p['half_phi3'] for p in pp], [p['full_phi3'] for p in pp])
            q['ratio_of_mean_phi3'] = q['phi_norm_3']['mean']/q['full_phi3']['mean'] if q['full_phi3']['mean'] > 0 else None
            q['steps'] = {}
            for k in range(4):
                q['steps'][str(k)] = {field: distribution([r['diagnostics'][k][field] for r in rows]) for field in
                    ('detector_gradient_norm', 'clip_gradient_norm', 'pre_attenuation_hybrid_norm', 'applied_half_dose_norm', 'saturation_rate', 'step_seconds')}
                if k < 3:
                    q['steps'][str(k)]['half_to_full_update_norm_ratio'] = ratio_summary(
                        [p[f'half_update_norm_{k}'] for p in pp], [p[f'full_update_norm_{k}'] for p in pp])
            safety[group][condition] = q
    random_data = read(controls/'analysis.json')
    random_names = [r['config'] for r in random_data['configurations'] if r['role'] == 'random']
    comparisons, positive_blocks, better_blocks = {}, {}, {}
    for detector in detectors:
        comparisons[detector] = {}
        for group, ds in contrasts.items():
            value = ds[detector][HALF]['no_adapt']['macro_corruption_AP_delta']
            draws = [random_data['summaries'][n][group][detector]['macro_AP_delta_no_adapt'] for n in random_names]
            comparisons[detector][group] = randomization(value, draws)
        positive_blocks[detector] = sum(r['candidate'] > 0 for g, r in comparisons[detector].items() if g != 'aggregate')
        better_blocks[detector] = sum(r['candidate'] > r['random']['median'] for g, r in comparisons[detector].items() if g != 'aggregate')
    clean = safety['aggregate']['clean_s0']
    criteria = half_dose_criteria(
        [comparisons[d]['aggregate']['candidate'] for d in ('target', 'ssd')],
        [positive_blocks[d] for d in ('target', 'ssd')],
        [comparisons[d]['aggregate']['candidate'] > comparisons[d]['aggregate']['random']['median'] for d in ('target', 'ssd')],
        [better_blocks[d] for d in ('target', 'ssd')],
        [contrasts['aggregate'][d][HALF]['no_adapt']['clean_AP_delta'] for d in detectors],
        clean['phi_norm_3']['mean'], clean['full_phi3']['mean']) if full else None
    analysis = {'full_cohort': full, 'criteria': criteria, 'half_dose_supported': all(criteria.values()) if full else None,
                'contrasts': contrasts, 'safety': safety, 'random_controls': comparisons,
                'positive_blocks': positive_blocks, 'blocks_above_random_median': better_blocks}
    (study/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n', encoding='utf-8')
    for name, rows in [('AP_tables.csv', ap_rows), ('macro_tables.csv', macro_rows), ('paired_dose.csv', paired)]:
        with (study/name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# T012 fixed half-dose results', '', f'Full cohort: {full}; supported: {analysis["half_dose_supported"]}.',
        f'Criteria: {criteria}', '', '| Detector | Macro delta no-adapt | Macro delta full | Random median | Positive blocks | Above random block median | Clean AP delta |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in detectors:
        q = contrasts['aggregate'][d][HALF]
        lines.append(f"| {d} | {q['no_adapt']['macro_corruption_AP_delta']:+.6f} | {q[FULL]['macro_corruption_AP_delta']:+.6f} | {comparisons[d]['aggregate']['random']['median']:+.6f} | {positive_blocks[d]}/{len(metrics)-1} | {better_blocks[d]}/{len(metrics)-1} | {q['no_adapt']['clean_AP_delta']:+.6f} |")
    lines += ['', f"Mean clean phi3 half/full: {clean['phi_norm_3']['mean']:.9f}/{clean['full_phi3']['mean']:.9f}; ratio {clean['ratio_of_mean_phi3']}.", '',
              'All condition/block AP/AP50/AP75 and paired deltas: AP_tables.csv. All macro signs/clean deltas: macro_tables.csv. All paired dose ratios: paired_dose.csv. All safety/step distributions and random-control comparisons: analysis.json.', '',
              'Fixed development cohort; no AP confidence intervals, no alpha search. Continuous half dose is not half compute. Smoke criteria are unassessed.', '']
    (study/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    receipt = {'images': len(env['evaluated_image_ids']), 'samples': len(samples), 'AP_rows': len(ap_rows), 'reused_endpoint_evaluations': anchor_checks,
               'original_supports_exactly_equal_T009': support_matches, 'all_half_algebra_and_identity_checks_passed': True,
               'references_verified': pins['full' if full else 'smoke'], 'T011_verified': pins['T011' if full else 'T011_smoke'],
               'prediction_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (study/'predictions').glob('*.json')},
               'hashes': {n: hashlib.sha256((study/n).read_bytes()).hexdigest() for n in ('samples.jsonl', 'metrics.json', 'analysis.json', 'AP_tables.csv', 'macro_tables.csv', 'paired_dose.csv', 'results.md')}}
    (study/'report_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'full_cohort': full, 'AP_rows': len(ap_rows), 'half_dose_supported': analysis['half_dose_supported'], 'criteria': criteria}), flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('study', 'reference', 'controls'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    report(a.study, a.reference, a.controls)


if __name__ == '__main__':
    main()
