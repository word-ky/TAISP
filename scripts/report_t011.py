"""T011 matched-control distributions and the predeclared beyond-thinning test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_t010 import promising_signal
from taisp.analysis.random_controls import CANDIDATE, randomization, selection_information


def report(study, prepared, candidate_study):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    meta, grid, safety = (read(prepared/n) for n in ('manifest.json', 'grid.json', 'safety.json'))
    completion = read(study/'completion.json')
    panels = {}
    for item in completion['panels']:
        path = study/'panels'/item['panel']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        p = read(path)
        key = p['detector'], p['case']
        panels.setdefault(key, {}).update(p['configs'])
    cases, groups, detectors = meta['cases'], list(meta['groups']), ['source', 'target', 'ssd']
    corruptions = [c for c in cases if c != 'clean_s0']
    rows = [json.loads(s) for s in (prepared/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
    masks = np.load(prepared/'decisions.npz')['selected']
    anchor_checks = 0
    for detector in detectors:
        for case in cases:
            configs = panels[detector, case]
            assert len(configs) == len(grid)
            reference = read(candidate_study/'panels'/f'{detector}_{case}.json')['configs']
            for anchor in ('no_adapt', 'full_hybrid', CANDIDATE):
                assert configs[anchor]['composed_prediction_sha256'] == reference[anchor]['composed_prediction_sha256']
                assert configs[anchor]['evaluations'] == reference[anchor]['evaluations']
                anchor_checks += len(configs[anchor]['evaluations'])
            indexes = [j for j, r in enumerate(rows) if r['case'] == case]
            for config, mask in zip(grid, masks):
                selected = sorted(rows[j]['image_id'] for j in indexes if mask[j])
                assert configs[config['name']]['selected_image_ids'] == selected
                assert configs[config['name']]['decision_sha256'] == config['decision_sha256']
    summaries, ap_rows, macro_rows = {}, [], []
    for config in grid:
        name = config['name']
        summaries[name] = {}
        for group in groups:
            summaries[name][group] = {}
            for detector in detectors:
                by_case = {}
                for case in cases:
                    configs = panels[detector, case]
                    values = configs[name]['evaluations'][group]
                    r = {'config': name, 'role': config['role'], 'group': group, 'detector': detector, 'case': case,
                         **{k: 100*values[k] for k in ('AP', 'AP50', 'AP75')}}
                    for ref in ('no_adapt', 'full_hybrid'):
                        baseline = configs[ref]['evaluations'][group]
                        r.update({f'{k}_delta_{ref}': 100*(values[k]-baseline[k]) for k in ('AP', 'AP50', 'AP75')})
                    by_case[case] = r
                    ap_rows.append(r)
                q = {}
                for ref in ('no_adapt', 'full_hybrid'):
                    for metric in ('AP', 'AP50', 'AP75'):
                        deltas = [by_case[c][metric+'_delta_'+ref] for c in corruptions]
                        q[f'macro_{metric}_delta_{ref}'] = sum(deltas)/len(deltas)
                        q[f'clean_{metric}_delta_{ref}'] = by_case['clean_s0'][metric+'_delta_'+ref]
                    q[f'positive_conditions_{ref}'] = sum(by_case[c]['AP_delta_'+ref] > 0 for c in corruptions)
                summaries[name][group][detector] = q
                macro_rows.append({'config': name, 'role': config['role'], 'group': group, 'detector': detector, **q})
    full = meta['full_cohort'] and len(groups) == 6
    config_rows = []
    random_names = [c['name'] for c in grid if c['role'] == 'random']
    for config in grid:
        name = config['name']
        result = summaries[name]
        positive = [sum(result[g][d]['macro_AP_delta_no_adapt'] > 0 for g in groups if g != 'aggregate') for d in ('target', 'ssd')]
        clean = safety[name]['aggregate']['clean_s0']
        corrupted = safety[name]['aggregate']['corrupted_overall']
        q = {'config': name, 'role': config['role'], 'seed': config.get('seed'), 'clean_coverage': clean['coverage'],
             'corruption_coverage': corrupted['coverage'], 'clean_effective_phi3': clean['effective_phi3']['mean'],
             'clean_nonzero_phi_fraction': clean['nonzero_phi_fraction'],
             'clean_latency_estimate_seconds': clean['latency_estimate_seconds']['mean'],
             'corruption_latency_estimate_seconds': corrupted['latency_estimate_seconds']['mean'],
             'target_positive_blocks': positive[0], 'ssd_positive_blocks': positive[1]}
        for detector in detectors:
            q.update({f'{detector}_{k}': v for k, v in result['aggregate'][detector].items()})
        q['R012_rule_met'] = promising_signal(clean['coverage'],
            [result['aggregate'][d]['clean_AP_delta_no_adapt'] for d in detectors],
            [result['aggregate'][d]['macro_AP_delta_no_adapt'] for d in ('target', 'ssd')], positive) if full else None
        config_rows.append(q)
    candidate = summaries[CANDIDATE]
    comparisons, blocks, beats_median = {}, {}, {}
    for detector in detectors:
        comparisons[detector] = {metric: randomization(value, [summaries[n]['aggregate'][detector][metric] for n in random_names])
            for metric, value in candidate['aggregate'][detector].items() if metric.startswith(('macro_', 'clean_'))}
        blocks[detector] = {group: randomization(candidate[group][detector]['macro_AP_delta_no_adapt'],
                [summaries[n][group][detector]['macro_AP_delta_no_adapt'] for n in random_names])
                for group in groups if group != 'aggregate'}
        beats_median[detector] = sum(r['candidate'] > r['random']['median'] for r in blocks[detector].values())
    candidate_deltas = [candidate['aggregate'][d]['macro_AP_delta_no_adapt'] for d in ('target', 'ssd')]
    q95 = [comparisons[d]['macro_AP_delta_no_adapt']['random']['p95'] for d in ('target', 'ssd')]
    clean_deltas = [candidate['aggregate'][d]['clean_AP_delta_no_adapt'] for d in detectors]
    supported = selection_information(candidate_deltas, q95, [beats_median[d] for d in ('target', 'ssd')], clean_deltas) if full else None
    pass_count = sum(r['R012_rule_met'] is True for r in config_rows if r['role'] == 'random') if full else None
    cost = {}
    candidate_row = next(r for r in config_rows if r['config'] == CANDIDATE)
    for field in ('clean_effective_phi3', 'clean_nonzero_phi_fraction', 'clean_latency_estimate_seconds', 'corruption_latency_estimate_seconds'):
        cost[field] = randomization(candidate_row[field], [r[field] for r in config_rows if r['role'] == 'random'])
    for filename, data in [('AP_tables.csv', ap_rows), ('macro_tables.csv', macro_rows), ('config_summary.csv', config_rows)]:
        with (study/filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    analysis = {'full_cohort': full, 'candidate': CANDIDATE, 'random_draws': len(random_names),
        'selection_information_beyond_thinning': supported, 'aggregate_distributions': comparisons,
        'block_distributions': blocks, 'candidate_beats_block_random_median': beats_median,
        'random_R012_pass_count': pass_count, 'random_R012_pass_fraction': pass_count/len(random_names) if full else None,
        'descriptive_safety_cost': cost, 'configurations': config_rows, 'summaries': summaries,
        'interpretation': 'matched-selector randomization distribution, not a bootstrap or AP confidence interval; development cohort only'}
    (study/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n', encoding='utf-8')
    lines = ['# T011 matched-random falsification', '',
        f"Images: {len(meta['image_ids'])}; configurations: {len(grid)}; random draws: {len(random_names)}; AP rows: {len(ap_rows)}.",
        f'Full cohort: {full}. Selection information beyond thinning: {supported}.', '',
        f'Random selectors satisfying R012: {pass_count}/{len(random_names)} (unassessed on smoke).', '',
        'All deltas are AP points. Percentiles are strict empirical ranks; corrected upper tail counts ties as >=. These are control distributions, not AP confidence intervals.', '',
        '| Detector | Candidate macro AP | Random mean / median / p05 / p95 | Candidate percentile | Upper tail | Candidate > block medians |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in detectors:
        r = comparisons[d]['macro_AP_delta_no_adapt']
        lines.append(f"| {d} | {r['candidate']:+.6f} | "+' / '.join(f"{r['random'][k]:+.6f}" for k in ('mean', 'median', 'p05', 'p95'))+
                     f" | {r['strict_empirical_percentile']:.2f}% | {r['one_sided_tail']:.6f} | {beats_median[d]}/{len(groups)-1} |")
    lines += ['', '## Every block', '', '| Detector | Block | Candidate | Random median / p05 / p95 | Candidate minus median |', '| --- | --- | ---: | ---: | ---: |']
    for d, values in blocks.items():
        for group, r in values.items():
            lines.append(f"| {d} | {group} | {r['candidate']:+.6f} | "+' / '.join(f"{r['random'][k]:+.6f}" for k in ('median', 'p05', 'p95'))+f" | {r['candidate']-r['random']['median']:+.6f} |")
    lines += ['', 'All 200 draws and three anchors: config_summary.csv. All condition/group AP/AP50/AP75 and both reference deltas: AP_tables.csv. All block macros and clean deltas: macro_tables.csv. Candidate-minus-control distributions for macro and clean AP/AP50/AP75, exact ties, all tail probabilities, and descriptive safety/cost are in analysis.json.', '',
              'Controls exactly match selected counts within condition×block and are not deployment rules. The candidate was selected in T010; even a passing test remains developmental. No model rerun, threshold change, alternative candidate, new cohort or meta-training was performed.', '']
    (study/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    receipt = {'AP_rows': len(ap_rows), 'macro_rows': len(macro_rows), 'configurations': len(grid),
        'candidate_and_endpoint_evaluations_exactly_match_T010': anchor_checks,
        'shared_detector_case_decisions_verified': len(grid)*len(detectors)*len(cases),
        'AP_rows_match_completion': len(ap_rows) == completion['official_evaluations'],
        'candidate_study': str(candidate_study),
        'hashes': {n: hashlib.sha256((study/n).read_bytes()).hexdigest() for n in
                   ('completion.json', 'AP_tables.csv', 'macro_tables.csv', 'config_summary.csv', 'analysis.json', 'results.md')}}
    (study/'report_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'full_cohort': full, 'AP_rows': len(ap_rows), 'selection_information_beyond_thinning': supported,
                      'random_R012_pass_count': pass_count}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--candidate-study', type=Path, required=True)
    a = p.parse_args()
    report(a.study, a.prepared, a.candidate_study)


if __name__ == '__main__':
    main()
