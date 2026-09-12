"""All fixed T010 configurations and predeclared developmental feasibility rule."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def promising_signal(clean_coverage, clean_deltas, target_deltas, positive_blocks):
    return (clean_coverage <= .5 and min(clean_deltas) >= -.10 and
            min(target_deltas) > 0 and min(positive_blocks) >= 4)


def report(study, prepared):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    meta, grid, safety = (read(prepared/n) for n in ('manifest.json', 'grid.json', 'safety.json'))
    complete = read(study/'completion.json')
    panels = {(p['detector'], p['case']): p for p in (read(path) for path in sorted((study/'panels').glob('*.json')))}
    detectors, groups, cases = ['source', 'target', 'ssd'], list(meta['groups']), meta['cases']
    corruptions = [c for c in cases if c != 'clean_s0']
    assert len(panels) == len(detectors)*len(cases)
    score_rows = [json.loads(s) for s in (prepared/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
    masks = np.load(prepared/'decisions.npz')['selected']
    for config, mask in zip(grid, masks):
        for case in cases:
            selected = sorted(r['image_id'] for j, r in enumerate(score_rows) if r['case'] == case and mask[j])
            for detector in detectors:
                result = panels[detector, case]['configs'][config['name']]
                assert result['selected_image_ids'] == selected
                assert result['decision_sha256'] == config['decision_sha256']
    ap_rows, macro_rows, summaries = [], [], {}
    for config in grid:
        name = config['name']
        summaries[name] = {}
        for group in groups:
            summaries[name][group] = {}
            for detector in detectors:
                by_case = {}
                for case in cases:
                    configs = panels[detector, case]['configs']
                    values = configs[name]['evaluations'][group]
                    row = {'config': name, 'group': group, 'detector': detector, 'case': case,
                           **{k: 100*values[k] for k in ('AP', 'AP50', 'AP75')}}
                    for reference in ('no_adapt', 'full_hybrid'):
                        baseline = configs[reference]['evaluations'][group]
                        row.update({f'{k}_delta_{reference}': 100*(values[k]-baseline[k]) for k in ('AP', 'AP50', 'AP75')})
                    ap_rows.append(row)
                    by_case[case] = row
                q = {}
                for reference in ('no_adapt', 'full_hybrid'):
                    deltas = [by_case[c]['AP_delta_'+reference] for c in corruptions]
                    q.update({f'macro_AP_delta_{reference}': sum(deltas)/len(deltas),
                              f'positive_conditions_{reference}': sum(v > 0 for v in deltas),
                              f'clean_AP_delta_{reference}': by_case['clean_s0']['AP_delta_'+reference]})
                summaries[name][group][detector] = q
                macro_rows.append({'config': name, 'group': group, 'detector': detector, **q})
    full = meta['full_cohort'] and len(groups) == 6
    config_rows, candidates = [], []
    for config in grid:
        name = config['name']
        result, panel = summaries[name], safety[name]['aggregate']
        clean_deltas = [result['aggregate'][d]['clean_AP_delta_no_adapt'] for d in detectors]
        target_deltas = [result['aggregate'][d]['macro_AP_delta_no_adapt'] for d in ('target', 'ssd')]
        positive = [sum(result[g][d]['macro_AP_delta_no_adapt'] > 0 for g in groups if g != 'aggregate') for d in ('target', 'ssd')]
        passes = promising_signal(panel['clean_s0']['coverage'], clean_deltas, target_deltas, positive) if full else None
        if passes:
            candidates.append(name)
        row = {'config': name, 'score': config['score'], 'orientation': config['orientation'],
               'nominal_coverage': config['coverage'], 'clean_coverage': panel['clean_s0']['coverage'],
               'corruption_coverage': panel['corrupted_overall']['coverage'],
               'clean_effective_phi3': panel['clean_s0']['effective_phi3']['mean'],
               'clean_nonzero_phi_fraction': panel['clean_s0']['nonzero_phi_fraction'],
               'clean_latency_estimate_seconds': panel['clean_s0']['latency_estimate_seconds']['mean'],
               'corruption_latency_estimate_seconds': panel['corrupted_overall']['latency_estimate_seconds']['mean'],
               'promising_rule_met': passes}
        for detector in detectors:
            row.update({f'{detector}_{k}': v for k, v in result['aggregate'][detector].items()})
        row.update(target_positive_blocks=positive[0], ssd_positive_blocks=positive[1])
        config_rows.append(row)
    for filename, rows in [('AP_tables.csv', ap_rows), ('macro_tables.csv', macro_rows), ('config_summary.csv', config_rows)]:
        with (study/filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    analysis = {'full_cohort': full, 'summaries': summaries, 'configurations': config_rows,
                'promising_configurations': candidates if full else None,
                'decision_source_target_ssd_identity_audited': True,
                'interpretation': 'developmental full-grid exploration; no independent gate validation or formal AP intervals'}
    (study/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n', encoding='utf-8')
    lines = ['# T010 frozen offline need-to-adapt grid', '',
             f"Images: {len(meta['image_ids'])}; observations: {meta['rows']}; configurations: {len(grid)}; official AP rows: {len(ap_rows)}.",
             f"Full cohort: {full}. Predeclared promising-rule matches: {candidates if full else 'not assessed on smoke' }.", '',
             'All deltas are AP points. This is development-set feasibility, not independent gate validation.', '',
             '| Configuration | Clean / corrupt coverage % | FCOS macro delta no-adapt / full | SSD macro delta no-adapt / full | Positive blocks FCOS / SSD | Clean delta source / FCOS / SSD | Clean effective phi3 | Rule |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in config_rows:
        lines.append(f"| {r['config']} | {100*r['clean_coverage']:.1f} / {100*r['corruption_coverage']:.1f} | "+
            ' / '.join(f"{r['target_macro_AP_delta_'+ref]:+.6f}" for ref in ('no_adapt', 'full_hybrid'))+' | '+
            ' / '.join(f"{r['ssd_macro_AP_delta_'+ref]:+.6f}" for ref in ('no_adapt', 'full_hybrid'))+
            f" | {r['target_positive_blocks']}/{len(groups)-1} / {r['ssd_positive_blocks']}/{len(groups)-1} | "+
            ' / '.join(f"{r[d+'_clean_AP_delta_no_adapt']:+.6f}" for d in detectors)+f" | {r['clean_effective_phi3']:.6f} | {r['promising_rule_met']} |")
    lines.extend(['', 'AP_tables.csv preserves every detector/condition/block AP/AP50/AP75 and both reference deltas. macro_tables.csv preserves all macro/sign/clean results. config_summary.csv includes all receipt-derived latency estimates. The fixed preparation/safety.json preserves all clean/corruption/family/block coverage and effective-phi distributions.', '',
                  'Latency estimates use saved source-setup and hybrid timing, not a new benchmark. Non-support scalar scores retain the joint source+CLIP identity step cost even on skipped images because separate scalar computation costs were not recorded. Target inference, IO and cohort ranking are excluded. No timing speedup claim is supported.', ''])
    (study/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    receipt = {'official_AP_rows': len(ap_rows), 'macro_rows': len(macro_rows), 'configurations': len(config_rows),
               'shared_decisions_verified': len(grid)*len(cases)*len(detectors),
               'all_fixed_configurations_retained': len(config_rows) == meta['configurations'],
               'completion_official_evaluations_match': len(ap_rows) == complete['official_evaluations'],
               'hashes': {n: hashlib.sha256((study/n).read_bytes()).hexdigest() for n in
                          ('completion.json', 'AP_tables.csv', 'macro_tables.csv', 'config_summary.csv', 'analysis.json', 'results.md')}}
    (study/'report_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'full_cohort': full, 'AP_rows': len(ap_rows), 'promising_configurations': analysis['promising_configurations']}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    a = p.parse_args()
    report(a.study, a.prepared)


if __name__ == '__main__':
    main()
