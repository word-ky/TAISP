"""Paired cross-detector transfer, joint benefits and severity diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_t002_linear import describe
from scripts.report_t003 import paired_delta, interval
from scripts.report_t005 import basics, paired, distribution, finite_json


def detector_rows(rows, detector, matched=False):
    output = []
    for row in rows:
        d = row[detector]
        output.append({**row, **d, 'g_sem': row['norm_matched']['gradient'] if matched else row['g_sem'],
                       'det_loss_delta_sem1': d['norm_matched']['det_loss_delta'] if matched else d['det_loss_delta_sem1']})
    return output


def joint_outcome(row, matched=False):
    values = [row[d]['norm_matched']['det_loss_delta'] if matched else row[d]['det_loss_delta_sem1']
              for d in ('source', 'target')]
    source, target = [v < 0 for v in values]
    return 'both' if source and target else 'source_only' if source else 'target_only' if target else 'neither'


def summarize(rows, reference, lr):
    result = {'count': len(rows), 'detectors': {}, 'joint': {}}
    for d in ('source', 'target'):
        raw, ref = detector_rows(rows, d), detector_rows(reference, d)
        matched, mr = detector_rows(rows, d, True), detector_rows(reference, d, True)
        g = {**basics(raw), 'paired_vs_clip': paired(raw, ref),
             'loss_delta3': distribution([r['det_loss_delta_sem3'] for r in raw]),
             'benefit_fraction3_all': float(np.mean([r['det_loss_delta_sem3'] < 0 for r in raw])),
             'norm_matched': {**basics(matched), 'paired_vs_clip': paired(matched, mr)},
             'coordinate_dot_mean': np.array([r[d]['coordinate_dot'] for r in rows]).mean(0).tolist(),
             'coordinate_dot_positive_fraction': (np.array([r[d]['coordinate_dot'] for r in rows]) > 0).mean(0).tolist(),
             'matched_coordinate_dot_mean': np.array([r[d]['norm_matched']['coordinate_dot'] for r in rows]).mean(0).tolist()}
        for name, chosen in (('raw_taylor', raw), ('matched_taylor', matched)):
            g[name], _ = describe([{**r, 'gradient_cosine': r['gradient_cosine'] if r['gradient_cosine'] is not None else 0.} for r in chosen], lr)
        result['detectors'][d] = g
    for name, matched in (('raw', False), ('norm_matched', True)):
        result['joint'][name] = {}
        for outcome in ('both', 'source_only', 'target_only', 'neither'):
            metric = lambda r: float(joint_outcome(r, matched) == outcome)
            result['joint'][name][outcome] = {'fraction': float(np.mean([metric(r) for r in rows])),
                                            'paired_vs_clip': paired_delta(rows, reference, metric)}
    agreement = lambda r: r['source_target_cosine'] if r['source_target_cosine'] is not None else 0.
    result['source_target_agreement'] = {'cosine_valid': distribution([r['source_target_cosine'] for r in rows if r['source_target_cosine'] is not None]),
        'cosine_zero_coded': paired_delta(rows, [{**r, 'source_target_cosine': 0.} for r in rows], agreement),
        'positive_fraction_all': float(np.mean([agreement(r) > 0 for r in rows])),
        'coordinate_dot_mean': np.array([r['source_target_coordinate_dot'] for r in rows]).mean(0).tolist()}
    for field in ('phi_norm_1', 'phi_norm_3', 'g_sem_norm', 'saturation_before', 'saturation_1', 'saturation_3',
                  'semantic_loss_before', 'semantic_loss_1', 'semantic_loss_3', 'adapt_seconds_3',
                  'deploy_seconds_3', 'peak_allocated_mb', 'source_setup_seconds', 'source_setup_peak_mb', 'support_count'):
        result[field] = distribution([r[field] for r in rows])
    result['fallback_fraction'] = float(np.mean([r['no_update_fallback'] for r in rows]))
    result['original_support_confidence'] = distribution([s for r in rows for s in r['support']['scores']])
    result['phi_trajectory_mean'] = np.array([[d['phi'] for d in r['diagnostics']] for r in rows]).mean(0).tolist()
    result['physical_change3_mean_abs'] = np.abs([r['physical_change_3'] for r in rows]).mean(0).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    study = parser.parse_args().study
    rows = [json.loads(line) for line in (study/'samples.jsonl').read_text().splitlines()]
    env = json.loads((study/'environment.json').read_text())
    variants, lr = env['config']['variants'], env['config']['semantic_lr']
    cases = list(json.loads((study/'summary.json').read_text())[variants[0]])
    metrics = json.loads((study/'metrics.json').read_text())
    groups = {}
    for case in ['corrupted_overall', 'severity_s1', 'severity_s2']+cases:
        if case == 'corrupted_overall':
            chosen = [r for r in rows if r['family'] != 'clean']
        elif case.startswith('severity_'):
            chosen = [r for r in rows if r['severity'] == int(case[-1])]
        else:
            chosen = [r for r in rows if f"{r['family']}_s{r['severity']}" == case]
        ref = [r for r in chosen if r['variant'] == 'global_generic']
        groups[case] = {}
        for variant in variants:
            vr = [r for r in chosen if r['variant'] == variant]
            g = summarize(vr, ref, lr)
            if case in cases:
                g['AP'] = {d: {metric: {str(k): {'value': 100*metrics[f'{d}_{case}_{variant}{k}'][metric],
                     'delta_before': 100*(metrics[f'{d}_{case}_{variant}{k}'][metric]-metrics[f'{d}_{case}_before'][metric]),
                     'delta_clip': 100*(metrics[f'{d}_{case}_{variant}{k}'][metric]-metrics[f'{d}_{case}_global_generic{k}'][metric])}
                     for k in (1, 3)} for metric in ('AP', 'AP50', 'AP75')} for d in ('source', 'target')}
            groups[case][variant] = g
        print(f'Analyzed {case}', flush=True)
    severity_ap = {}
    for s in (1, 2):
        selected = [c for c in cases if c.endswith(f'_s{s}')]
        severity_ap[str(s)] = {d: {v: {str(k): {metric: {field: float(np.mean([groups[c][v]['AP'][d][metric][str(k)][field] for c in selected]))
                       for field in ('value', 'delta_before', 'delta_clip')} for metric in ('AP', 'AP50', 'AP75')}
                       for k in (1, 3)} for v in variants} for d in ('source', 'target')}
    payload = {'source_revision': env['source_revision'], 'groups': groups, 'severity_AP_macro': severity_ap,
               'bootstrap': '2000 paired image-cluster percentile draws seed20260912; all selected cases perimage stay together; clean separate; exploratory no multiplicity adjustment',
               'zero_rule': 'Raw undefined cosine=null; zero-coded explicitly for aggregate paired cosine/rates. Fallbacks remain in all denominators. Constant correlations=null.',
               'AP_rule': 'Official percase subset AP/AP50/AP75; severity macro is arithmetic mean of3condition metrics, not pooled COCO AP; no AP CIs.',
               'loss_units': 'Source and target native losses have different scales; signed coordinate products are diagnostic, not comparable magnitudes across detectors.'}
    (study/'analysis.json').write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False)+'\n')
    lines = ['# T006 cross-detector transfer results', '',
             f"Source {env['source_revision']}; {len(env['evaluated_image_ids'])} images, {len(rows)} variant observations, smoke={env['smoke_limit']}.", '',
             'Source-only frozen Faster R-CNN fixed ROI pseudo-confidence. Target FCOS is analysis/evaluation only. '
             'Both detectors evaluate identical enhanced images. Global8D ISP,lr0.1,K3,identity initialization, '
             'hard clamp. No target in support/update and no oracle information in norm matching.', '',
             '## Source / target official AP1 / AP3 (0–100 points)', '',
             '| Detector | Case | Before AP / AP50 / AP75 | CLIP AP1 /3 | Pseudo AP1 /3 | Pseudo AP50 1 /3 | Pseudo AP75 1 /3 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for d in ('source', 'target'):
        for c in cases:
            base = ' / '.join(f"{100*metrics[f'{d}_{c}_before'][m]:.3f}" for m in ('AP', 'AP50', 'AP75'))
            cell = lambda v, m: ' / '.join(f"{100*metrics[f'{d}_{c}_{v}{k}'][m]:.3f}" for k in (1, 3))
            lines.append(f"| {d} | {c} | {base} | {cell('global_generic','AP')} | {cell('det_pseudo','AP')} | {cell('det_pseudo','AP50')} | {cell('det_pseudo','AP75')} |")
    lines += ['', 'CLIP AP50/AP75 and all before/CLIP deltas are retained in analysis.json; official aggregate AP has no invented per-image intervals.', '',
              '## Paired mechanism vs CLIP', '',
              'All episodes included. *Undefined cosine zero-coded here; valid-only distributions retained separately.95% intervals use2000 paired image-cluster draws.', '',
              '| Group | Detector | Variant | Cosine* | Positive% | Raw benefit% | Matched benefit% | Delta cosine [CI] | Delta raw benefit pp [CI] | Delta matched benefit pp [CI] |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c, vg in groups.items():
        for d in ('source', 'target'):
            for v in variants:
                g = vg[v]['detectors'][d]
                p, n = g['paired_vs_clip'], g['norm_matched']
                lines.append(f"| {c} | {d} | {v} | {g['cosine_zero_coded']['mean']:.5f} | {100*g['positive_fraction_all']:.2f} | {100*g['benefit_fraction_all']:.2f} | {100*n['benefit_fraction_all']:.2f} | {interval(p['cosine_zero_coded'])} | {interval(p['benefit_rate'],100)} | {interval(n['paired_vs_clip']['benefit_rate'],100)} |")
    lines += ['', '## Detector agreement and joint pseudo-step outcomes', '',
              '| Group | Source-target cosine [CI] | Agreement positive% | Raw both / source-only / target-only / neither% | Matched both / source-only / target-only / neither% |',
              '|---|---:|---:|---:|---:|']
    for c, vg in groups.items():
        g = vg['det_pseudo']; a = g['source_target_agreement']
        cells = [' / '.join(f"{100*g['joint'][kind][s]['fraction']:.2f}" for s in ('both', 'source_only', 'target_only', 'neither')) for kind in ('raw', 'norm_matched')]
        lines.append(f"| {c} | {interval(a['cosine_zero_coded'])} | {100*a['positive_fraction_all']:.2f} | "+' | '.join(cells)+' |')
    lines += ['', '## Severity: macro-average of3 condition AP deltas for pseudo', '',
              'Descriptive arithmetic means of official condition AP; not pooled detections or new COCO AP.', '',
              '| Severity | Detector | DeltaAP1 /3 vs before | DeltaAP1 /3 vs CLIP |', '|---|---|---:|---:|']
    for s, detectors in severity_ap.items():
        for d, vg in detectors.items():
            g = vg['det_pseudo']
            cells = [' / '.join(f"{g[str(k)]['AP'][f]:+.3f}" for k in (1, 3)) for f in ('delta_before', 'delta_clip')]
            lines.append(f'| s{s} | {d} | '+' | '.join(cells)+' |')
    lines += ['', '## Clean control and resource costs', '',
              '| Variant | Clean phi norm1 /3 | Clean source deltaAP3 | Clean target deltaAP3 | Corrupted adapt / deploy sec3 | Adapt peak MiB | Fallback% |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        c, g = groups['clean_s0'][v], groups['corrupted_overall'][v]
        lines.append(f"| {v} | {c['phi_norm_1']['mean']:.5f} / {c['phi_norm_3']['mean']:.5f} | {c['AP']['source']['AP']['3']['delta_before']:+.3f} | {c['AP']['target']['AP']['3']['delta_before']:+.3f} | {g['adapt_seconds_3']['mean']:.4f} / {g['deploy_seconds_3']['mean']:.4f} | {g['peak_allocated_mb']['mean']:.1f} | {100*g['fallback_fraction']:.2f} |")
    lines += ['', 'Full signed-coordinate dot products, source-target agreement, Taylor correlations/predictions, '
              'raw/matched paired mean-loss intervals, joint-outcome paired intervals, support/confidence, '
              'phi trajectories and saturation are in analysis.json. Both models and CLIP are resident during '
              'memory measurements; target evaluation is excluded from deployment timing. No family is dropped.', '',
              '![Pseudo AP3 changes on source and target](transfer_ap.png)', '']
    (study/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    values = np.array([[groups[c]['det_pseudo']['AP'][d]['AP']['3'][field] for d in ('source', 'target') for field in ('delta_before', 'delta_clip')] for c in cases])
    bound = max(abs(values).max(), .01)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    im = ax.imshow(values, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(4), ['Source vs before', 'Source vs CLIP', 'FCOS vs before', 'FCOS vs CLIP'])
    ax.set_yticks(range(len(cases)), cases)
    ax.set_title('T006: pseudo-confidence AP change after 3 steps')
    for i in range(len(cases)):
        for j in range(4):
            ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center', color='white' if abs(values[i,j]) > bound*.6 else 'black')
    fig.colorbar(im, ax=ax, label='AP points')
    fig.savefig(study/'transfer_ap.png', dpi=180)
    fig.savefig(study/'transfer_ap.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
