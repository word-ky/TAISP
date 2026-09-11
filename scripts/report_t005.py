"""T005 paired signal screen, including empty-support episodes and clean controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_t003 import paired_delta, interval
from scripts.analyze_t002_linear import describe


def zero_cos(row):
    return row['gradient_cosine'] if row['gradient_cosine'] is not None else 0.


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'mean': float(a.mean()),
            'p05': float(np.quantile(a, .05)), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95))} if len(a) else {'count': 0}


def paired(rows, reference):
    return {'cosine_zero_coded': paired_delta(rows, reference, zero_cos),
            'positive_rate': paired_delta(rows, reference, lambda r: float(zero_cos(r) > 0)),
            'benefit_rate': paired_delta(rows, reference, lambda r: float(r['det_loss_delta_sem1'] < 0)),
            'loss_delta1': paired_delta(rows, reference, lambda r: r['det_loss_delta_sem1'])}


def norm_rows(rows):
    return [{**r, 'g_sem': r['norm_matched']['gradient'],
             'det_loss_delta_sem1': r['norm_matched']['det_loss_delta']} for r in rows]


def basics(rows):
    gradients = np.array([r['g_sem'] for r in rows])
    squares = gradients**2
    sums = squares.sum(1, keepdims=True)
    energy = np.divide(squares, sums, out=np.zeros_like(squares), where=sums > 0)
    return {'count': len(rows), 'cosine_valid': distribution([r['gradient_cosine'] for r in rows if r['gradient_cosine'] is not None]),
            'cosine_zero_coded': distribution([zero_cos(r) for r in rows]),
            'positive_fraction_all': float(np.mean([zero_cos(r) > 0 for r in rows])),
            'benefit_fraction_all': float(np.mean([r['det_loss_delta_sem1'] < 0 for r in rows])),
            'loss_delta1': distribution([r['det_loss_delta_sem1'] for r in rows]),
            'gradient_norm': distribution(np.linalg.norm(gradients, axis=1)),
            'coordinate_energy_mean_zero_for_empty': energy.mean(0).tolist(),
            'gradient_mean': gradients.mean(0).tolist(),
            'zero_gradient_count': int((sums == 0).sum())}


def support_summary(rows):
    nbase = np.array([len(r['support']['base']['scores']) for r in rows])
    nflip = np.array([len(r['support']['flipped']['scores']) for r in rows])
    nstable = np.array([len(r['support']['stable']['scores']) for r in rows])
    ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    result = {'base_count': distribution(nbase), 'flip_count': distribution(nflip),
              'stable_count': distribution(nstable),
              'stable_over_base_mean_zero_if_empty': float(ratio(nstable, nbase).mean()),
              'symmetric_match_rate_mean_zero_if_empty': float(ratio(2*nstable, nbase+nflip).mean()),
              'fallback_fraction': float(np.mean([r['no_update_fallback'] for r in rows])),
              'confidence': {kind: distribution([s for r in rows for s in r['support'][kind]['scores']])
                             for kind in ('base', 'flipped', 'stable')}, 'strata': {}}
    for label, lo, hi in (('0', 0, 0), ('1-2', 1, 2), ('3-5', 3, 5), ('6+', 6, 20)):
        selected = [r for r in rows if lo <= r['support_count'] <= hi]
        if selected:
            result['strata'][label] = {'raw': basics(selected), 'norm_matched': basics(norm_rows(selected))}
    return result


def summarize(rows, reference, lr):
    result = basics(rows)
    result['paired_vs_global'] = paired(rows, reference)
    nr = norm_rows(rows)
    result['norm_matched'] = {**basics(nr), 'paired_vs_global': paired(nr, norm_rows(reference))}
    for name, chosen in (('raw_taylor', rows), ('matched_taylor', nr)):
        # Undefined zero-gradient cosine is explicitly zero-coded for this
        # all-observation diagnostic. Constant-correlation outputs become null.
        result[name], _ = describe([{**r, 'gradient_cosine': zero_cos(r)} for r in chosen], lr)
    for field in ('phi_norm_1', 'phi_norm_3', 'semantic_loss_before', 'semantic_loss_1', 'semantic_loss_3',
                  'saturation_before', 'saturation_1', 'saturation_3', 'det_loss_delta_sem3',
                  'adapt_seconds_3', 'deploy_seconds_3', 'peak_allocated_mb'):
        result[field] = distribution([r[field] for r in rows])
    result['setup'] = {k: distribution([r['setup'][k] for r in rows]) for k in rows[0]['setup']}
    result['own_loss_decreases3_fraction'] = float(np.mean([r['semantic_loss_3'] < r['semantic_loss_before'] for r in rows]))
    result['phi_trajectory_mean'] = np.array([[d['phi'] for d in r['diagnostics']] for r in rows]).mean(0).tolist()
    result['physical_change3_mean_abs'] = np.abs([r['physical_change_3'] for r in rows]).mean(0).tolist()
    if rows[0]['support_count'] is not None:
        result['support'] = support_summary(rows)
        result['ce_js'] = {k: np.array([r['native_ce_js_0_1_3'][k] for r in rows]).mean(0).tolist() for k in ('0', '1', '3')}
    return result


def finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


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
    for case in ['corrupted_overall']+cases:
        chosen = [r for r in rows if (r['family'] != 'clean' if case == 'corrupted_overall' else f"{r['family']}_s{r['severity']}" == case)]
        ref = [r for r in chosen if r['variant'] == 'global_generic']
        groups[case] = {}
        for variant in variants:
            vr = [r for r in chosen if r['variant'] == variant]
            g = summarize(vr, ref, lr)
            if variant in ('det_stable', 'det_stable_js'):
                ablation = 'det_pseudo' if variant == 'det_stable' else 'det_stable'
                ar = [r for r in chosen if r['variant'] == ablation]
                g['paired_ablation'] = {'reference': ablation, 'raw': paired(vr, ar), 'norm_matched': paired(norm_rows(vr), norm_rows(ar))}
            if case != 'corrupted_overall':
                g['AP_delta_before'] = {str(k): 100*(metrics[f'{case}_{variant}{k}']['AP']-metrics[f'{case}_before']['AP']) for k in (1, 3)}
                g['AP_delta_global'] = {str(k): 100*(metrics[f'{case}_{variant}{k}']['AP']-metrics[f'{case}_global_generic{k}']['AP']) for k in (1, 3)}
            groups[case][variant] = g
        print(f'Analyzed {case}', flush=True)
    payload = {'source_revision': env['source_revision'], 'groups': groups,
               'bootstrap': '2000 paired image-cluster percentile draws seed20260912; corrupted overall keeps6cases together; clean separate; exploratory no multiplicity adjustment',
               'zero_rule': 'All episodes included; undefined raw cosine=null, zero-coded for paired cosine and positive-rate denominator. Valid-only cosine separately reported. Zero energy=0. Constant correlations=null.',
               'AP': 'Official subset AP on full support, not mean per-image AP; differences without AP CIs.'}
    (study/'analysis.json').write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False)+'\n')
    lines = ['# T005 detector-native signal screen', '',
             f"Source {env['source_revision']}; {len(env['evaluated_image_ids'])} images, {len(rows)} observations, smoke={env['smoke_limit']}.", '',
             'Frozen detector, fixed original base/flip ROI support; global8D ISP, lr0.1,K3, JS coefficient1.0. '
             'Fresh within-run annotated oracle only for analysis. Clean reported separately. Empty-support '
             'episodes retained with exact no update from ISP(phi0); undefined cosine zero-coded only for '
             'all-observation summaries and paired contrasts. Valid-only cosine in analysis.json.', '',
             '## AP1 / AP3 (0–100 subset points)', '',
             '| Case | Before | CLIP global | Pseudo confidence | Stable confidence | Stable + JS |',
             '|---|---:|---:|---:|---:|---:|']
    for case in cases:
        cells = [f"{100*metrics[f'{case}_{v}1']['AP']:.3f} / {100*metrics[f'{case}_{v}3']['AP']:.3f}" for v in variants]
        lines.append(f"| {case} | {100*metrics[f'{case}_before']['AP']:.3f} | "+' | '.join(cells)+' |')
    lines += ['', '## Mechanism and paired95% intervals', '',
              'Intervals use2000 paired image-cluster draws; no multiplicity adjustment. Frequencies include fallbacks.', '',
              '| Group | Variant | Mean cosine* | Positive% | Benefit% | Matched benefit% | Δcos vs CLIP [CI] | Δbenefit pp [CI] | Matched Δbenefit pp [CI] |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case, vg in groups.items():
        for v, g in vg.items():
            p, nm = g['paired_vs_global'], g['norm_matched']
            lines.append(f"| {case} | {v} | {g['cosine_zero_coded']['mean']:.5f} | {100*g['positive_fraction_all']:.2f} | {100*g['benefit_fraction_all']:.2f} | {100*nm['benefit_fraction_all']:.2f} | {interval(p['cosine_zero_coded'])} | {interval(p['benefit_rate'],100)} | {interval(nm['paired_vs_global']['benefit_rate'],100)} |")
    lines += ['', '*Undefined zero-gradient cosine contributes0 here; valid-only distributions and exact counts retained.', '',
              '## Clean control and deployment costs', '',
              '| Variant | Clean ΔAP1 / ΔAP3 | Clean mean phi norm1 /3 | Corrupted adapt / deploy seconds3 | Adapt peak MiB |',
              '|---|---:|---:|---:|---:|']
    for v in variants:
        c, g = groups['clean_s0'][v], groups['corrupted_overall'][v]
        lines.append(f"| {v} | {c['AP_delta_before']['1']:+.3f} / {c['AP_delta_before']['3']:+.3f} | {c['phi_norm_1']['mean']:.5f} / {c['phi_norm_3']['mean']:.5f} | {g['adapt_seconds_3']['mean']:.4f} / {g['deploy_seconds_3']['mean']:.4f} | {g['peak_allocated_mb']['mean']:.1f} |")
    lines += ['', 'Setup peaks are separately retained in analysis.json; timings include unchanged diagnostics, fixed variant order.', '',
              '## Support and fallback', '',
              '| Group | Variant | Base / stable count | Stable/base | Symmetric match | Fallback% |',
              '|---|---|---:|---:|---:|---:|']
    for case, vg in groups.items():
        for v in variants[1:]:
            s = vg[v]['support']
            lines.append(f"| {case} | {v} | {s['base_count']['mean']:.2f} / {s['stable_count']['mean']:.2f} | {s['stable_over_base_mean_zero_if_empty']:.3f} | {s['symmetric_match_rate_mean_zero_if_empty']:.3f} | {100*s['fallback_fraction']:.2f} |")
    lines += ['', '## Support-size outcomes', '',
              '| Group | Variant | Support | N | Mean cosine* | Benefit% | Matched benefit% | Mean loss delta1 |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for case, vg in groups.items():
        for v in variants[1:]:
            for name, s in vg[v]['support']['strata'].items():
                r, n = s['raw'], s['norm_matched']
                lines.append(f"| {case} | {v} | {name} | {r['count']} | {r['cosine_zero_coded']['mean']:.5f} | {100*r['benefit_fraction_all']:.2f} | {100*n['benefit_fraction_all']:.2f} | {r['loss_delta1']['mean']:.6g} |")
    lines += ['', 'Full per-case gradient norms/energy, confidence distributions, CE/JS components, '
              'Taylor predictions/correlations, mean-loss paired CIs, stable-minus-pseudo and JS-minus-stable '
              'paired effects, saturation and phi trajectories are in analysis.json. Negative families are retained.', '',
              '![Subset AP change at3steps](ap_change.png)', '']
    (study/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    values = np.array([[groups[c][v]['AP_delta_before']['3'] for v in variants] for c in cases])
    bound = max(abs(values).max(), .01)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    im = ax.imshow(values, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(4), ['CLIP global', 'Pseudo CE', 'Stable CE', 'Stable CE + JS'])
    ax.set_yticks(range(len(cases)), cases)
    ax.set_title('T005: subset AP change after 3 steps vs unadapted')
    for i in range(len(cases)):
        for j in range(4):
            ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center', color='white' if abs(values[i,j]) > bound*.6 else 'black')
    fig.colorbar(im, ax=ax, label='AP points')
    fig.savefig(study/'ap_change.png', dpi=180)
    fig.savefig(study/'ap_change.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
