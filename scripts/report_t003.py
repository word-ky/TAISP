"""T003 fixed-run paired mechanism, downstream, condition and saturation report."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_t002_linear import describe
from scripts.analyze_t003_coordinates import decomposition, saturation_groups


def paired_delta(rows, reference, metric, replicates=2000, seed=20260912):
    key = lambda r: (r['image_id'], r['family'], r['severity'])
    ref = {key(r): r for r in reference}
    values = np.array([metric(r)-metric(ref[key(r)]) for r in rows], dtype=float)
    ids = np.array([r['image_id'] for r in rows])
    clusters = np.array([np.flatnonzero(ids == i) for i in np.unique(ids)])
    selected = np.random.default_rng(seed).integers(0, len(clusters), (replicates, len(clusters)))
    boot = values[clusters[selected]].mean((1, 2))
    return {'estimate': float(values.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist()}


def interval(d, scale=1):
    return f"{scale*d['estimate']:.3f} [{scale*d['ci95'][0]:.3f}, {scale*d['ci95'][1]:.3f}]"


def summarize_variant(rows, reference, lr):
    taylor, _ = describe(rows, lr)
    cos = np.array([r['gradient_cosine'] for r in rows])
    result = {'count': len(rows), 'cosine_mean': float(cos.mean()), 'cosine_median': float(np.median(cos)),
        'positive_fraction': float((cos > 0).mean()),
        'benefit_fraction': float(np.mean([r['det_loss_delta_sem1'] < 0 for r in rows])),
        'own_semantic_decrease3_fraction': float(np.mean([r['semantic_loss_3'] < r['semantic_loss_before'] for r in rows])),
        'taylor': taylor, 'coordinate_decomposition': decomposition(rows), 'saturation_strata': saturation_groups(rows)}
    for field in ('semantic_loss_1', 'semantic_loss_3', 'generic_semantic_loss_1', 'generic_semantic_loss_3',
                  'det_loss_delta_sem1', 'det_loss_delta_sem3', 'saturation_before', 'saturation_1', 'saturation_3',
                  'adapt_seconds_3', 'peak_allocated_mb', 'g_sem_norm'):
        values = np.array([r[field] for r in rows])
        result[field] = {'mean': float(values.mean()), 'p05': float(np.quantile(values, .05)),
                         'median': float(np.median(values)), 'p95': float(np.quantile(values, .95))}
    for field in ('physical_change_1', 'physical_change_3', 'g_sem', 'g_sem_raw', 'coordinate_gate'):
        values = np.array([r[field] for r in rows])
        result[field] = {'mean': values.mean(0).tolist(), 'mean_abs': abs(values).mean(0).tolist()}
    result['raw_phi_trajectory_mean'] = np.array([[d['phi'] for d in r['diagnostics']] for r in rows]).mean(0).tolist()
    result['paired_vs_generic'] = {
        'cosine': paired_delta(rows, reference, lambda r: r['gradient_cosine']),
        'benefit_rate': paired_delta(rows, reference, lambda r: float(r['det_loss_delta_sem1'] < 0)),
        'loss_delta1': paired_delta(rows, reference, lambda r: r['det_loss_delta_sem1']),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    study = args.study
    rows = [json.loads(line) for line in (study / 'samples.jsonl').read_text().splitlines()]
    env = json.loads((study / 'environment.json').read_text())
    variants = env['config']['variants']
    cases = list(json.loads((study / 'summary.json').read_text())[variants[0]])
    metrics = json.loads((study / 'metrics.json').read_text())
    baseline = json.loads((study / 'baseline_metrics.json').read_text())
    case_of = lambda r: f"{r['family']}_s{r['severity']}"
    groups = {}
    for case in ['overall'] + cases:
        chosen = [r for r in rows if case == 'overall' or case_of(r) == case]
        reference = [r for r in chosen if r['variant'] == 'generic']
        groups[case] = {v: summarize_variant([r for r in chosen if r['variant'] == v], reference,
                                            env['config']['semantic_lr']) for v in variants}
        print(f'Analyzed {case}', flush=True)
    condition = {}
    for case in cases:
        chosen = [r for r in rows if r['variant'] == 'soft' and case_of(r) == case]
        weights = np.array([r['condition_weights'] for r in chosen])
        expected = {'gamma': 0, 'contrast': 1, 'color_cast': 2}[chosen[0]['family']]
        condition[case] = {'mean_weights': weights.mean(0).tolist(),
            'correct_top1_fraction': float((weights.argmax(1) == expected).mean()),
            'top1_fraction_per_concept': [(weights.argmax(1) == i).mean().item() for i in range(3)],
            'mean_max_weight': float(weights.max(1).mean()),
            'mean_normalized_entropy': float(-(weights*np.log(weights)).sum(1).mean()/np.log(3))}
    checks = json.loads((study / 'baseline_checks.json').read_text())
    audit = {'max_forward_loss_error_vs_T002': max(r['loss_abs_error'] for r in checks),
             'max_detector_gradient_error_vs_T002': max(r['gradient_max_abs_error'] for r in checks),
             'generic_AP_differences_vs_T002': {f'{c}_step{k}': 100*(metrics[f'{c}_generic{k}']['AP']-baseline[f'{c}_semantic{k}']['AP']) for c in cases for k in (1, 3)}}
    payload = {'groups': groups, 'condition_inference': condition, 'cross_run_audit': audit,
        'bootstrap': '2000 image-cluster percentile draws, seed20260912; paired; exploratory, no multiplicity adjustment',
        'effective_gradient': 'All cosines/Taylor use actual gated update vector, no norm compensation',
        'timing': env['timing']}
    (study / 'analysis.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    lines = ['# T003 condition-aware direction and coordinate gating', '',
        f"Source {env['source_revision']}; {len(env['evaluated_image_ids'])} images, {len(rows)} variant observations. "
        f"Smoke limit: {env['smoke_limit']}. Fixed temperature 0.05, lr 0.1, K=3.", '',
        'Generic is rerun in T003. All six variants share a fresh g_det per image/condition. '
        'Clean/corrupted AP uses the matching T002 subset. Oracle variants know the synthetic family only '
        'inside analysis; none uses annotations to choose a direction or a mask. No learned prompts or training.', '',
        '## Downstream AP (0–100 subset points)', '',
        '| Case | Corrupted | Generic 1 / 3 | Soft 1 / 3 | Oracle prompt 1 / 3 | Soft gate 1 / 3 | Soft + oracle gate 1 / 3 | Oracle both 1 / 3 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        cells = [f'{100*metrics[f"{case}_{v}1"]["AP"]:.3f} / {100*metrics[f"{case}_{v}3"]["AP"]:.3f}' for v in variants]
        lines.append(f'| {case} | {100*baseline[case+"_corrupted"]["AP"]:.3f} | '+ ' | '.join(cells)+' |')
    lines += ['', f'Clean subset AP: {100*baseline["clean"]["AP"]:.3f}. Full AP50/AP75/size metrics are in metrics.json.', '',
        '## Gradient and measured one-step behavior', '',
        'Cosines and Taylor predictions use the effective gated gradient. A smaller gate also shortens '
        'the step; no rescaling was used. Cosine improvement alone does not establish useful restoration.', '',
        '| Group | Variant | Mean / median cosine | Positive | Detector loss benefit | Taylor sign match | Taylor Spearman [95% CI] |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case, vg in groups.items():
        for v, g in vg.items():
            lines.append(f"| {case} | {v} | {g['cosine_mean']:.4f} / {g['cosine_median']:.4f} | "
                f"{100*g['positive_fraction']:.1f}% | {100*g['benefit_fraction']:.1f}% | "
                f"{100*g['taylor']['sign_agreement']['estimate']:.1f}% | {interval(g['taylor']['spearman_linear_observed'])} |")
    lines += ['', '## Paired differences against generic', '',
        'Intervals use 2,000 image-cluster bootstrap draws, keeping the six conditions of an image together '
        'overall. Exploratory 95% percentile intervals, no multiplicity adjustment. These are not AP intervals.', '',
        '| Group | Variant | Δ mean cosine [95% CI] | Δ benefit rate, percentage points [95% CI] | Δ observed detector-loss change [95% CI] |',
        '|---|---|---:|---:|---:|']
    for case, vg in groups.items():
        for v, g in vg.items():
            if v != 'generic':
                p = g['paired_vs_generic']
                lines.append(f"| {case} | {v} | {interval(p['cosine'])} | {interval(p['benefit_rate'], 100)} | {interval(p['loss_delta1'])} |")
    lines += ['', '## Original-image condition inference (analysis labels only for this table)', '',
        '| Case | Darkness / contrast / color weight | Correct top-1 | Max weight | Normalized entropy |',
        '|---|---:|---:|---:|---:|']
    for case, c in condition.items():
        weights = ' / '.join(f'{w:.3f}' for w in c['mean_weights'])
        lines.append(f"| {case} | {weights} | {100*c['correct_top1_fraction']:.1f}% | {c['mean_max_weight']:.3f} | {c['mean_normalized_entropy']:.3f} |")
    lines += ['', '## Loss, step size, saturation and runtime', '',
        'Own CLIP losses use each variant\'s own direction; the common/generic CLIP loss is also measured '
        'for all variants. Latency includes condition inference/setup, K=3 adaptation, diagnostics and GPU '
        'synchronization; excludes annotated loss/AP evaluation and common-loss diagnostics.', '',
        '| Group | Variant | Own CLIP Δ1 / Δ3 | Common CLIP Δ3 | Own loss decreases at 3 | Effective gradient norm | Saturation before / 1 / 3 | Sat3 p95 | Seconds3 | Peak MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case, vg in groups.items():
        for v, g in vg.items():
            sat = ' / '.join(f'{100*g[s]["mean"]:.2f}%' for s in ('saturation_before', 'saturation_1', 'saturation_3'))
            lines.append(f"| {case} | {v} | {g['semantic_loss_1']['mean']:.5f} / {g['semantic_loss_3']['mean']:.5f} | "
                f"{g['generic_semantic_loss_3']['mean']:.5f} | {100*g['own_semantic_decrease3_fraction']:.1f}% | "
                f"{g['g_sem_norm']['mean']:.5f} | {sat} | {100*g['saturation_3']['p95']:.2f}% | "
                f"{g['adapt_seconds_3']['mean']:.4f} | {g['peak_allocated_mb']['mean']:.1f} |")
    lines += ['', '## Saturation and coordinate details', '',
        'analysis.json retains per-case/variant and per-saturation-bucket signed coordinate contributions, '
        'sign agreement, gradient energy, harmful/beneficial/zero outcomes and outside-subspace norm; '
        'joint initial-to-one-step saturation strata are included. Small strata and post-treatment selection '
        'preclude causal claims. Every sample retains raw/physical parameter trajectories, raw and effective '
        'gradients, condition similarities/weights, direction and gate in samples.jsonl.', '',
        '## Cross-run numerical audit', '', '```json', json.dumps(audit, indent=2), '```', '',
        'Native detector backward and CUDA antialiased bicubic backward are not bitwise reproducible. '
        'The failed cache-equality smokes and repeated-gradient diagnostic are preserved. All primary '
        'T003 comparisons use contemporaneous generic/variant rows with a common fresh detector gradient.', '',
        '![Three-step AP change from corrupted baseline](ap_change.png)', '']
    (study / 'results.md').write_text('\n'.join(lines), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    changes = np.array([[100*(metrics[f'{c}_{v}3']['AP']-baseline[c+'_corrupted']['AP']) for c in cases] for v in variants])
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    limit = max(abs(changes).max(), .1)
    im = ax.imshow(changes, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(len(cases)), cases, rotation=20, ha='right')
    ax.set_yticks(range(len(variants)), variants)
    for (i, j), value in np.ndenumerate(changes):
        ax.text(j, i, f'{value:+.3f}', ha='center', va='center', color='white' if abs(value) > .65*limit else 'black')
    ax.set_title('T003 · three-step AP change from corrupted input')
    fig.colorbar(im, ax=ax, label='COCO subset AP points')
    fig.savefig(study / 'ap_change.png', dpi=180)
    fig.savefig(study / 'ap_change.pdf')
    plt.close(fig)
    print(study / 'results.md')


if __name__ == '__main__':
    main()
