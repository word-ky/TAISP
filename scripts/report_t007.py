"""Disjoint-set finite-step confirmation; collinearity is not alignment gain."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_t002_linear import describe
from scripts.report_t003 import paired_delta, interval
from scripts.report_t005 import basics, paired, distribution, finite_json
from scripts.report_t006 import detector_rows, joint_outcome

HYBRID = 'det_pseudo_clip_radius'


def ratio_stratum(row):
    d = row['diagnostics'][0]
    r = d['detector_clip_ratio']
    if r is None:
        return 'clip_zero'
    return '[0,1)' if r < 1 else '[1,2)' if r < 2 else '[2,4)' if r < 4 else '[4,inf)'


def flip_values(hybrid, raw):
    a, b = raw['target']['det_loss_delta_sem1'], hybrid['target']['det_loss_delta_sem1']
    return {'harmful_to_beneficial': float(a > 0 and b < 0),
            'beneficial_to_harmful': float(a < 0 and b > 0),
            'either_zero': float(a == 0 or b == 0)}


def cluster_fraction(rows, values):
    # Ratio strata may contain different numbers of conditions for each image.
    ids = np.array([r['image_id'] for r in rows])
    unique = np.unique(ids)
    sums = np.array([np.sum(np.asarray(values)[ids == i]) for i in unique])
    counts = np.array([(ids == i).sum() for i in unique])
    selected = np.random.default_rng(20260912).integers(0, len(unique), (2000, len(unique)))
    boot = sums[selected].sum(1)/counts[selected].sum(1)
    return {'estimate': float(np.mean(values)), 'ci95': np.quantile(boot, [.025, .975]).tolist()}


def summarize(chosen, variants, lr):
    rows = {v: [r for r in chosen if r['variant'] == v] for v in variants}
    result = {'variants': {}, 'contrasts': {}}
    for v, rr in rows.items():
        g = {'count': len(rr), 'detectors': {}}
        for d in ('source', 'target'):
            dr = detector_rows(rr, d)
            taylor, _ = describe([{**r, 'gradient_cosine': r['gradient_cosine'] or 0.} for r in dr], lr)
            g['detectors'][d] = {**basics(dr), 'loss_delta3': distribution([r['det_loss_delta_sem3'] for r in dr]),
                'benefit_fraction3': float(np.mean([r['det_loss_delta_sem3'] < 0 for r in dr])), 'taylor': taylor}
        g['joint'] = {o: float(np.mean([joint_outcome(r) == o for r in rr])) for o in ('both', 'source_only', 'target_only', 'neither')}
        for f in ('phi_norm_1', 'phi_norm_3', 'saturation_before', 'saturation_1', 'saturation_3', 'support_count',
                  'g_sem_norm', 'adapt_seconds_3', 'deploy_seconds_3', 'peak_allocated_mb'):
            g[f] = distribution([r[f] for r in rr])
        g['fallback_fraction'] = float(np.mean([r['no_update_fallback'] for r in rr]))
        g['phi_trajectory_mean'] = np.array([[d['phi'] for d in r['diagnostics']] for r in rr]).mean(0).tolist()
        g['support_confidence'] = distribution([s for r in rr for s in r['support']['scores']])
        result['variants'][v] = g
    for a, b in ((HYBRID, 'det_pseudo'), (HYBRID, 'global_generic'), ('det_pseudo', 'global_generic')):
        ra, rb = rows[a], rows[b]
        q = {'detectors': {d: paired(detector_rows(ra, d), detector_rows(rb, d)) for d in ('source', 'target')},
             'joint_benefit': paired_delta(ra, rb, lambda r: float(joint_outcome(r) == 'both'))}
        for f in ('phi_norm_1', 'phi_norm_3', 'saturation_1', 'saturation_3'):
            q[f] = paired_delta(ra, rb, lambda r: r[f])
        result['contrasts'][a+'-minus-'+b] = q
    hybrid, raw = rows[HYBRID], rows['det_pseudo']
    keyed = {(r['image_id'], r['family'], r['severity']): r for r in raw}
    result['scale_steps'] = {str(k): {f: distribution([r['diagnostics'][k][f] for r in hybrid if r['diagnostics'][k][f] is not None])
        for f in ('detector_gradient_norm', 'clip_gradient_norm', 'detector_clip_ratio', 'scale_factor', 'gradient_norm', 'step_seconds')}
        for k in range(4)}
    result['scale_steps_null_ratios'] = [sum(r['diagnostics'][k]['detector_clip_ratio'] is None for r in hybrid) for k in range(4)]
    gd = np.array([r['diagnostics'][0]['detector_gradient'] for r in hybrid])
    gh = np.array([r['g_sem'] for r in hybrid])
    denom = np.linalg.norm(gd, axis=1)*np.linalg.norm(gh, axis=1)
    valid = denom > 0
    collinear = np.sum(gd[valid]*gh[valid], axis=1)/denom[valid]
    result['hybrid_same_forward_direction_cosine'] = distribution(collinear)
    result['max_one_minus_collinear_cosine'] = float(np.max(abs(1-collinear))) if len(collinear) else None
    result['flips_by_ratio'] = {}
    for name in ('all', '[0,1)', '[1,2)', '[2,4)', '[4,inf)', 'clip_zero'):
        hh = [r for r in hybrid if name == 'all' or ratio_stratum(r) == name]
        if not hh:
            result['flips_by_ratio'][name] = {'count': 0}
            continue
        flips = [flip_values(r, keyed[(r['image_id'], r['family'], r['severity'])]) for r in hh]
        result['flips_by_ratio'][name] = {'count': len(hh), 'image_clusters': len(set(r['image_id'] for r in hh)),
            **{k: cluster_fraction(hh, [f[k] for f in flips]) for k in flips[0]}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    study = parser.parse_args().study
    read = lambda name: json.loads((study/name).read_text(encoding='utf-8'))
    env, metrics = read('environment.json'), read('metrics.json')
    rows = [json.loads(s) for s in (study/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    variants = env['config']['variants']
    cases = list(read('summary.json')[variants[0]])
    groups = {}
    for c in ['corrupted_overall', 'severity_s1', 'severity_s2']+cases:
        if c == 'corrupted_overall':
            rr = [r for r in rows if r['family'] != 'clean']
        elif c.startswith('severity_'):
            rr = [r for r in rows if r['severity'] == int(c[-1])]
        else:
            rr = [r for r in rows if f"{r['family']}_s{r['severity']}" == c]
        groups[c] = summarize(rr, variants, env['config']['semantic_lr'])
        print('Analyzed '+c, flush=True)
    ap = {c: {d: {v: {str(k): {m: {'value': 100*metrics[f'{d}_{c}_{v}{k}'][m],
            **{f'delta_{ref}': 100*(metrics[f'{d}_{c}_{v}{k}'][m]-metrics[f'{d}_{c}_'+('before' if ref == 'before' else ref+str(k))][m])
               for ref in ('before', 'global_generic', 'det_pseudo')}} for m in ('AP', 'AP50', 'AP75')}
            for k in (1, 3)} for v in variants} for d in ('source', 'target')} for c in cases}
    payload = {'source_revision': env['source_revision'], 'groups': groups, 'AP': ap,
        'bootstrap': '2000 paired image-cluster percentile draws seed20260912; all selected cases travel together; ratio strata use cluster-weighted observation fraction; exploratory; no AP intervals',
        'direction': 'Hybrid is collinear with its own current-phi source gradient; independent CUDA forwards may differ numerically. No alignment gain is claimed.',
        'strata': '[0,1),[1,2),[2,4),[4,inf),clip_zero; fixed before results; ratio from initial hybrid gradient evaluation'}
    (study/'analysis.json').write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# T007 disjoint-set trust-radius results', '',
        f"Source {env['source_revision']}; images={len(env['evaluated_image_ids'])}, rows={len(rows)}, smoke={env['smoke_limit']}.", '',
        'Frozen source pseudo direction, current-phi CLIP norm, eps1e-12, lr0.1,K1/3. Target FCOS/annotations analysis-only. Hybrid cosine is not a new direction. Both detectors evaluate the same enhanced images.', '',
        '## Official subset AP / AP50 / AP75', '',
        '| Detector | Case | Variant | Before AP /50 /75 | After1 AP /50 /75 | After3 AP /50 /75 | DeltaAP3 vs before / CLIP / raw |',
        '|---|---|---|---:|---:|---:|---:|']
    for c in cases:
        for d in ('source', 'target'):
            for v in variants:
                g = ap[c][d][v]
                before = ' / '.join(f"{100*metrics[f'{d}_{c}_before'][m]:.3f}" for m in ('AP', 'AP50', 'AP75'))
                steps = [' / '.join(f"{g[str(k)][m]['value']:.3f}" for m in ('AP', 'AP50', 'AP75')) for k in (1, 3)]
                delta = ' / '.join(f"{g['3']['AP'][f'delta_{r}']:+.3f}" for r in ('before', 'global_generic', 'det_pseudo'))
                lines.append(f'| {d} | {c} | {v} | {before} | '+' | '.join(steps)+f' | {delta} |')
    lines += ['', 'All K1/K3 AP/AP50/AP75 deltas retained in analysis.json; no invented AP CIs.', '',
        '## Paired one-step finite behavior', '',
        '| Group | Detector | Contrast | Delta cosine [CI] (not a direction gain) | Delta benefit pp [CI] | Delta mean loss [CI] |', '|---|---|---|---:|---:|---:|']
    for c, g in groups.items():
        for contrast, q in g['contrasts'].items():
            for d, s in q['detectors'].items():
                z = s['loss_delta1']
                lines.append(f"| {c} | {d} | {contrast} | {interval(s['cosine_zero_coded'])} | {interval(s['benefit_rate'],100)} | {z['estimate']:+.6f} [{z['ci95'][0]:+.6f}, {z['ci95'][1]:+.6f}] |")
    lines += ['', '## Clean and corrupted update magnitudes', '',
        '| Group | Variant | Phi1 /3 | Saturation before /1 /3 % | Deploy sec3 | Peak MiB | Fallback% |', '|---|---|---:|---:|---:|---:|---:|']
    for c in ('corrupted_overall', 'clean_s0'):
        for v, g in groups[c]['variants'].items():
            sat = ' / '.join(f"{100*g[f]['mean']:.3f}" for f in ('saturation_before', 'saturation_1', 'saturation_3'))
            lines.append(f"| {c} | {v} | {g['phi_norm_1']['mean']:.5f} / {g['phi_norm_3']['mean']:.5f} | {sat} | {g['deploy_seconds_3']['mean']:.4f} | {g['peak_allocated_mb']['mean']:.1f} | {100*g['fallback_fraction']:.2f} |")
    lines += ['', '## Target loss sign flips by initial detector / CLIP norm ratio', '',
        'Strict harmful>0 and beneficial<0; zeros separate. Strata only analyze outcomes, never control deployment.', '',
        '| Group | Ratio | N | Raw harmful to hybrid beneficial% [CI] | Reverse% [CI] | Either zero% [CI] |', '|---|---|---:|---:|---:|---:|']
    for c in ('corrupted_overall', 'clean_s0'):
        for s, q in groups[c]['flips_by_ratio'].items():
            if q['count']:
                lines.append(f"| {c} | {s} | {q['count']} | "+' | '.join(interval(q[k],100) for k in ('harmful_to_beneficial', 'beneficial_to_harmful', 'either_zero'))+' |')
    lines += ['', 'Step0/1/2/3 detector/CLIP/update norms, ratios/scales, collinearity, support, source/target loss1/3 and joint fractions are saved in analysis.json. Step3 gradients are terminal diagnostics, not an additional update. Raw observations retain every trajectory and failure/fallback.', '',
              '![Hybrid target and source AP3 deltas](transfer_ap.png)', '']
    (study/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    refs = ('before', 'det_pseudo', 'global_generic')
    values = np.array([[ap[c][d][HYBRID]['3']['AP']['delta_'+r] for d in ('source', 'target') for r in refs] for c in cases])
    bound = max(abs(values).max(), .01)
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    im = ax.imshow(values, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(6), ['Source vs before', 'Source vs raw', 'Source vs CLIP', 'FCOS vs before', 'FCOS vs raw', 'FCOS vs CLIP'])
    ax.set_yticks(range(len(cases)), cases)
    ax.set_title('T007 disjoint subset: hybrid AP change after 3 steps')
    for i in range(len(cases)):
        for j in range(6):
            ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center', color='white' if abs(values[i,j]) > .6*bound else 'black')
    fig.colorbar(im, ax=ax, label='AP points')
    fig.savefig(study/'transfer_ap.png', dpi=180)
    fig.savefig(study/'transfer_ap.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
