"""T003 Stage A: coordinate cross-talk and saturation from saved T002 receipts."""

import argparse
import json
from pathlib import Path

import numpy as np

COORDINATES = ('gamma', 'red_gain', 'green_gain', 'blue_gain', 'contrast',
               'brightness', 'tone', 'sharpening')
SUBSPACES = {'gamma': [0, 5, 6], 'contrast': [4, 6], 'color_cast': [1, 2, 3]}
BUCKETS = ('[0,1%]', '(1%,5%]', '(5%,10%]', '(10%,100%]')


def bucket(value):
    return int(np.searchsorted([.01, .05, .10], value, side='left'))


def decomposition(rows):
    sem = np.asarray([r['g_sem'] for r in rows])
    det = np.asarray([r['g_det'] for r in rows])
    energy = sem**2 / (sem**2).sum(1, keepdims=True)
    outside = []
    for r, e in zip(rows, energy):
        outside.append(float(np.sqrt(max(0., 1 - e[SUBSPACES[r['family']]].sum()))))
    delta = np.array([r['det_loss_delta_sem1'] for r in rows])
    result = {'count': len(rows), 'harm_fraction': float((delta > 0).mean()),
              'benefit_fraction': float((delta < 0).mean()), 'zero_count': int((delta == 0).sum()),
              'outside_norm_fraction_mean': float(np.mean(outside)),
              'outside_energy_fraction_mean': float(np.mean(np.square(outside))),
              'coordinates': {}}
    for j, name in enumerate(COORDINATES):
        active = (sem[:, j] != 0) & (det[:, j] != 0)
        contribution = sem[:, j] * det[:, j]
        result['coordinates'][name] = {
            'signed_contribution_mean': float(contribution.mean()),
            'signed_contribution_median': float(np.median(contribution)),
            'absolute_contribution_mean': float(abs(contribution).mean()),
            'sign_agreement': float((np.sign(sem[:, j]) == np.sign(det[:, j])).mean()),
            'nonzero_pairs': int(active.sum()),
            'nonzero_sign_agreement': float((contribution[active] > 0).mean()) if active.any() else None,
            'relative_semantic_energy_mean': float(energy[:, j].mean()),
        }
    return result


def saturation_groups(rows):
    result = {}
    for stage, key in [('initial', 'saturation_before'), ('post1', 'saturation_1')]:
        for b, label in enumerate(BUCKETS):
            group = [r for r in rows if bucket(r[key]) == b]
            result[f'{stage}:{label}'] = decomposition(group) if group else {'count': 0}
    for a, alabel in enumerate(BUCKETS):
        for b, blabel in enumerate(BUCKETS):
            group = [r for r in rows if bucket(r['saturation_before']) == a and bucket(r['saturation_1']) == b]
            result[f'{alabel}->{blabel}'] = decomposition(group) if group else {'count': 0}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.study / 'samples.jsonl').read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = sorted({f"{r['family']}_s{r['severity']}" for r in rows})
    groups = {'overall': rows, **{case: [r for r in rows if f"{r['family']}_s{r['severity']}" == case] for case in cases}}
    payload = {'source': str(args.study), 'definitions': {'subspaces': SUBSPACES, 'buckets': BUCKETS,
        'energy': 'mean of per-observation g_sem_j^2 / sum_j g_sem_j^2',
        'outside_norm': 'mean of norm(g_out) / norm(g_sem)', 'sign': 'exact signs, zeros included'},
        'groups': {name: {'decomposition': decomposition(group), 'saturation': saturation_groups(group)}
                   for name, group in groups.items()}}
    (args.output / 'coordinates.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# T003 Stage A: saved-gradient mechanism decomposition', '',
             'Source: final parity-corrected T002 receipts, 200 images / 1,200 observations. No new GPU run.', '',
             'Signed contribution is g_det[j]*g_sem[j]; positive favors a local descent step. '
             'Energy is averaged after normalizing each observation. Saturation is fraction of RGB values '
             'within 1e-4 of either output boundary. Post-step strata are descriptive and cannot establish causality.', '']
    for name, group in payload['groups'].items():
        d = group['decomposition']
        lines += [f'## {name}', '', f"N={d['count']}; outside-subspace norm fraction={d['outside_norm_fraction_mean']:.4f}; "
                  f"outside energy={d['outside_energy_fraction_mean']:.4f}; harm={100*d['harm_fraction']:.1f}%, "
                  f"benefit={100*d['benefit_fraction']:.1f}%.", '',
                  '| Coordinate | Mean signed contribution | Median contribution | Sign agreement | Semantic energy |',
                  '|---|---:|---:|---:|---:|']
        for coord, c in d['coordinates'].items():
            lines.append(f"| {coord} | {c['signed_contribution_mean']:.6g} | {c['signed_contribution_median']:.6g} | "
                         f"{100*c['sign_agreement']:.1f}% | {100*c['relative_semantic_energy_mean']:.2f}% |")
        lines += ['', '| Saturation stratum | N | Harm | Benefit | Outside norm |', '|---|---:|---:|---:|---:|']
        for stratum, s in group['saturation'].items():
            if s['count']:
                lines.append(f"| {stratum} | {s['count']} | {100*s['harm_fraction']:.1f}% | "
                             f"{100*s['benefit_fraction']:.1f}% | {s['outside_norm_fraction_mean']:.4f} |")
        lines.append('')
    lines += ['All joint strata, per-stratum coordinates, zero counts and nonzero-pair sign statistics are in coordinates.json.', '']
    (args.output / 'coordinates.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.output / 'coordinates.md')


if __name__ == '__main__':
    main()
