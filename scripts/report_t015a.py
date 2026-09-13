"""Render T015-A saved arrays and triage; no model or scientific recomputation."""
import argparse
import json
from pathlib import Path


def render(rows, summary):
    lines = ['# T015-A frozen differential-subspace tables', '',
             'Only authoritative fresh T014-A1 reference vectors are used. Null cosines denote zero-norm cases.', '']

    def table(title, headers, values):
        lines.extend(['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        for row in values:
            lines.append('| ' + ' | '.join(format(v, '.12g') if isinstance(v, float) else str(v) for v in row) + ' |')
        lines.append('')

    scopes = summary['scopes']
    table('Primary quantities', ['Scope', 'N', 'Median R_extra', 'Median task diff energy',
                                'Median pseudo diff energy', 'Median C_diff', 'Positive C_diff', 'Median cos_diff'],
          [[name, s['n'], s['metrics']['R_extra']['median'], s['metrics']['f_diff_task']['median'],
            s['metrics']['f_diff_pseudo']['median'], s['metrics']['C_diff']['median'],
            s['metrics']['C_diff']['positive_count'], s['metrics']['cos_diff']['median']] for name, s in scopes.items()])
    table('All metrics by scope', ['Scope', 'Metric', 'Valid N', 'Mean', 'Median', 'Min', 'Max', 'Positive count'],
          [[name, metric] + [v[k] for k in ('valid_count', 'mean', 'median', 'min', 'max', 'positive_count')]
           for name, s in scopes.items() for metric, v in s['metrics'].items()])
    table('Zero norms and signs', ['Scope', 'Zero norm counts', 'Differential dot sign counts'],
          [[name, s['zero_norm_counts'], s['diff_dot_sign_counts']] for name, s in scopes.items()])
    names = list(rows[0]['analysis']['metrics'])
    table('Per-episode metrics', ['Episode', 'Image', 'Case', 'Block', 'Support', 'Mask area'] + names,
          [[r[k] for k in ('episode_index', 'image_id', 'case', 'block', 'support_count', 'mask_area_fraction')] +
           [r['analysis']['metrics'][k] for k in names] for r in rows])
    quantities = ['norm', 'shared_norm', 'diff_norm', 'f_diff', 'vector_reconstruction_error',
                  'energy_reconstruction_error', 'orthogonality_residual']
    table('Norms and numerical reconstruction', ['Episode', 'Objective'] + quantities,
          [[r['episode_index'], obj] + [r['analysis'][obj][k] for k in quantities]
           for r in rows for obj in ('task', 'pseudo')])
    table('All shared and differential 8-D vectors', ['Episode', 'Objective', 'Component'] + [f'phi{i}' for i in range(8)],
          [[r['episode_index'], obj, component] + r['analysis'][obj][component]
           for r in rows for obj in ('task', 'pseudo') for component in ('s', 'd')])
    table('Frozen triage flags', ['Branch', 'Flag', 'Passed'],
          [[branch, k, v] for branch in ('task_flags', 'pseudo_flags') for k, v in summary['triage'][branch].items()])
    lines += [json.dumps(summary['triage'], indent=2), '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = json.loads((args.audit_root / 'records.json').read_text())['records']
    summary = json.loads((args.audit_root / 'summary.json').read_text())
    args.output.write_text(render(rows, summary), encoding='utf-8')
