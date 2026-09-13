"""Render T016-A heldout calibration, nulls and fixed gates from saved receipts."""
import argparse
import json
from pathlib import Path


def render(observed, summary, nulls):
    lines = ['# T016-A cross-fitted calibration tables', '',
             'All primary metrics use heldout predictions. Null cosines explicitly denote zero norms.', '']

    def table(title, headers, values):
        lines.extend(['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        for row in values:
            lines.append('| ' + ' | '.join(format(v, '.12g') if isinstance(v, float) else str(v) for v in row) + ' |')
        lines.append('')

    table('Primary quantities by scope', ['Scope', 'N', 'Median raw cosine', 'Median cal cosine',
          'Raw positive C_diff', 'Cal positive C_diff', 'Median C_diff_cal', 'Positive Delta_D', 'Median Delta_D'],
          [[name, s['n'], s['metrics']['cos_diff_raw']['median'], s['metrics']['cos_diff_cal']['median'],
            s['metrics']['C_diff_raw']['positive_count'], s['metrics']['C_diff_cal']['positive_count'],
            s['metrics']['C_diff_cal']['median'], s['metrics']['Delta_D_cal']['positive_count'],
            s['metrics']['Delta_D_cal']['median']] for name, s in summary['scopes'].items()])
    table('All metrics by scope', ['Scope', 'Metric', 'Valid N', 'Mean', 'Median', 'Min', 'Max', 'Positive count'],
          [[name, metric] + [v[k] for k in ('valid_count', 'mean', 'median', 'min', 'max', 'positive_count')]
           for name, s in summary['scopes'].items() for metric, v in s['metrics'].items()])
    table('Null comparisons', ['Metric', 'Observed', 'Null95', 'Strict percentile', 'Null >= observed', 'Ties', 'Corrected tail'],
          [[name] + [c[k] for k in ('observed', 'null_95th_percentile_linear', 'empirical_percentile_strict',
                                     'null_at_least_observed', 'ties', 'corrected_one_sided_tail')]
           for name, c in summary['comparisons'].items()])
    table('All 256 null outcomes', ['Permutation', 'Median Delta_D', 'Positive Delta_D', 'Positive C_diff_cal', 'Median cos_diff_cal', 'Norm checks'],
          [[r['index']] + [r['pooled'][k] for k in ('median_Delta_D_cal', 'positive_Delta_D_cal',
                                                   'positive_C_diff_cal', 'median_cos_diff_cal')] + [r['all_norm_checks_passed']]
           for r in nulls])
    table('Fold coefficients and isolation', ['Fold', 'Train rows', 'Target rows', 'Heldout rows'] + [f'a{i}' for i in range(8)],
          [[r[k] for k in ('fold', 'train_indices', 'target_indices', 'test_indices')] + r['a'] for r in observed['folds']])
    table('Coefficient stability', ['Coordinate', 'Values', 'Signs', 'Mean', 'Population std', 'Min', 'Max', 'Positive folds', 'Negative folds', 'Zero folds'],
          [[r[k] for k in ('coordinate', 'values', 'signs', 'mean', 'std_population', 'min', 'max',
                           'positive_folds', 'negative_folds', 'zero_folds')] for r in observed['coefficient_stability']])
    rows = observed['records']
    names = list(rows[0]['analysis']['metrics'])
    table('All heldout metrics', ['Episode', 'Image', 'Case', 'Block'] + names,
          [[r[k] for k in ('episode_index', 'image_id', 'case', 'block')] +
           [r['analysis']['metrics'][k] for k in names] for r in rows])
    table('Norm and zero diagnostics', ['Episode', 'Norm checks', 'Zero norms', 'Raw dot sign', 'Cal dot sign'],
          [[r['episode_index']] + [r['analysis'][k] for k in ('norm_check', 'zero_norm', 'raw_diff_dot_sign', 'diff_dot_sign')] for r in rows])
    table('Calibrated heldout vectors', ['Episode', 'Vector', 'Coordinates'],
          [[r['episode_index'], k, r['analysis'][k]] for r in rows for k in ('predicted_d', 'q_diff_nm', 'q')])
    table('Frozen advancement flags', ['Flag', 'Passed'], summary['gate']['flags'].items())
    lines += [json.dumps(summary['gate'], indent=2), '',
              'All null training mappings, coefficients, heldout vectors and metrics are retained in the raw permutation files.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    observed = json.loads((args.audit_root / 'observed.json').read_text())
    summary = json.loads((args.audit_root / 'summary.json').read_text())
    nulls = [json.loads(p.read_text()) for p in sorted(args.audit_root.glob('permutation_[0-9][0-9][0-9].json'))]
    args.output.write_text(render(observed, summary, nulls), encoding='utf-8')
