"""Render saved T013-I capacity results; standard library only."""
import argparse
import json
from pathlib import Path


def render(summary, observed):
    lines = ['# T013-I complete linear capacity tables', '',
             'Fixed source arrays, float64 algebra, no model execution or optimizer. These are gradient predictions, not AP or finite-step performance.', '']

    def table(title, headers, rows):
        lines.extend(['## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(format(v,'.12g') if isinstance(v,float) else str(v) for v in row)+' |')
        lines.append('')

    table('Fold fit numerical information',
          ['Fold','Rank','Tolerance','B Frobenius norm','All singular condition','Retained condition','Training SSE'],
          [[f[k] for k in ('fold','numerical_rank','rank_tolerance','coefficient_frobenius_norm',
                          'condition_all_singular_values','condition_retained_subspace','training_residual_sse')] for f in observed['folds']])
    for f in observed['folds']:
        table(f"Fold {f['fold']} singular values",['Index','Value'],enumerate(f['singular_values']))
    table('Residual statistics by scope and reference',
          ['Scope','Method','N','SSE','Residual energy','R2','Median residual cosine','Positive residual dot'],
          [[name,method]+[v[k] for k in ('n','residual_sse','residual_energy','R2_residual','median_residual_cosine',
                                       'positive_residual_dot_count')] for name,methods in summary['summaries'].items() for method,v in methods.items()])
    table('Full-gradient first-order statistics by scope and reference',
          ['Scope','Method','Median gradient cosine','Negative g dot delta','Mean g dot delta','Median g dot delta','Mean absolute g dot delta'],
          [[name,method]+[v[k] for k in ('median_full_gradient_cosine','negative_first_order_count','first_order_mean',
                                       'first_order_median','first_order_mean_absolute')]
           for name,methods in summary['summaries'].items() for method,v in methods.items()])
    table('Norms by scope and reference',['Scope','Method','Vector','Mean','Median','Max'],
          [[name,method,vector]+[v[k] for k in ('mean','median','max')]
           for name,methods in summary['summaries'].items() for method,m in methods.items() for vector,v in m['norms'].items()])
    table('Matched permutation comparisons',
          ['Metric','Observed','Null95 linear','Empirical percentile (<)','Ties','#null>=observed','Corrected upper tail'],
          [[name]+[v[k] for k in ('observed','null_95th_percentile_linear','empirical_percentile_strict','ties',
                                 'null_at_least_observed','corrected_one_sided_tail')] for name,v in summary['comparisons'].items()])
    names = list(summary['comparisons'])
    table('All 128 null outcomes',['Permutation']+names,
          [[i]+[summary['comparisons'][n]['values'][i] for n in names] for i in range(128)])
    table('All held-out episode outcomes',
          ['Episode','Image','Case','Fold','Method','SSE','Residual dot','Residual cosine','Gradient cosine','g dot delta','norm r','norm rhat','norm ghat'],
          [[r['episode_index'],r['image_id'],r['case'],r['fold'],name]
           +[v[k] for k in ('residual_sse','residual_dot','residual_cosine','full_gradient_cosine','g_dot_delta',
                            'norm_r','norm_r_hat','norm_g_hat')] for r in observed['episodes'] for name,v in r['methods'].items()])
    table('Predeclared gate',['Flag','Passed'],summary['gate']['flags'].items())
    lines.extend([f"Decision: {summary['gate']['decision']}",'',
                  'Full input/residual/prediction/delta vectors, SVD coefficients and fold indices are in observed.json. Each permutation_NNN.json retains all four fitted coefficients, pair/row mappings, held-out residual/full-gradient predictions and pooled metrics. No null outcome is discarded.',''])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('summary','observed','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    args.output.write_text(render(json.loads(args.summary.read_text()),json.loads(args.observed.read_text())),encoding='utf-8')
