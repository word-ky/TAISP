"""Render T013-H saved results with stdlib; no model or tensor execution."""
import argparse
import json
import statistics
from pathlib import Path


def render(a, collection):
    lines = ['# T013-H source replication tables', '',
             '64 primary records, no optimizer steps. Float64 saved-array algebra; first-order source predictions only.', '']

    def table(title, headers, rows):
        lines.extend(['## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(format(v,'.12g') if isinstance(v,float) else str(v) for v in row)+' |')
        lines.append('')

    scopes = [('overall',a['overall'])]+[(f'block{i}',b) for i,b in enumerate(a['blocks'])]
    table('Replication thresholds', ['Scope','Median rho','r_C','r_C_out','Full centered fraction','Clean/corrupt gradient cosine'],
          [[n,b['features']['median_rho'],b['r_C'],b['r_C_out'],b['full_output']['centered_fraction'],
            b['clean_corrupted_gradient_cosine']] for n,b in scopes])
    table('Feature and factorization scale',['Scope','mu norm','Raw H energy','Centered H energy','Centered H fraction','Effective rank','Norm A','Norm C','Cos A,C','2<A,C>'],
          [[n]+[b['features'][k] for k in ('mu_norm','total_energy','centered_energy','centered_fraction','participation_effective_rank')]
           +[b[k] for k in ('A_frobenius_norm','C_frobenius_norm','A_C_cosine','A_C_cross_energy')] for n,b in scopes])
    for n,b in scopes:
        table(n+' feature singular values',['Index','Value'],enumerate(b['features']['centered_singular_values']))
        table(n+' same-image feature displacement',['Clean local index','Corrupt local index','Distance','Median other-clean distance','rho'],
              [[p[k] for k in ('clean_index','corrupt_index','condition_distance','median_other_clean_distance','rho')] for p in b['features']['pairs']])
        table(n+' output components',['Component','Raw energy','Centered energy'],
              [[key,v['total_energy'],v['centered_energy']] for key,v in {**b['components'],'full':b['full_output']}.items()])
        table(n+' all cross terms',['First','Second','Raw cross energy','Centered cross energy'],
              [v['components']+[v['raw_cross_energy'],v['centered_cross_energy']] for v in b['cross_terms']])
    records = a['crossfit']['records']
    summary = []
    for fold in [None,0,1,2,3]:
        for direction in ('full','common','cov'):
            for group in ('all','clean','corrupted'):
                chosen = [r['directions'][direction] for r in records if (fold is None or r['fold']==fold)
                          and (group=='all' or r['clean']==(group=='clean'))]
                summary.append(['overall' if fold is None else fold,direction,group,len(chosen),
                                sum(r['g_dot_delta']<0 for r in chosen),statistics.mean(r['g_dot_delta'] for r in chosen),
                                statistics.median(r['cosine_to_negative_g'] for r in chosen),
                                statistics.mean(r['norm'] for r in chosen)])
    table('Cross-fitted first-order descriptive summary',['Fold','Direction','Group','N','g dot delta <0','Mean g dot delta','Median cosine','Mean norm'],summary)
    table('All per-episode cross-fitted predictions',['Episode','Image','Case','Fold','Direction','g dot delta','Cosine to -g','Norm','Same-image separation'],
          [[r['episode_index'],a['episodes'][r['episode_index']]['image_id'],a['episodes'][r['episode_index']]['case'],r['fold'],name]
           +[v[k] for k in ('g_dot_delta','cosine_to_negative_g','norm','same_image_separation')]
           for r in records for name,v in r['directions'].items()])
    table('Primary source records',['Episode','Image','Case','Block','Outer loss','phi0 norm','phi3 norm','Supports','Saturation','Isolation'],
          [[r[k] for k in ('episode_index','image_id','case','block','outer_loss','phi0_norm','phi3_norm','support_count','saturation_rate')]
           +[all(r['isolation'].values())] for r in collection['records']])
    table('Fixed-bound saved gradient checks',['Check','Passed','Max abs error','Frobenius error','Max error/bound'],
          [[n]+[v[k] for k in ('passed','max_absolute_error','frobenius_error','max_error_over_bound')] for n,v in a['reconstruction'].items()])
    table('Utility criteria',['Quantity','Value'],a['utility'].items())
    table('Fixed decision',['Flag','Value'],a['decision'].items())
    lines.extend(['Full H, gradients, A/C/mu/gbar, fold indices, output/delta vectors and distance matrices are retained in collection/records.json and algebra/audit.json.',''])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit','collection','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    args.output.write_text(render(json.loads(args.audit.read_text()),json.loads(args.collection.read_text())),encoding='utf-8')
