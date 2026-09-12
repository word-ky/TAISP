"""Render saved T013-G matrices and checks; standard library only."""
import argparse
import json
from pathlib import Path


def render(a):
    lines = ['# T013-G complete feature and factorization tables', '',
             'Float64 offline algebra from pinned T013-C arrays. No performance measurements.', '']
    labels = [f"{e['image_id']}/{e['case']}" for e in a['episodes']]

    def table(title,headers,rows):
        lines.extend([f'## {title}', '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(format(v,'.12g') if isinstance(v,float) else str(v) for v in row)+' |')
        lines.append('')

    def matrix(title,values,rowlabels,collabels):
        table(title,['Row']+collabels,[[n]+r for n,r in zip(rowlabels,values)])

    f = a['features']
    table('Feature summaries',['Quantity','Value'],[[k,f[k]] for k in
          ('mu_norm','total_energy','centered_energy','centered_fraction','participation_effective_rank','median_rho')])
    table('Four condition displacements and nearest clean features',
          ['Clean episode','Corrupt episode','Distance','Median other-clean distance','rho','Nearest clean','Nearest distance','Own image'],
          [[labels[p['clean_index']],labels[p['corrupt_index']],p['condition_distance'],p['median_other_clean_distance'],
            p['rho'],labels[p['nearest_clean_index']],p['nearest_clean_distance'],p['nearest_clean_is_own_image']] for p in f['pairs']])
    table('Centered feature singular values',['Index','Value'],enumerate(f['centered_singular_values']))
    matrix('Full feature distance matrix',f['euclidean_distances'],labels,labels)
    matrix('Full feature cosine matrix',f['cosine_matrix'],labels,labels)
    for name in ('H','Z'):
        matrix(name,f[name],labels,[f'h{i}' for i in range(16)])
    table('Feature mean',['Coordinate','Value'],enumerate(f['mu']))
    matrix('Explicit phi0 gradients G',a['G'],labels,[f'phi{i}' for i in range(8)])
    table('Mean explicit gradient',['Coordinate','Value'],enumerate(a['g_bar']))
    for name in ('A','C','G_W'):
        matrix(name,a[name],[f'phi{i}' for i in range(8)],[f'h{i}' for i in range(16)])
    table('Gradient factorization summaries',['Quantity','Value'],[[k,a[k]] for k in
          ('A_frobenius_norm','C_frobenius_norm','A_C_cosine','A_C_cross_energy','r_C','r_C_out','float64_factorization_max_error')])
    table('All fixed-bound reconstructions',['Check','Passed','Max abs error','Frobenius error','Reference scale','Largest element bound','Max error/bound'],
          [[n]+[v[k] for k in ('passed','max_absolute_error','frobenius_error','reference_scale','max_elementwise_bound','max_error_over_bound')]
           for n,v in a['checks'].items()])
    outputs = {**a['components'],'full_output':a['full_output'],**a.get('counterfactuals',{})}
    table('Output energies',['Component / counterfactual','Raw energy','Centered energy'],
          [[n,v['total_energy'],v['centered_energy']] for n,v in outputs.items()])
    table('Nonorthogonal cross terms',['Component A','Component B','Raw cross energy','Centered cross energy'],
          [v['components']+[v['raw_cross_energy'],v['centered_cross_energy']] for v in a['cross_terms']])
    table('All same-image output distances',['Component / counterfactual']+[labels[i] for i in range(0,8,2)],
          [[n]+v['same_image_distances'] for n,v in outputs.items()])
    for name,value in outputs.items():
        matrix(name+' vectors',value['outputs'],labels,[f'phi{i}' for i in range(8)])
        matrix(name+' pairwise distances',value['pairwise_distances'],labels,labels)
    lines.extend(['## Fixed triage', '', a['triage'], '',
                  'No architecture, optimization or performance recommendation follows automatically.', ''])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.audit.read_text())),encoding='utf-8')
