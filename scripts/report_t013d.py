"""Render the retained T013-D audit; no models, training or new measurements."""
import argparse
import json
from pathlib import Path


def render(root):
    summary = json.loads((root/'repeatability_summary.json').read_text())
    algebra = json.loads((root/'common_mode.json').read_text())
    repeats = [json.loads((root/f'repeat_{i:02d}.json').read_text()) for i in range(12)]
    lines = ['# T013-D full diagnostic tables', '',
             'All 12 repeats retained. Standard deviations use denominator 12. Counts include repeat 0.',
             'Effect comparisons are descriptive, not p-values or confidence intervals.', '']

    def table(title, headers, rows):
        lines.extend([f'## {title}', '', '| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(format(x, '.10g') if isinstance(x, float) else str(x) for x in row)+' |')
        lines.append('')

    def label(row):
        return f"{row['image_id']}/{row['case']}"

    table('Aggregate geometry, every repeat', ['Repeat', 'phi cosine', 'head cosine',
          'phi clean norm', 'phi corrupt norm', 'head clean norm', 'head corrupt norm'],
          [[r['repeat'], r['geometry']['phi']['cosine'], r['geometry']['head']['cosine']]+
           [r['geometry'][space][group+'_norm'] for space in ('phi','head') for group in ('clean','corrupted')]
           for r in repeats])
    table('Aggregate cosine summary', ['Space','Median','Min','Max','Std','Negative / 12','Same sign'],
          [[k]+[v['cosine_stats'][f] for f in ('median','min','max','std_population')]+[v['negative_count'],v['all_same_sign']]
           for k,v in summary['aggregate_geometry'].items()])
    table('Aggregate norm variability', ['Space','Group','Mean','Std','Min','Max','Range'],
          [[k,g]+[v[g+'_norm_stats'][f] for f in ('mean','std_population','min','max','range')]
           for k,v in summary['aggregate_geometry'].items() for g in ('clean','corrupted')])
    table('All raw per-episode loss, radius and saturation',
          ['Repeat','Episode','Loss','phi3 norm','Saturation','Support count'],
          [[r['repeat'],label(e),e['outer_loss'],e['phi3_norm'],e['saturation_rate'],e['support_count']]
           for r in repeats for e in r['episodes']])
    for field in ('loss','phi3_radius'):
        table(f'Per-episode {field} variability', ['Episode','Mean','Std','Min','Max','Range','Max abs deviation r0'],
              [[label(e)]+[e[field][f] for f in ('mean','std_population','min','max','range','max_abs_deviation_from_r0')]
               for e in summary['episode_variability']])
    table('Per-coordinate phi3 variability', ['Episode','Coordinate','Mean','Std','Min','Max','Range','Max abs deviation r0'],
          [[label(e),j]+[c[f] for f in ('mean','std_population','min','max','range','max_abs_deviation_from_r0')]
           for e in summary['episode_variability'] for j,c in enumerate(e['phi3_coordinates'])])
    for space in ('phi','head'):
        table(f'{space} gradient cosine against repeat 0', ['Episode']+[str(i) for i in range(12)],
              [[label(e)]+e[space+'_gradient_cosines_to_r0'] for e in summary['episode_variability']])
    table('Group loss distributions', ['Group','Mean','Std','Min','Max','Range','Max abs deviation r0'],
          [[g]+[v[f] for f in ('mean','std_population','min','max','range','max_abs_deviation_from_r0')]
           for g,v in summary['group_loss_distributions'].items()])
    effect_headers = ['Probe','Group/episode','Signed effect','Abs / std','Abs / range','Count / 12']

    def effect_row(name, name2, value):
        return [name,name2,value['signed_effect'],value['abs_effect_over_std'],
                value['abs_effect_over_range'],value['count_equal_or_larger_abs_change']]

    comparisons = summary['t013c_fixed_effect_comparison']
    for field in ('group_loss_effects','group_radius_effects'):
        table(field, effect_headers,
              [effect_row(n,g,v) for n,p in comparisons.items() for g,v in p[field].items()])
    for field in ('loss','phi3_radius'):
        table(f'Every per-episode {field} effect', effect_headers,
              [effect_row(n,label(e),e[field]) for n,p in comparisons.items() for e in p['episodes']])
    table('Every per-coordinate phi3 effect', effect_headers,
          [effect_row(n,f'{label(e)}/coordinate{j}',v)
           for n,p in comparisons.items() for e in p['episodes'] for j,v in enumerate(e['phi3_coordinates'])])
    labels = [label(e) for e in repeats[0]['episodes']]
    table('Wh decomposition', ['Episode','norm Wh','norm mean Wh','norm centered Wh'],
          [[l,v['wh'],v['common'],v['centered']] for l,v in zip(labels,algebra['per_episode_norms'])])
    names = ('full_original_output','bias_removed','common_mode_removed_upper_bound')
    table('Original and exactly two algebraic counterfactuals',
          ['Output','Total energy','Centered energy','Centered fraction','Numerical rank'],
          [[n]+[algebra[n][f] for f in ('total_energy','centered_energy','centered_energy_fraction','centered_numerical_rank')]
           for n in names])
    for name in names:
        v = algebra[name]
        table(name+' full pairwise distances', ['Episode']+labels,
              [[l]+r for l,r in zip(labels,v['pairwise_distances'])])
        table(name+' same-image separation', ['Image','Distance'],
              [[labels[2*i],d] for i,d in enumerate(v['same_image_distances'])])
        table(name+' different-image separation', ['Episode A','Episode B','Distance'],
              [[labels[d['episodes'][0]],labels[d['episodes'][1]],d['distance']] for d in v['different_image_distances']])
        table(name+' centered singular values', ['Index','Singular value'], enumerate(v['centered_singular_values']))
    table('Literal update reconstruction vs saved T013-C', ['Quantity','Max absolute error'],
          [[k,v] for k,v in algebra.items() if k.startswith('max_saved')])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.audit), encoding='utf-8')
