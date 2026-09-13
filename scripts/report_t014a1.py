"""Render saved common-Jacobian replay and spatial geometry; stdlib only."""
import argparse
import json
from pathlib import Path


def render(summary,records,stage_c):
    lines=['# T014-A1 numerical replay and full R024 geometry tables','','Only the fresh32 full records contribute to scientific summaries. Debug samples are numerical checks. No finite-step spatial adaptation or AP measurement.','']

    def table(title,headers,rows):
        lines.extend(['## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(format(v,'.12g') if isinstance(v,float) else str(v) for v in row)+' |')
        lines.append('')

    def numerical_rows(rows):
        for row in rows:
            for name in ('task','pseudo'):
                r=row[name]
                p,c=r['reference_checks']['partition'],r['reference_checks']['parity']
                old=r['reverse']['checks']['regional_gradient_sum']
                yield [row['episode_index'],row['image_id'],row['case'],name,p['max_absolute_error'],p['max_error_over_bound'],
                       c['cosine'],c['relative_l2_error'],old['passed'],old['max_absolute_error'],old['max_error_over_bound']]

    headers=['Episode','Image','Case','Objective','Closure max error','Closure error/bound','Parity cosine','Parity relative L2','Old reverse check','Old max error','Old error/bound']
    table('Stage C fixed numerical debug',headers,numerical_rows(stage_c['records']))
    table('Full32 numerical checks',headers,numerical_rows(records))
    table('Primary gate quantities by scope',['Scope','N','Median R_task','Median Delta_D','Positive Delta_D','Empty supports','Mean mask area'],
          [[name,v['n'],v['metrics']['R_task']['median'],v['metrics']['Delta_D']['median'],v['metrics']['Delta_D']['positive_count'],
            v['empty_support_count'],v['mask_area_mean']] if v['n'] else [name,0,None,None,0,0,None]
           for name,v in summary['scopes'].items()])
    table('All spatial metrics by scope',['Scope','N','Metric','Mean','Median','Min','Max','Positive count'],
          [[name,v['n'],metric]+[m[k] for k in ('mean','median','min','max','positive_count')]
           for name,v in summary['scopes'].items() for metric,m in v['metrics'].items()])
    metric_names=['task_regional_cosine','pseudo_regional_cosine','task_cancellation','R_task','A_global','A_spatial','Delta_A','D_global','D_spatial','Delta_D']
    table('All per-episode scientific metrics',['Episode','Image','Case','Block','Mask area','Support']+metric_names,
          [[r['episode_index'],r['image_id'],r['case'],r['block'],r['mask_area_fraction'],r['support_count']]+[r['geometry'][k] for k in metric_names] for r in records])
    table('Loss and saturation diagnostics',['Episode','Objective','Loss','Identity saturation','Source/ISP isolation'],
          [[r['episode_index'],name,r[name]['reverse']['loss'],r[name]['reverse']['saturation_at_identity'],all(r['isolation'].values())]
           for r in records for name in ('task','pseudo')])
    table('All fresh reference and reverse vectors',['Episode','Objective','Estimator','Region']+[f'phi{k}' for k in range(8)],
          [[r['episode_index'],name,estimator,region]+r[name][estimator][region] for r in records
           for name in ('task','pseudo') for estimator in ('reference','reverse') for region in ('global','object','background')])
    table('Unchanged R024 advancement gate',['Flag','Passed'],summary['gate']['flags'].items())
    lines.extend([summary['gate']['decision'],'',
                  'Stage-C and full raw receipts retain image checks, all closure error/bound vectors, parity, Jacobian norms/cotangent hashes, original mask rectangles and model/ISP isolation. The original T014-A failed receipt remains unchanged.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    a=args.audit_root
    args.output.write_text(render(json.loads((a/'summary.json').read_text()),json.loads((a/'records.json').read_text())['records'],
                                  json.loads((a/'stage_c.json').read_text())),encoding='utf-8')
