"""Render preserved T017-A numerical receipts without interpreting blocked science."""
import argparse
import json
from pathlib import Path


def render(rows):
    lines=['# T017-A numerical receipts', '', 'No attribution is inferred from an incomplete cohort.', '']
    def table(title, headers, values):
        lines.extend(['## '+title, '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
        for row in values:
            lines.append('| '+' | '.join(format(v,'.12g') if isinstance(v,float) else str(v) for v in row)+' |')
        lines.append('')
    table('Independent component-sum closure', ['Episode','Region','Passed','Max error','Max error/bound'],
          [[r['episode_index'],region,c['passed'],c['max_absolute_error'],c['max_error_over_bound']]
           for r in rows for region,c in r['numerical_checks']['component_closure'].items()])
    table('All coordinate reconstruction errors', ['Episode','Region','Coordinate','Component sum','Independent total','Error','Bound'],
          [[r['episode_index'],region,i,c['sum'][i],c['reference'][i],c['errors'][i],c['bounds'][i]]
           for r in rows for region,c in r['numerical_checks']['component_closure'].items() for i in range(8)])
    table('Saved A1 parity', ['Episode','Region','Estimator','Passed','Cosine','Relative L2','Max error'],
          [[r['episode_index'],region,name,c['passed'],c['cosine'],c['relative_l2_error'],c['max_absolute_error']]
           for r in rows for region,v in r['numerical_checks']['saved_A1_parity'].items() for name,c in v.items()])
    table('All component and independent-total reference vectors', ['Episode','Objective','Region']+[f'phi{i}' for i in range(8)],
          [[r['episode_index'],name,region]+v for r in rows for name,values in r['references'].items() for region,v in values.items()])
    table('Losses', ['Episode','Component','Value'], [[r['episode_index'],k,v] for r in rows for k,v in r['losses'].items()])
    table('Direct parity, partition and isolation', ['Episode','Direct parity','Partitions','Isolation'],
          [[r['episode_index'],r['numerical_checks']['direct_global_parity'],
            {k:v['passed'] for k,v in r['numerical_checks']['partitions'].items()},r['isolation']] for r in rows])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted(args.audit_root.glob('record_[0-9][0-9].json'))]
    args.output.write_text(render(rows),encoding='utf-8')
