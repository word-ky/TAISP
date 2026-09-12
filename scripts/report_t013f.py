"""Render all retained T013-F matched-cycle results without model execution."""
import argparse
import json
from pathlib import Path


def render(root):
    summary = json.loads((root/'summary.json').read_text())
    freeze = json.loads((root/'checkpoint_freeze.json').read_text())
    cycles = [json.loads((root/f'cycle_{i:02d}.json').read_text()) for i in range(1,9)]
    lines = ['# T013-F complete matched-cycle tables', '',
             'Negative loss delta is improvement. All eight cycles retained.',
             'Correction = probe minus original, minus same-cycle null B minus A.',
             'Resolved iff >=7/8 corrected signs agree with median and abs median > max abs null.',
             'This is a fixed microset repeatability diagnostic, not population significance.', '']

    def table(title, headers, rows):
        lines.extend([f'## {title}', '', '| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(format(x,'.10g') if isinstance(x,float) else str(x) for x in row)+' |')
        lines.append('')

    table('Frozen checkpoints', ['Checkpoint','File SHA256','Parameter delta norm'],
          [[n,v['file_sha256'],v['parameter_delta_norm']] for n,v in freeze['checkpoints'].items()])
    table('Actual saved schedule', ['Cycle','Pair block order','Inside-pair order'],
          [[c['cycle'],', '.join(c['pair_order']),', '.join(c['role_order'])] for c in cycles])
    table('Every null effect', ['Group']+[str(i) for i in range(1,9)],
          [[g,v['signed']['values'][0],*v['signed']['values'][1:]] for g,v in summary['null'].items()])
    table('Null summaries', ['Group','Median','Min','Max','Negative','Zero','Positive','Min abs','Max abs','Abs range'],
          [[g]+[v['signed'][f] for f in ('median','min','max','negative_count','zero_count','positive_count')]+
           [v['absolute'][f] for f in ('min','max','range')] for g,v in summary['null'].items()])
    for kind in ('paired','corrected'):
        table(f'Every {kind} cycle effect', ['Probe','Group']+[str(i) for i in range(1,9)],
              [[p,g]+v[kind]['values'] for p,groups in summary['effects'].items() for g,v in groups.items()])
        headers = ['Probe','Group','Median','Min','Max','Negative','Zero','Positive']
        fields = ('median','min','max','negative_count','zero_count','positive_count')
        if kind == 'corrected':
            headers += ['Median sign / 8','Max abs null','Resolved','Direction']
            fields += ('median_sign_count','max_absolute_null','resolved','direction')
        table(f'{kind} summaries', headers,
              [[p,g]+[v[kind][f] for f in fields] for p,groups in summary['effects'].items() for g,v in groups.items()])
    table('All per-episode paired deltas', ['Cycle','Pair','Episode index','Delta'],
          [[c['cycle'],p,i,d] for c in summary['cycles'] for p,v in c['paired'].items()
           for i,d in enumerate(v['episode_loss_deltas'])])
    rows = [r for c in cycles for p in c['pairs'] for e in p['evaluations'] for r in e['episodes']]
    table('All 512 raw episode evaluations',
          ['Cycle','Block','Pair','Eval position','Role','Image','Case','Loss','phi0 norm','phi3 norm','Saturation','Empty support'],
          [[r[f] for f in ('cycle','block_position','pair','evaluation_position','role','image_id','case','outer_loss',
                           'phi0_norm','phi3_norm','saturation_rate','empty_support')] for r in rows])
    lines.extend(['## Predeclared decision', '', '```json', json.dumps(summary['decision'],indent=2), '```', '',
                  'Full phi0/phi3 coordinate vectors, source loss components and isolation hashes are in each raw evaluation JSON.', ''])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.audit),encoding='utf-8')
