"""Render retained T013-C measurements into review tables; no model execution."""
import argparse
import json
from pathlib import Path


def render(receipt):
    audit, probes = receipt['gradient_audit'], receipt['probes']
    episodes = audit['episodes']
    lines = ['# T013-C complete diagnostic tables', '',
             'Episode order is fixed; C=clean, X=corrupted. Loss deltas are after minus before.', '']
    labels = [f"{r['image_id']}{'C' if r['case']=='clean_s0' else 'X'}" for r in episodes]

    def table(title, header, rows):
        lines.extend([f'## {title}', '', '| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |'])
        for row in rows:
            values = [format(v, '.9g') if isinstance(v, float) else str(v) for v in row]
            lines.append('| '+' | '.join(values)+' |')
        lines.append('')

    table('Full phi0-gradient cosine matrix', ['Episode']+labels,
          [[label]+row for label, row in zip(labels, audit['geometry']['phi_pairwise_cosines'])])
    table('Every phi0 gradient vector', ['Episode','gamma','red','green','blue','contrast','brightness','tone','sharpening'],
          [[label]+row['phi0_gradient'] for label, row in zip(labels, episodes)])
    table('Per-episode gradient scales', ['Episode','Loss','phi0 norm','head W norm','head b norm','trunk norm'],
          [[label,r['outer_loss'],r['phi0_gradient_norm'],r['head_weight_gradient_norm'],
            r['head_bias_gradient_norm'],r['trunk_gradient_norm']] for label,r in zip(labels,episodes)])
    for name, probe in probes.items():
        delta = probe['delta_comparison']
        table(f'{name}: predictions and actual deltas', ['Episode','Before','After','Predicted delta','Actual delta','Same sign'],
              [[label,b['outer_loss'],a['outer_loss'],p,d,s] for label,b,a,p,d,s in zip(
                  labels,episodes,probe['episodes'],delta['predicted'],delta['actual'],delta['sign_agreement'])])
        table(f'{name}: state and saturation', ['Episode','phi0 norm','phi3 norm','Saturation','Support count'],
              [[label,r['phi0_norm'],r['phi3_norm'],r['saturation_rate'],r['support_count']]
               for label,r in zip(labels,probe['episodes'])])
    c = probes['joint']['conditioning']
    table('Joint one-step output decomposition', ['Episode','norm Wh','norm b','norm phi0'],
          [[label,w,c['bias_norm'],v] for label,w,v in zip(labels,c['weight_term_norms'],c['output_norms'])])
    table('Joint one-step full pairwise output distances', ['Episode']+labels,
          [[label]+row for label,row in zip(labels,c['pairwise_distances'])])
    repeat = receipt['original_no_update_repeat']
    table('Original predictor no-update repeat', ['Episode','Baseline loss','Repeat loss','Delta','Baseline phi3 norm','Repeat phi3 norm'],
          [[label,b['outer_loss'],r['outer_loss'],d,b['phi3_norm'],r['phi3_norm']]
           for label,b,r,d in zip(labels,episodes,repeat['episodes'],repeat['loss_deltas'])])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.receipt.read_text(encoding='utf-8'))), encoding='utf-8')
