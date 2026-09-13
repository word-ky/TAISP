"""Offline summaries of all predeclared R036 comparisons; no thresholds."""
import argparse
import json
from pathlib import Path


def aggregate(pairs):
    return {key:dict(pair_count=len(pairs),exact_pairs=sum(p['fields'][key]['exact'] for p in pairs),
        max_absolute=max(p['fields'][key]['max_absolute'] for p in pairs),
        max_relative_l2=max(p['fields'][key]['relative_l2'] for p in pairs),
        minimum_cosine=min((p['fields'][key]['cosine'] for p in pairs if p['fields'][key]['cosine'] is not None),default=None))
        for key in pairs[0]['fields']}


def summarize(root,out):
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    result=dict(completion=read('completion.json'),fixed_cotangents={str(r['step']):aggregate(r['pairs']) for r in read('fixed_cotangent_pairs.json')},
        regional_jvp={str(r['step'])+'/'+r['mask']:aggregate(r['pairs']) for r in read('jvp_pairs.json')})
    if (root/'dose_pairs.json').exists():result['dose']=aggregate(read('dose_pairs.json'))
    if (root/'runtime_pairs.json').exists():result['runtime']={r['case']:aggregate(r['pairs']) for r in read('runtime_pairs.json')}
    if 'runtime' in result:
        empty=result['runtime']['empty_support']
        result['empty_support_state_and_image_exact']=all(empty[k]['exact_pairs']==10 for k in ('final_state','final_image'))
        result['empty_support_pseudo_cotangents_exact']=all(empty[f's{step}_cp']['exact_pairs']==10 for step in range(4))
        result['raw_completion_note']='Raw terminal control check includes CLIP cotangent variability; evaluate state/image reset separately, without changing raw records or tolerances.'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.output)
