"""R038 saved corrected common-Jacobian vectors only; standard-library binary64."""
import argparse
import hashlib
import json
import math
import platform
import statistics as st
import sys
import time
from pathlib import Path
EPS=1e-12


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b,strict=True))
def norm(a):return math.sqrt(dot(a,a))
def cosine(a,b):return dot(a,b)/(norm(a)*norm(b)) if norm(a) and norm(b) else None


def closure(obj,bg,saved):
    value=[a+b for a,b in zip(obj,bg,strict=True)]
    errors=[abs(a-b) for a,b in zip(value,saved,strict=True)]
    bounds=[1e-12+1e-10*(abs(a)+abs(b)) for a,b in zip(obj,bg,strict=True)]
    return value,dict(passed=all(a<=b for a,b in zip(errors,bounds)),max_absolute=max(errors),max_error_over_bound=max(a/b for a,b in zip(errors,bounds)))


def utility(po,pb,to,tb):
    ps=[a+b for a,b in zip(po,pb,strict=True)];ts=[a+b for a,b in zip(to,tb,strict=True)]
    so=dot(to,po)/(norm(po)+EPS);sg=dot(ts,ps)/(norm(ps)+EPS)
    return dict(p_o=po,p_b=pb,t_o=to,t_b=tb,p_s=ps,t_s=ts,S_obj=so,S_global=sg,DeltaS=so-sg,
        cos_obj=cosine(to,po),cos_global=cosine(ts,ps),norms={k:norm(v) for k,v in dict(p_o=po,p_b=pb,t_o=to,t_b=tb,p_s=ps,t_s=ts).items()})


def describe(rows):
    metrics={}
    for key in ['S_obj','S_global','DeltaS','cos_obj','cos_global']:
        v=[r[key] for r in rows if r[key] is not None]
        metrics[key]=dict(defined=len(v),undefined=len(rows)-len(v),positive=sum(x>0 for x in v),zero=sum(x==0 for x in v),mean=st.mean(v) if v else None,median=st.median(v) if v else None,min=min(v) if v else None,max=max(v) if v else None)
    return dict(n=len(rows),metrics=metrics,zero_norms={k:sum(r['norms'][k]==0 for r in rows) for k in ['p_o','p_b','p_s','t_o','t_b','t_s']})


def decision(rows,integrity):
    corrupt=[r for r in rows if r['case']!='clean_s0']
    block_medians=[st.median(r['DeltaS'] for r in rows if r['block']==b) for b in range(4)]
    flags=dict(record_hash_and_closure=integrity,S_obj_overall_ge20=sum(r['S_obj']>0 for r in rows)>=20,
        S_obj_corrupted_ge10=sum(r['S_obj']>0 for r in corrupt)>=10,DeltaS_overall_ge20=sum(r['DeltaS']>0 for r in rows)>=20,
        DeltaS_corrupted_ge10=sum(r['DeltaS']>0 for r in corrupt)>=10,median_DeltaS_overall_positive=st.median(r['DeltaS'] for r in rows)>0,
        median_DeltaS_corrupted_positive=st.median(r['DeltaS'] for r in corrupt)>0,at_least3positive_block_medians=sum(v>0 for v in block_medians)>=3)
    return dict(flags=flags,passed=all(flags.values()),block_medians=block_medians,decision='object_only_candidate_pending_research_review' if all(flags.values()) else 'close_object_only_action',status='NEEDS_REVIEW')


def run(project,output):
    start=time.perf_counter();plan_path=project/'research_log/T023A_plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
    assert sys.float_info.mant_dig==53 and sys.float_info.max_exp==1024
    root=project/plan['input_root'];checks={}
    for name,want in plan['input_manifest']['files'].items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest();checks[name]=dict(expected=want,actual=actual,passed=actual==want)
    assert all(v['passed'] for v in checks.values())
    methods={}
    for name,want in plan['protected_modules_LF'].items():
        actual=hashlib.sha256((project/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest();methods[name]=actual==want
    assert all(methods.values())
    raw=json.loads((root/'records.json').read_text())['records'];assert len(raw)==32
    assert [r['episode_index'] for r in raw]==list(range(32)) and [r['block'] for r in raw]==[i//8 for i in range(32)]
    assert sum(r['case']=='clean_s0' for r in raw)==16
    rows=[]
    for r in raw:
        saved=json.loads((root/'full'/f"record_{r['episode_index']:02d}.json").read_text())
        assert r['numerical_passed'] and all(r['isolation'].values())
        assert all(r[k]==saved[k] for k in ['image_id','case','block','support_count','mask_area_fraction','mask_uint8_sha256'])
        refs={k:r[k]['reference'] for k in ['pseudo','task']};cl={}
        for k,v in refs.items():
            assert v==saved[k]['reference'] and all(len(v[a])==8 and all(math.isfinite(x) for x in v[a]) for a in ['object','background','global'])
            _,cl[k]=closure(v['object'],v['background'],v['global']);assert cl[k]['passed']
        u=utility(refs['pseudo']['object'],refs['pseudo']['background'],refs['task']['object'],refs['task']['background'])
        area=r['mask_area_fraction'];count=r['support_count']
        area_bin='zero' if area==0 else '(0,.01]' if area<=.01 else '(.01,.05]' if area<=.05 else '(.05,.2]' if area<=.2 else '(.2,1]'
        support_bin='zero' if count==0 else '1-5' if count<=5 else '6-10' if count<=10 else '11-20' if count<=20 else '>20'
        rows.append({**{k:r[k] for k in ['episode_index','image_id','case','block','support_count','mask_area_fraction','mask_uint8_sha256']},**u,'closure':cl,'mask_bin':area_bin,'support_bin':support_bin})
    groups=dict(overall=rows,clean=[r for r in rows if r['case']=='clean_s0'],corrupted=[r for r in rows if r['case']!='clean_s0'])
    groups.update({f'block{b}':[r for r in rows if r['block']==b] for b in range(4)})
    groups.update({'case/'+c:[r for r in rows if r['case']==c] for c in sorted({r['case'] for r in rows})})
    for k,field in [('mask_bins','mask_bin'),('support_bins','support_bin')]:groups.update({field+'/'+v:[r for r in rows if r[field]==v] for v in plan[k]})
    scopes={k:describe(v) for k,v in groups.items()};gate=decision(rows,True)
    output.mkdir(parents=True,exist_ok=True)
    def write(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write('records.json',rows);write('summary.json',dict(scopes=scopes,gate=gate));write('input_checks.json',checks)
    write('environment.json',dict(python=platform.python_version(),device='CPU',arithmetic='IEEE754 binary64 Python float, math.fsum dot products',eps=EPS,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),protected_module_checks=methods,new_model_calls=0,evaluations=0))
    write('completion.json',dict(status='NEEDS_REVIEW',episodes=32,elapsed_seconds=time.perf_counter()-start,gate=gate,max_closure_error=max(v['max_absolute'] for r in rows for v in r['closure'].values()),new_model_calls=0,evaluations=0))
    print(json.dumps(dict(gate=gate,overall=scopes['overall'],corrupted=scopes['corrupted'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.project,a.output)
