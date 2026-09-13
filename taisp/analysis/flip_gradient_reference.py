"""T027-A post-lock original-view oracle and frozen E1/E2 analysis."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .flip_gradient_consensus import norm,EPS
from .roi_equivariance import sha,write,setup,images,common_jvp,state_hash,isolated
from .oracle import detector_task_loss
from .common_jacobian import reference_checks


def score(task,g):
    return math.fsum(a*b for a,b in zip(task,g))/(norm(g)+EPS)


def describe(values):
    if not values:return {'n':0,'positive':0,'positive_fraction':None,'mean':None,'median':None,'quantiles':None}
    v=sorted(values)
    def quantile(q):
        x=(len(v)-1)*q;i=int(x);j=min(i+1,len(v)-1)
        return v[i]+(v[j]-v[i])*(x-i)
    count=sum(x>0 for x in v)
    return {'n':len(v),'positive':count,'positive_fraction':count/len(v),'mean':statistics.mean(v),
        'median':statistics.median(v),'quantiles':{str(q):quantile(q) for q in (0.,.25,.5,.75,1.)}}


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):result[order[k]]=(i+j-1)/2
        i=j
    return result


def spearman(x,y):
    a,b=ranks(x),ranks(y);ma,mb=statistics.mean(a),statistics.mean(b)
    a,b=[v-ma for v in a],[v-mb for v in b];den=norm(a)*norm(b)
    return math.fsum(u*v for u,v in zip(a,b))/den if den else None


def summarize(rows,threshold):
    def group(chosen):
        d={k:describe([r[k] for r in chosen]) for k in ('S_orig','S_cons','S_flip','Delta_cons','agreement')}
        return {'n':len(chosen),'distributions':d,'abstentions':sum(r['abstain'] for r in chosen),
            'both_views_nonzero':sum(r['both_nonzero'] for r in chosen)}
    def reliability(chosen):
        high=[r for r in chosen if r['agreement']>=threshold];low=[r for r in chosen if r['agreement']<threshold]
        h,l=describe([r['S_orig'] for r in high]),describe([r['S_orig'] for r in low])
        delta=h['positive_fraction']-l['positive_fraction'] if high and low else None
        med=h['median']-l['median'] if high and low else None
        rho=spearman([r['agreement'] for r in chosen],[r['S_orig'] for r in chosen])
        return {'spearman':rho,'high':h,'low':l,'positive_fraction_difference':delta,'median_difference':med,
            'correlation_pass':rho is not None and rho>=.30,'fraction_pass':delta is not None and delta>=.20,
            'median_pass':med is not None and med>0}
    corrupt=[r for r in rows if r['case']!='clean_s0'];a,b=group(rows),group(corrupt)
    blocks={str(i):group([r for r in rows if r['block']==i]) for i in range(4)}
    conditions={c:group([r for r in rows if r['case']==c]) for c in sorted({r['case'] for r in rows})}
    da,db=a['distributions'],b['distributions']
    E1={'positive_S_overall':da['S_cons']['positive']>=30,'positive_S_corrupt':db['S_cons']['positive']>=15,
        'positive_Delta_overall':da['Delta_cons']['positive']>=30,'positive_Delta_corrupt':db['Delta_cons']['positive']>=15,
        'median_overall':da['Delta_cons']['median']>0,'median_corrupt':db['Delta_cons']['median']>0,
        'blocks':sum(v['distributions']['Delta_cons']['median']>0 for v in blocks.values())>=3,
        'no_systematic_zero':a['abstentions']<=4 and b['abstentions']<=2,'integrity':all(r['integrity_passed'] for r in rows)}
    ra,rb=reliability(rows),reliability(corrupt)
    E2={f'{group_name}_{criterion}':r[criterion] for group_name,r in [('overall',ra),('corrupt',rb)]
        for criterion in ('correlation_pass','fraction_pass','median_pass')}
    E2.update({'coverage_overall':a['both_views_nonzero']>=44,'coverage_corrupt':b['both_views_nonzero']>=22})
    e1,e2=all(E1.values()),all(E2.values())
    return {'status':'NEEDS_REVIEW','overall':a,'corrupt':b,'conditions':conditions,'blocks':blocks,
        'pinned_median_agreement':threshold,'reliability':{'overall':ra,'corrupt':rb},'E1':E1,'E2':E2,
        'E1_passed':e1,'E2_passed':e2,'disposition':'nominate_consensus_for_review' if e1 else
            ('reliability_signal_only_for_review' if e2 else 'close_flip_gradient_consensus_and_reliability'),
        'AP_or_runtime_authorized':False}


def run(cohort_path,candidate_root,lock_path,output):
    lock=json.loads(lock_path.read_text());assert lock['candidate_commit'] and len(lock['files'])==53
    for f,h in lock['files'].items():assert sha(candidate_root/f)==h,f
    assert sha(cohort_path)==lock['cohort_sha256']
    candidates=json.loads((candidate_root/'records.json').read_text());assert len(candidates)==48
    split=json.loads((candidate_root/'agreement_split.json').read_text());threshold=split['overall_median_agreement']
    manifest=json.loads(cohort_path.read_text());output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();source,isp,env=setup()
    env.update({'candidate_commit':lock['candidate_commit'],'candidate_lock_sha256':sha(lock_path),
        'pinned_split_sha256':sha(candidate_root/'agreement_split.json'),'pinned_threshold':threshold,
        'ordering':'all48candidates/threshold checked before annotation load','cohort_sha256':sha(cohort_path)})
    write(output/'environment.json',env)
    assert sha(manifest['annotation_file'])==manifest['annotation_sha256']
    data=json.loads(Path(manifest['annotation_file']).read_text());annotations={i:[] for i in manifest['image_ids']}
    for ann in data['annotations']:
        if ann['image_id'] in annotations and not ann.get('iscrowd',0) and ann['bbox'][2]>0 and ann['bbox'][3]>0 and ann.get('area',1)>0:
            annotations[ann['image_id']].append(ann)
    del data
    rows=[]
    for index,(info,case,image) in enumerate(images(manifest)):
        c=candidates[index];assert (info['image_id'],case,info['block'])==(c['image_id'],c['case'],c['block'])
        anns=annotations[info['image_id']]
        boxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns]
        targets=[{'boxes':image.new_tensor(boxes).reshape(-1,4),'labels':torch.tensor([a['category_id'] for a in anns],device=image.device)}]
        y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
        loss,components=detector_task_loss(source,y,targets,seed=20260913)
        ct=torch.autograd.grad(loss,y)[0].detach();assert torch.isfinite(ct).all()
        refs,jacobian=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'task':ct})
        r=refs['task'];ts=r['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist()
        checks=reference_checks(ts,r['object'],r['background'],direct)
        so,sc,sf=score(ts,c['original']['gradient']),score(ts,c['g_cons']),score(ts,c['flipped']['gradient'])
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'task_loss':loss.item(),'task_components':{k:v.item() for k,v in components.items()},
            'task_gradient':ts,'jacobian':jacobian,'reference_checks':checks,'S_orig':so,'S_cons':sc,'S_flip':sf,
            'Delta_cons':sc-so,'agreement':c['agreement'],'both_nonzero':c['both_nonzero'],'abstain':c['abstain'],
            'group':'high' if c['agreement']>=threshold else 'low',
            'integrity_passed':c['isolation'] and all(c[v]['parity']['passed'] for v in ('original','flipped'))
                and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values())}
        write(output/f'record_{index:02d}.json',row);assert row['integrity_passed'],'Reference integrity blocker; stop.'
        rows.append(row);print(json.dumps({'stage':'reference','episode':index,'integrity':True}),flush=True)
    assert state_hash(source)==env['source_state_sha256'] and isolated(source)
    write(output/'records.json',rows);write(output/'summary.json',summarize(rows,threshold))
    write(output/'completion.json',{'status':'NEEDS_REVIEW','episodes':48,'source_hash_after':state_hash(source),
        'seconds':time.perf_counter()-started,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--candidate-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cohort,a.candidate_root,a.candidate_lock,a.output)
