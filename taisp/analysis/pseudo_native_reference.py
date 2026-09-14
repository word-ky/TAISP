"""T030-A post-lock task reference and frozen hard/native alignment gate."""
import argparse
import json
from pathlib import Path
import time
import torch
import statistics
import sys
from .native_vjp_attribution import digest,rng_state
from .pseudo_native_candidates import norm
from .native_task_signal import KEYS
import math
from .roi_equivariance import sha,write,setup,images,common_jvp,state_hash,isolated


def score(task,g):
    return math.fsum(a*b for a,b in zip(task,g))/(norm(g)+1e-12)


def describe(values):
    if not values:return {'n':0,'positive':0,'positive_fraction':None,'mean':None,'median':None,'quantiles':None}
    v=sorted(values)
    def quantile(q):
        x=(len(v)-1)*q;i=int(x);j=min(i+1,len(v)-1)
        return v[i]+(v[j]-v[i])*(x-i)
    count=sum(x>0 for x in v)
    return {'n':len(v),'positive':count,'positive_fraction':count/len(v),'mean':statistics.mean(v),
        'median':statistics.median(v),'quantiles':{str(q):quantile(q) for q in (0.,.25,.5,.75,1.)}}



def summarize(rows):
    def group(chosen):
        keys=['S_hard','S_native','Delta','norm_hard','norm_native','norm_ratio','gradient_cosine']+[f'{prefix}_{k}' for k in KEYS for prefix in ['S','norm','cos_task']]
        return {'n':len(chosen),'distributions':{k:describe([r[k] for r in chosen if r[k] is not None]) for k in keys},
                'zero_hard':sum(r['norm_hard']==0 for r in chosen),'zero_native':sum(r['norm_native']==0 for r in chosen),
                'undefined_cosine':sum(r['gradient_cosine'] is None for r in chosen),
                'undefined_ratio':sum(r['norm_ratio'] is None for r in chosen)}
    a=group(rows);b=group([r for r in rows if r['case']!='clean_s0']);clean=group([r for r in rows if r['case']=='clean_s0'])
    blocks={str(i):group([r for r in rows if r['block']==i]) for i in range(4)}
    conditions={c:group([r for r in rows if r['case']==c]) for c in sorted({r['case'] for r in rows})}
    da,db=a['distributions'],b['distributions']
    flags={'S_native_overall':da['S_native']['positive']>=80,'S_native_corrupt':db['S_native']['positive']>=45,
        'Delta_overall':da['Delta']['positive']>=68,'Delta_corrupt':db['Delta']['positive']>=35,
        'median_overall':da['Delta']['median']>0,'median_corrupt':db['Delta']['median']>0,
        'positive_blocks':sum(v['distributions']['Delta']['median']>0 for v in blocks.values())>=3,
        'median_clean':clean['distributions']['Delta']['median']>=0,'integrity':all(r['integrity_passed'] for r in rows)}
    passed=all(flags.values())
    return {'status':'NEEDS_REVIEW','overall':a,'corrupt':b,'clean':clean,'conditions':conditions,'blocks':blocks,
        'gates':flags,'passed':passed,'AP_calls':0,'runtime_authorized':False,
        'disposition':'four_loss_native_pseudo_alignment_pending_review' if passed else 'close_exact_four_loss_native_pseudo_family'}


def candidate_integrity(c):
    # Component decomposition is a retained diagnostic, prospectively demoted by R047.
    return (c['isolation'] and c['support_match'] and c['target_match'] and c['rng']['restored']
        and all(v['parity']['passed'] and v['parity']['relative_l2']<=1e-5 and v['parity']['cosine']>=.999999
            and len(v['gradient'])==8 and all(math.isfinite(x) for x in v['gradient']) and math.isfinite(v['loss'])
            for v in c['objectives'].values())
        and c['supports']['boxes']==c['pseudo_targets']['boxes'] and c['supports']['labels']==c['pseudo_targets']['labels']
        and digest(c['supports'])==c['supports_sha256'])


def verify_inputs(cohort_path,candidate_root,lock_path):
    lock=json.loads(lock_path.read_text());project=lock_path.resolve().parents[1]
    assert lock['reviewed_commit']=='f9ca28e5ab56375c06e63c97700b97574acba775'
    assert lock['candidate_commit']=='59efa11b99648c74d952918b3de3bebfce6df64e'
    assert sha(candidate_root/'records.json')==lock['candidate_records_sha256']=='031731120f517c03bfb8a6a5ae595676ae26b6bb6ba6076cbcfde00be6a7bdb6'
    assert sha(candidate_root/'sha256.json')==lock['candidate_manifest_sha256']=='e5b019de2034f1697c5e630538869bf487674ca9cd06b10a3ed489bbdf6deac7'
    assert sha(cohort_path)==lock['cohort_sha256']=='3ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752'
    assert len(lock['files'])==123 and json.loads((candidate_root/'sha256.json').read_text())==lock['files']
    for f,h in lock['files'].items():assert sha(candidate_root/f)==h,f
    for f,h in lock['extra_files'].items():assert sha(project/f)==h,f
    prior=json.loads((project/'research_log/T030A2/integrity.json').read_text());assert prior['candidate_all120_integrity'] and prior['source_state_final_same']
    env=json.loads((candidate_root/'environment.json').read_text());completion=json.loads((candidate_root/'completion.json').read_text())
    assert env['source_revision']==lock['candidate_commit'][:7] and env['native_sampling_seed']==20260930
    assert env['source_state_sha256']==completion['source_hash_after']==lock['source_state_sha256']
    assert completion['episodes']==120 and not completion['GT_loaded'] and not completion['forbidden_modules_loaded'] and completion['AP_calls']==0
    image_manifest=project/'research_log/T030A/candidate_images.json'
    assert sha(image_manifest)==env['manifest_sha256']
    manifest=json.loads(cohort_path.read_text());im=json.loads(image_manifest.read_text())
    assert im['images']==manifest['images'] and im['corrupted_cases']==manifest['corrupted_cases']
    candidates=json.loads((candidate_root/'records.json').read_text());assert len(candidates)==120
    order=[(info['image_id'],case,info['block']) for info,cc in zip(manifest['images'],manifest['corrupted_cases']) for case in ['clean_s0',cc]]
    assert [(c['image_id'],c['case'],c['block']) for c in candidates]==order
    assert all(c['episode_index']==i and candidate_integrity(c) and c==json.loads((candidate_root/f'record_{i:03d}.json').read_text()) for i,c in enumerate(candidates))
    forbidden=[k for k in sys.modules if k in ['taisp.analysis.oracle','taisp.analysis.common_jacobian','taisp.analysis.flip_gradient_reference']]
    assert not forbidden,forbidden
    return candidates,manifest,{'candidate_all120_integrity':True,'candidate_files_verified':123,'order_verified':120,
        'reviewed_commit':lock['reviewed_commit'],'candidate_commit':lock['candidate_commit'],
        'candidate_records_sha256':lock['candidate_records_sha256'],'candidate_manifest_sha256':lock['candidate_manifest_sha256'],
        'cohort_sha256':lock['cohort_sha256'],'component_additivity_diagnostic_pass_count':sum(c['component_sum']['passed'] for c in candidates),
        'before_annotation_load':True,'oracle_modules_before_preflight':forbidden,'candidate_recomputed':False}


def run(cohort_path,candidate_root,lock_path,output):
    candidates,manifest,preflight=verify_inputs(cohort_path,candidate_root,lock_path)
    output.mkdir(parents=True,exist_ok=False);write(output/'preflight.json',preflight)
    # The complete immutable lock is verified before either oracle import or annotation load.
    from .oracle import detector_task_loss
    from .common_jacobian import reference_checks
    lock=json.loads(lock_path.read_text())
    started=time.perf_counter();source,isp,env=setup()
    env.update({'candidate_commit':lock['candidate_commit'],'candidate_lock_sha256':sha(lock_path),
        'ordering':'all120candidates checked before annotation load','cohort_sha256':sha(cohort_path)})
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
        before=state_hash(source);rng_before=rng_state(image.device)
        loss,components=detector_task_loss(source,y,targets,seed=20260913)
        ct=torch.autograd.grad(loss,y)[0].detach();assert torch.isfinite(ct).all() and torch.isfinite(loss)
        rng_after=rng_state(image.device);after=state_hash(source)
        refs,jacobian=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'task':ct})
        r=refs['task'];ts=r['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist()
        checks=reference_checks(ts,r['object'],r['background'],direct)
        gh,gs=c['objectives']['hard']['gradient'],c['objectives']['native']['gradient']
        nh,ns=norm(gh),norm(gs)
        sh,ss=score(ts,gh),score(ts,gs)
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'task_loss':loss.item(),'task_components':{k:v.item() for k,v in components.items()},
            'task_gradient':ts,'jacobian':jacobian,'reference_checks':checks,'S_hard':sh,'S_native':ss,'Delta':ss-sh,
            'norm_hard':nh,'norm_native':ns,'norm_ratio':ns/nh if nh else None,
            'gradient_cosine':math.fsum(a*b for a,b in zip(gh,gs))/(nh*ns) if nh and ns else None,
            'candidate_vectors_read_from_lock':True,'candidate_hard_gradient':gh,'candidate_native_gradient':gs,
            'component_sum_diagnostic':c['component_sum'],'rng':{'before':rng_before,'after':rng_after,'restored':rng_before==rng_after},
            'state_before':before,'state_after':after,
            'integrity_passed':preflight['candidate_all120_integrity'] and rng_before==rng_after and before==after==env['source_state_sha256']
                and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values())}
        for key in KEYS:
            g=c['objectives'][key]['gradient'];n=norm(g)
            row['S_'+key]=score(ts,g);row['norm_'+key]=n
            row['cos_task_'+key]=math.fsum(a*b for a,b in zip(ts,g))/(norm(ts)*n) if norm(ts) and n else None
        write(output/f'record_{index:02d}.json',row);assert row['integrity_passed'],'Reference integrity blocker; stop.'
        rows.append(row);print(json.dumps({'stage':'reference','episode':index,'integrity':True}),flush=True)
    assert state_hash(source)==env['source_state_sha256'] and isolated(source)
    assert sha(candidate_root/'records.json')==lock['candidate_records_sha256']
    assert len(rows)==120
    write(output/'records.json',rows);write(output/'summary.json',summarize(rows))
    write(output/'completion.json',{'status':'NEEDS_REVIEW','episodes':120,'source_hash_after':state_hash(source),
        'seconds':time.perf_counter()-started,'AP_calls':0,'candidate_recomputed':False,'reference_seed':20260913,'candidate_records_sha256':lock['candidate_records_sha256']})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--candidate-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cohort,a.candidate_root,a.candidate_lock,a.output)
