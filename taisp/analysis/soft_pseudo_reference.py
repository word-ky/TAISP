"""T029-A post-lock task reference and frozen hard/soft alignment gate."""
import argparse
import json
from pathlib import Path
import time
import torch
from .flip_gradient_reference import score, describe
from .soft_pseudo_candidates import norm
import math
from .roi_equivariance import sha,write,setup,images,common_jvp,state_hash,isolated
from .oracle import detector_task_loss
from .common_jacobian import reference_checks


def summarize(rows):
    def group(chosen):
        keys=['S_hard','S_soft','Delta','norm_hard','norm_soft','norm_ratio','gradient_cosine']
        return {'n':len(chosen),'distributions':{k:describe([r[k] for r in chosen if r[k] is not None]) for k in keys},
                'zero_hard':sum(r['norm_hard']==0 for r in chosen),'zero_soft':sum(r['norm_soft']==0 for r in chosen),
                'undefined_cosine':sum(r['gradient_cosine'] is None for r in chosen),
                'undefined_ratio':sum(r['norm_ratio'] is None for r in chosen)}
    a=group(rows);b=group([r for r in rows if r['case']!='clean_s0']);clean=group([r for r in rows if r['case']=='clean_s0'])
    blocks={str(i):group([r for r in rows if r['block']==i]) for i in range(4)}
    conditions={c:group([r for r in rows if r['case']==c]) for c in sorted({r['case'] for r in rows})}
    da,db=a['distributions'],b['distributions']
    flags={'S_soft_overall':da['S_soft']['positive']>=80,'S_soft_corrupt':db['S_soft']['positive']>=45,
        'Delta_overall':da['Delta']['positive']>=68,'Delta_corrupt':db['Delta']['positive']>=35,
        'median_overall':da['Delta']['median']>0,'median_corrupt':db['Delta']['median']>0,
        'positive_blocks':sum(v['distributions']['Delta']['median']>0 for v in blocks.values())>=3,
        'median_clean':clean['distributions']['Delta']['median']>=0,'integrity':all(r['integrity_passed'] for r in rows)}
    passed=all(flags.values())
    return {'status':'NEEDS_REVIEW','overall':a,'corrupt':b,'clean':clean,'conditions':conditions,'blocks':blocks,
        'gates':flags,'passed':passed,'AP_calls':0,'runtime_authorized':False,
        'disposition':'power2_soft_pseudo_alignment_pending_review' if passed else 'close_exact_power2_soft_pseudo_family'}


def run(cohort_path,candidate_root,lock_path,output):
    lock=json.loads(lock_path.read_text());assert lock['candidate_commit'] and len(lock['files'])==123
    for f,h in lock['files'].items():assert sha(candidate_root/f)==h,f
    assert sha(cohort_path)==lock['cohort_sha256']
    candidates=json.loads((candidate_root/'records.json').read_text());assert len(candidates)==120
    manifest=json.loads(cohort_path.read_text());output.mkdir(parents=True,exist_ok=False)
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
        loss,components=detector_task_loss(source,y,targets,seed=20260913)
        ct=torch.autograd.grad(loss,y)[0].detach();assert torch.isfinite(ct).all()
        refs,jacobian=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'task':ct})
        r=refs['task'];ts=r['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist()
        checks=reference_checks(ts,r['object'],r['background'],direct)
        gh,gs=c['objectives']['hard']['gradient'],c['objectives']['soft']['gradient']
        nh,ns=norm(gh),norm(gs)
        sh,ss=score(ts,gh),score(ts,gs)
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'task_loss':loss.item(),'task_components':{k:v.item() for k,v in components.items()},
            'task_gradient':ts,'jacobian':jacobian,'reference_checks':checks,'S_hard':sh,'S_soft':ss,'Delta':ss-sh,
            'norm_hard':nh,'norm_soft':ns,'norm_ratio':ns/nh if nh else None,
            'gradient_cosine':math.fsum(a*b for a,b in zip(gh,gs))/(nh*ns) if nh and ns else None,
            'integrity_passed':c['isolation'] and c['support_match'] and c['target_valid'] and all(v['parity']['passed'] for v in c['objectives'].values())
                and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values())}
        write(output/f'record_{index:02d}.json',row);assert row['integrity_passed'],'Reference integrity blocker; stop.'
        rows.append(row);print(json.dumps({'stage':'reference','episode':index,'integrity':True}),flush=True)
    assert state_hash(source)==env['source_state_sha256'] and isolated(source)
    write(output/'records.json',rows);write(output/'summary.json',summarize(rows))
    write(output/'completion.json',{'status':'NEEDS_REVIEW','episodes':120,'source_hash_after':state_hash(source),
        'seconds':time.perf_counter()-started,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--candidate-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cohort,a.candidate_root,a.candidate_lock,a.output)
