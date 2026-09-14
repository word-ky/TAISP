"""R051 representation-locked train fit and committed-direction holdout reveal."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch
from .object_state_stages import candidates,locked_representation,read,vector_sha
from .object_state_math import fit_affine,corrected,evaluate,summary
from .orthogonal_candidates import rng_state
from .roi_equivariance import setup,images,sha,write,common_jvp,state_hash,isolated


def verify(cohort_path,candidate_root,lock_path,representation_root,representation_lock,phase,directions_root,directions_lock):
    cs,lock=candidates(candidate_root,lock_path)
    assert sha(cohort_path)==lock['cohort_sha256'];full=read(cohort_path)
    expected=[(im['image_id'],case,im['block'],im['partition']) for im,cc in zip(full['images'],full['corrupted_cases']) for case in ['clean_s0',cc]]
    assert [(r['image_id'],r['case'],r['block'],r['partition']) for r in cs]==expected
    rep,states,rl=locked_representation(representation_root,representation_lock,lock_path)
    assert [r['episode_index'] for r in states]==list(range(600))
    assert [r['partition'] for r in states]==[r['partition'] for r in cs]
    dirs=None;direction_receipt=None
    if phase=='holdout':
        dl=read(directions_lock);assert dl['directions_commit']
        for f,h in dl['files'].items():assert sha(directions_root/f)==h,f
        done=read(directions_root/'completion.json');dirs=read(directions_root/'directions.json')
        assert done['representation_lock_sha256']==sha(representation_lock) and done['candidate_lock_sha256']==sha(lock_path)
        assert done['task_gradients_computed']==0 and not done['GT_loaded'] and done['model_commit']==dl['model_commit']
        assert len(dirs)==120 and all((d['episode_index'],d['g_hard'])==(c['episode_index'],c['g_hard']) for d,c in zip(dirs,cs[480:]))
        assert all(vector_sha(d['u_obj'])==d['u_obj_sha256'] for d in dirs)
        direction_receipt={**done,'directions_commit':dl['directions_commit'],
            'directions_sha256':sha(directions_root/'directions.json'),'directions_lock_sha256':sha(directions_lock)}
    chosen=[(im,cc) for im,cc in zip(full['images'],full['corrupted_cases']) if im['partition']==phase]
    manifest={**full,'images':[im for im,cc in chosen],'corrupted_cases':[cc for im,cc in chosen]}
    selected=[c for c in cs if c['partition']==phase]
    zs=[s['h_std'] for s in states if s['partition']==phase]
    assert not any(k in sys.modules for k in ['taisp.analysis.oracle','taisp.analysis.common_jacobian'])
    preflight={'phase':phase,'candidate_lock_sha256':sha(lock_path),'candidate_records_sha256':sha(candidate_root/'records.json'),
        'candidate_commit':lock['candidate_commit'],'source_state_sha256':lock['source_state_sha256'],
        'representation_commit':rl['representation_commit'],'representation_lock_sha256':sha(representation_lock),
        'train_holdout_disjoint':True,'before_annotations_and_oracle':True,'map_lock':None,'direction_lock':direction_receipt}
    return selected,manifest,zs,dirs,preflight


def run(cohort_path,candidate_root,lock_path,representation_root,representation_lock,phase,output,directions_root=None,directions_lock=None):
    candidates,manifest,states,directions,preflight=verify(cohort_path,candidate_root,lock_path,
        representation_root,representation_lock,phase,directions_root,directions_lock)
    output.mkdir(parents=True,exist_ok=False);write(output/'preflight.json',preflight)
    from .oracle import detector_task_loss
    from .common_jacobian import reference_checks
    started=time.perf_counter();source,isp,env=setup()
    assert env['source_state_sha256']==preflight['source_state_sha256']
    env.update(phase=phase,candidate_lock_sha256=sha(lock_path),cohort_sha256=sha(cohort_path),map_lock=preflight['map_lock'])
    annotation=manifest['partition_annotations'][phase];assert sha(annotation['path'])==annotation['sha256']
    data=json.loads(Path(annotation['path']).read_text());ids={im['image_id'] for im in manifest['images']}
    assert {im['id'] for im in data['images']}==ids==set(annotation['image_ids'])
    assert all(a['image_id'] in ids for a in data['annotations'])
    env['annotation_path']=annotation['path'];env['annotation_sha256']=annotation['sha256'];write(output/'environment.json',env)
    anns={i:[] for i in ids}
    for a in data['annotations']:
        if not a.get('iscrowd',0) and a['bbox'][2]>0 and a['bbox'][3]>0 and a.get('area',1)>0:anns[a['image_id']].append(a)
    del data
    rows=[]
    for index,(info,case,image) in enumerate(images(manifest)):
        c=candidates[index];assert (c['image_id'],c['case'])==(info['image_id'],case)
        aa=anns[info['image_id']];boxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa]
        targets=[{'boxes':image.new_tensor(boxes).reshape(-1,4),'labels':torch.tensor([a['category_id'] for a in aa],device=image.device)}]
        before=state_hash(source);rb=rng_state(image.device)
        y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
        loss,parts=detector_task_loss(source,y,targets,seed=20260913);ct=torch.autograd.grad(loss,y)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(ct).all()
        refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'task':ct})
        r=refs['task'];t=r['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist();checks=reference_checks(t,r['object'],r['background'],direct)
        ra=rng_state(image.device);after=state_hash(source)
        row={'episode_index':c['episode_index'],'image_id':info['image_id'],'case':case,'block':info['block'],'partition':phase,
            'g_hard':c['g_hard'],'task_gradient':t,'task_loss':loss.item(),'task_components':{k:v.item() for k,v in parts.items()},
            'jacobian':jac,'reference_checks':checks,'state_before':before,'state_after':after,'rng':{'before':rb,'after':ra,'restored':rb==ra},
            'candidate_recomputed':False,'integrity_passed':before==after==env['source_state_sha256'] and rb==ra and isolated(source)
                and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values()) and all(math.isfinite(v) for v in t)}
        row['support_count']=c['support_count']
        if phase=='holdout':
            row.update(u_obj=directions[index]['u_obj'])
            row.update(evaluate(t,c['g_hard'],directions[index]['u_obj']))
        write(output/f'record_{index:03d}.json',row);assert row['integrity_passed'],'Reference integrity blocker'
        rows.append(row);print(json.dumps({'stage':phase,'episode':index}),flush=True)
    assert len(rows)==(480 if phase=='train' else 120) and sha(candidate_root/'records.json')==preflight['candidate_records_sha256']
    write(output/'records.json',rows)
    if phase=='train':
        model=fit_affine(states,[r['g_hard'] for r in rows],[r['task_gradient'] for r in rows])
        model.update(candidate_lock_sha256=sha(lock_path),representation_lock_sha256=sha(representation_lock),
            train_reference_sha256=sha(output/'records.json'),holdout_task_gradients_computed=0,
            source_revision=env['source_revision'],code_sha256=sha(Path(__file__)),
            math_code_sha256=sha(Path(__file__).with_name('object_state_math.py')))
        write(output/'model.json',model)
        predictions=corrected(states,[r['g_hard'] for r in rows],model)
        write(output/'train_predictions.json',[{'episode_index':r['episode_index'],
            **{k:v[i] for k,v in predictions.items()},'target':model['target'][i]} for i,r in enumerate(rows)])
    else:write(output/'summary.json',summary(rows))
    write(output/'completion.json',{'status':'tangent_model_complete' if phase=='train' else 'NEEDS_REVIEW',
        'episodes':len(rows),'phase':phase,'seconds':time.perf_counter()-started,
        'source_hash_after':state_hash(source),'candidate_recomputed':False,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','candidate-root','candidate-lock','representation-root','representation-lock','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['train','holdout'],required=True)
    for name in ['directions-root','directions-lock']:p.add_argument('--'+name,type=Path)
    a=p.parse_args();run(a.cohort,a.candidate_root,a.candidate_lock,a.representation_root,a.representation_lock,
        a.phase,a.output,a.directions_root,a.directions_lock)
