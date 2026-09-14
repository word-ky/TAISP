"""R049 separate train-fit and post-map-lock holdout task-gradient processes."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch
from .orthogonal_candidates import rng_state
from .orthogonal_transport import fit_vectors,evaluate,summarize
from .roi_equivariance import setup,images,sha,write,common_jvp,state_hash,isolated


def verify_inputs(cohort_path,candidate_root,lock_path,phase,map_root=None,map_lock_path=None):
    lock=json.loads(lock_path.read_text());assert lock['candidate_commit'] and len(lock['files'])==483
    assert sha(cohort_path)==lock['cohort_sha256']
    assert sha(candidate_root/'sha256.json')==lock['manifest_sha256']
    for f,h in lock['files'].items():assert sha(candidate_root/f)==h,f
    candidates=json.loads((candidate_root/'records.json').read_text());manifest=json.loads(cohort_path.read_text())
    expected=[(im['image_id'],case,im['block'],im['partition']) for im,cc in zip(manifest['images'],manifest['corrupted_cases']) for case in ['clean_s0',cc]]
    assert len(candidates)==480 and [(c['image_id'],c['case'],c['block'],c['partition']) for c in candidates]==expected
    assert all(c['integrity_passed'] and c['hard']['parity']['passed'] and c['rng']['restored'] and c['finite'] for c in candidates)
    env=json.loads((candidate_root/'environment.json').read_text());completion=json.loads((candidate_root/'completion.json').read_text())
    assert env['source_revision']==lock['candidate_code_commit'][:7] and env['manifest_sha256']==lock['candidate_images_sha256']
    assert env['source_state_sha256']==completion['source_hash_after']==lock['source_state_sha256']
    assert not completion['GT_loaded'] and not completion['forbidden_modules_loaded']
    train={im['image_id'] for im in manifest['images'] if im['partition']=='train'}
    holdout={im['image_id'] for im in manifest['images'] if im['partition']=='holdout'}
    assert len(train)==180 and len(holdout)==60 and not train&holdout
    rotation=None;map_receipt=None
    if phase=='holdout':
        ml=json.loads(map_lock_path.read_text());assert ml['map_commit'] and ml['candidate_lock_sha256']==sha(lock_path)
        for f,h in ml['files'].items():assert sha(map_root/f)==h,f
        rotation=json.loads((map_root/'R.json').read_text());tr=json.loads((map_root/'records.json').read_text())
        assert len(tr)==360 and {r['image_id'] for r in tr}==train and all(r['partition']=='train' for r in tr)
        assert rotation['candidate_lock_sha256']==sha(lock_path) and rotation['train_reference_sha256']==sha(map_root/'records.json')
        assert rotation['train_image_ids']==sorted(train) and rotation['holdout_task_gradients_computed']==0
        map_receipt={'map_commit':ml['map_commit'],'map_sha256':sha(map_root/'R.json'),'map_lock_sha256':sha(map_lock_path)}
    chosen=[(im,cc) for im,cc in zip(manifest['images'],manifest['corrupted_cases']) if im['partition']==phase]
    selected={**manifest,'images':[im for im,cc in chosen],'corrupted_cases':[cc for im,cc in chosen]}
    cs=[c for c in candidates if c['partition']==phase]
    assert not any(k in sys.modules for k in ['taisp.analysis.oracle','taisp.analysis.common_jacobian'])
    return cs,selected,rotation,{'phase':phase,'verified_candidate_files':483,'verified_candidate_episodes':480,
        'candidate_lock_sha256':sha(lock_path),'candidate_records_sha256':sha(candidate_root/'records.json'),
        'source_state_sha256':lock['source_state_sha256'],'candidate_commit':lock['candidate_commit'],
        'cohort_sha256':sha(cohort_path),'train_holdout_disjoint':True,'before_annotations_and_oracle':True,'map_lock':map_receipt}


def run(cohort_path,candidate_root,lock_path,phase,output,map_root=None,map_lock_path=None):
    candidates,manifest,rotation,preflight=verify_inputs(cohort_path,candidate_root,lock_path,phase,map_root,map_lock_path)
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
        if phase=='holdout':row.update(evaluate(t,c['g_hard'],rotation['R']))
        write(output/f'record_{index:03d}.json',row);assert row['integrity_passed'],'Reference integrity blocker'
        rows.append(row);print(json.dumps({'stage':phase,'episode':index}),flush=True)
    assert len(rows)==(360 if phase=='train' else 120) and sha(candidate_root/'records.json')==preflight['candidate_records_sha256']
    write(output/'records.json',rows)
    if phase=='train':
        fit=fit_vectors([r['g_hard'] for r in rows],[r['task_gradient'] for r in rows])
        fit.update(train_image_ids=sorted(ids),train_reference_sha256=sha(output/'records.json'),candidate_lock_sha256=sha(lock_path),
            code_sha256=sha(Path(__file__)),fit_code_sha256=sha(Path(__file__).with_name('orthogonal_transport.py')),
            source_revision=env['source_revision'],holdout_task_gradients_computed=0)
        write(output/'R.json',fit)
    else:write(output/'summary.json',summarize(rows))
    write(output/'completion.json',{'status':'train_map_complete' if phase=='train' else 'NEEDS_REVIEW','episodes':len(rows),'phase':phase,
        'seconds':time.perf_counter()-started,'source_hash_after':state_hash(source),'candidate_recomputed':False,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--candidate-lock',type=Path,required=True)
    p.add_argument('--phase',choices=['train','holdout'],required=True);p.add_argument('--map-root',type=Path)
    p.add_argument('--map-lock',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.cohort,a.candidate_root,a.candidate_lock,a.phase,a.output,a.map_root,a.map_lock)
