"""T028-A post-lock original-task reference; source labels for capacity audit only."""
import argparse
import json
from pathlib import Path
import time
import torch
from .flip_gradient_reference import score
from .roi_equivariance import sha,write,setup,images,common_jvp,state_hash,isolated
from .oracle import detector_task_loss
from .common_jacobian import reference_checks


def run(cohort_path,candidate_root,lock_path,output):
    lock=json.loads(lock_path.read_text());assert lock['candidate_commit'] and len(lock['files'])==363
    for f,h in lock['files'].items():assert sha(candidate_root/f)==h,f
    assert sha(cohort_path)==lock['cohort_sha256']
    candidates=json.loads((candidate_root/'records.json').read_text());assert len(candidates)==360
    manifest=json.loads(cohort_path.read_text());output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();source,isp,env=setup()
    env.update({'candidate_commit':lock['candidate_commit'],'candidate_lock_sha256':sha(lock_path),
        'ordering':'all360features checked before annotation load','cohort_sha256':sha(cohort_path)})
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
        so=score(ts,c['pseudo']['gradient'])
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'partition':info['partition'],'task_loss':loss.item(),'task_components':{k:v.item() for k,v in components.items()},
            'task_gradient':ts,'jacobian':jacobian,'reference_checks':checks,'S_orig':so,'y':1 if so>0 else -1,
            'integrity_passed':c['isolation'] and c['pseudo']['parity']['passed'] and c['clip_parity']['passed']
                and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values())}
        write(output/f'record_{index:02d}.json',row);assert row['integrity_passed'],'Reference integrity blocker; stop.'
        rows.append(row);print(json.dumps({'stage':'reference','episode':index,'integrity':True}),flush=True)
    assert state_hash(source)==env['source_state_sha256'] and isolated(source)
    write(output/'records.json',rows)
    write(output/'completion.json',{'status':'reference_complete','episodes':360,'source_hash_after':state_hash(source),
        'seconds':time.perf_counter()-started,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--candidate-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cohort,a.candidate_root,a.candidate_lock,a.output)
