"""R051 unchanged hard gradient plus own-support frozen ROI object state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from .orthogonal_candidates import candidate as hard_candidate, rng_state
from .roi_equivariance import setup,images,sha,write,state_hash,isolated,fixed_roi_representation


def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pool_features(z,scores):
    if not len(scores):return z.new_zeros(1024)
    unit=F.normalize(z,p=2,dim=-1,eps=1e-12)
    mean=(scores[:,None]*unit).sum(0)/(scores.sum()+1e-12)
    return F.normalize(mean,p=2,dim=0,eps=1e-12)


@torch.no_grad()
def object_state(source,isp,image,support):
    boxes=image.new_tensor(support['boxes']).reshape(-1,4)
    scores=image.new_tensor(support['scores'])
    if len(boxes):
        identity=isp(image,image.new_zeros(8))
        z,_=fixed_roi_representation(source,identity,boxes)
        assert z.shape==(len(boxes),1024) and torch.isfinite(z).all()
    else:z=image.new_zeros(0,1024)
    d=pool_features(z,scores)
    return {'d_obj':d.cpu().tolist(),'descriptor_sha256':tensor_sha(d),'descriptor_norm':d.double().norm().item(),
            'roi_feature_sha256':[tensor_sha(v) for v in z],'support_count':len(boxes),
            'mean_score':scores.mean().item() if len(scores) else 0.,
            'descriptor_finite':bool(torch.isfinite(d).all()),'descriptor_uses_classes':False}


def candidate(source,isp,image):
    hard=hard_candidate(source,isp,image)
    obj=object_state(source,isp,image,hard['hard']['supports'])
    after=state_hash(source);rng_after=rng_state(image.device)
    integrity=hard['integrity_passed'] and after==hard['state_before'] and rng_after==hard['rng']['before']
    integrity=integrity and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()) and obj['descriptor_finite']
    return {**hard,**obj,'state_after':after,'rng':{**hard['rng'],'after':rng_after,'restored':rng_after==hard['rng']['before']},
            'integrity_passed':bool(integrity),'descriptor_support_sha256':hard['hard']['supports_sha256']}


def run(manifest_path,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();source,isp,env=setup()
    env.update(manifest_sha256=sha(manifest_path),objective='unchanged_current_hard_plus_own_ROI_state',model_seed=20260913)
    write(output/'environment.json',env);rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
             'partition':info['partition'],**candidate(source,isp,image)}
        write(output/f'record_{index:03d}.json',row);assert row['integrity_passed'],'Candidate integrity blocker'
        rows.append(row);print(json.dumps({'stage':'candidate','episode':index,'supports':row['support_count']}),flush=True)
    assert len(rows)==600 and state_hash(source)==env['source_state_sha256']
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in
        ('annotation','oracle','reference','source_meta','common_jacobian','clip_semantic','memory','native_task_signal','soft_pseudo'))]
    assert not forbidden,forbidden
    write(output/'records.json',rows)
    write(output/'completion.json',{'status':'candidate_complete','episodes':600,'seconds':time.perf_counter()-started,
        'source_hash_after':state_hash(source),'GT_loaded':False,'forbidden_modules_loaded':forbidden,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)
