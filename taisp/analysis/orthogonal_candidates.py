"""R049 unchanged original-view hard gradients, no annotations or task oracle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
from .flip_gradient_consensus import view_gradient,norm
from .roi_equivariance import setup,images,sha,write,state_hash,isolated


def rng_state(device):
    states={'cpu':torch.random.get_rng_state()}
    if device.type=='cuda':states['cuda']=torch.cuda.get_rng_state(device)
    return {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in states.items()}


def candidate(source,isp,image):
    before=state_hash(source);rng_before=rng_state(image.device)
    hard=view_gradient(source,isp,image)
    after=state_hash(source);rng_after=rng_state(image.device);g=hard['gradient']
    isolation=isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())
    finite=all(math.isfinite(v) for v in g) and math.isfinite(hard['loss'])
    return {'hard':hard,'g_hard':g,'norm_hard':norm(g),'zero':norm(g)==0,'finite':finite,
        'rng':{'before':rng_before,'after':rng_after,'restored':rng_before==rng_after},
        'state_before':before,'state_after':after,'isolation':isolation,
        'integrity_passed':before==after and rng_before==rng_after and isolation and finite and hard['parity']['passed']}


def run(manifest_path,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();source,isp,env=setup()
    env.update(manifest_sha256=sha(manifest_path),objective='unchanged_current_hard',model_seed=20260913)
    write(output/'environment.json',env);rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],'partition':info['partition'],**candidate(source,isp,image)}
        write(output/f'record_{index:03d}.json',row);assert row['integrity_passed'],'Candidate integrity blocker'
        rows.append(row);print(json.dumps({'stage':'candidate','episode':index}),flush=True)
    assert len(rows)==480 and state_hash(source)==env['source_state_sha256']
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('oracle','reference','source_meta','common_jacobian','clip_semantic','memory','native_task_signal','soft_pseudo'))]
    assert not forbidden,forbidden
    write(output/'records.json',rows)
    write(output/'completion.json',{'status':'candidate_complete','episodes':480,'seconds':time.perf_counter()-started,
        'source_hash_after':state_hash(source),'forbidden_modules_loaded':forbidden,'AP_calls':0,'GT_loaded':False})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)
