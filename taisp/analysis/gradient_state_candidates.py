"""R043 GT-free identity gradient-state records; no reference imports."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from taisp.losses.clip_semantic import (load_clip_guidance, CLIP_MODEL, CLIP_REVISION,
                                      POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
from .gradient_state_features import features, EPS
from .flip_gradient_consensus import view_gradient, norm
from .roi_equivariance import setup, images, sha, write, state_hash, isolated, common_jvp


def run(manifest_path, output):
    output.mkdir(parents=True, exist_ok=False)
    started=time.perf_counter(); source,isp,env=setup()
    clip=load_clip_guidance('cuda:0',local_files_only=True).eval()
    ch=state_hash(clip)
    assert ch=='6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c'
    env.update(clip_model=CLIP_MODEL,clip_revision=CLIP_REVISION,clip_state_sha256=ch,
        positive_prompts=POSITIVE_PROMPTS,negative_prompts=NEGATIVE_PROMPTS,
        manifest_sha256=sha(manifest_path),feature_dimension=21)
    write(output/'environment.json',env);rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        pseudo=view_gradient(source,isp,image)
        y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
        loss=clip(image,y);ct=torch.autograd.grad(loss,y)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(ct).all()
        refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'clip':ct})
        gc=refs['clip']['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist()
        ng,nd=norm(gc),norm(direct)
        cosine=math.fsum(a*b for a,b in zip(gc,direct))/(ng*nd) if ng and nd else (1. if ng==nd==0 else 0.)
        rel=norm([a-b for a,b in zip(gc,direct)])/max(ng,EPS)
        vector=features(pseudo['gradient'],gc,pseudo['loss'],loss.item())
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
             'partition':info['partition'],'pseudo':pseudo,'clip_gradient':gc,'clip_loss':loss.item(),
             'clip_jacobian':jac,'clip_direct_reverse':direct,
             'clip_parity':{'cosine':cosine,'relative_l2':rel,'passed':cosine>=.999999 and rel<=1e-5},
             'norm_pseudo':norm(pseudo['gradient']),'norm_clip':ng,'features':vector,
             'isolation':isolated(source) and isolated(clip) and isp.phi.grad is None and not bool(isp.phi.any())}
        write(output/f'record_{index:03d}.json',row)
        assert len(vector)==21 and all(math.isfinite(v) for v in vector)
        assert row['isolation'] and pseudo['parity']['passed'] and row['clip_parity']['passed']
        rows.append(row)
        print(json.dumps({'stage':'candidate','episode':index,'integrity':True}),flush=True)
    assert len(rows)==360 and state_hash(source)==env['source_state_sha256'] and state_hash(clip)==ch
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','source_meta','common_jacobian','_reference','memory'))]
    assert not forbidden,forbidden
    write(output/'records.json',rows)
    write(output/'completion.json',{'status':'candidate_complete','episodes':360,'source_hash_after':state_hash(source),
        'clip_hash_after':state_hash(clip),'forbidden_modules_loaded':forbidden,'seconds':time.perf_counter()-started,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)
