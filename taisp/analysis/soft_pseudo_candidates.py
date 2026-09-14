"""T029-A hard/soft identity gradients with a single shared support; GT-free."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
from taisp.losses.detector_native import DetectorNativeLoss
from .soft_pseudo import SoftPseudoLoss
from .roi_equivariance import setup,images,select_predictions,common_jvp,sha,write,isolated,state_hash


def norm(g):return math.sqrt(math.fsum(x*x for x in g))


def gradients(image,isp,objectives):
    losses={};cotangents={}
    for name,objective in objectives.items():
        y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
        loss=objective(image,y);cotangents[name]=torch.autograd.grad(loss,y)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(cotangents[name]).all()
        losses[name]=loss.item()
    refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),cotangents)
    results={}
    for name,c in cotangents.items():
        g=refs[name]['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*c).sum(),phi)[0].cpu().tolist()
        n,nd=norm(g),norm(direct)
        cosine=math.fsum(a*b for a,b in zip(g,direct))/(n*nd) if n and nd else (1. if n==nd==0 else 0.)
        relative=norm([a-b for a,b in zip(g,direct)])/max(n,1e-12)
        results[name]={'gradient':g,'norm':n,'loss':losses[name],'direct_reverse':direct,
            'parity':{'cosine':cosine,'relative_l2':relative,'passed':cosine>=.999999 and relative<=1e-5}}
    return results,jac


def run(manifest_path,output):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    source,isp,env=setup();env.update(manifest_sha256=sha(manifest_path),alpha=2,classes=91)
    write(output/'environment.json',env);rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        with torch.no_grad():support=select_predictions(source(image)[0],.5,20)
        hard=DetectorNativeLoss(source,{'base':support},'det_pseudo');soft=SoftPseudoLoss(hard,image)
        values,jac=gradients(image,isp,{'hard':hard,'soft':soft})
        supports={k:v.cpu().tolist() for k,v in support.items()}
        q=soft.target
        support_match=torch.equal(hard.boxes,soft.boxes) and torch.equal(hard.weights,soft.weights)
        target_valid=not q.requires_grad and bool(torch.isfinite(q).all()) and bool((q>=0).all()) and bool(torch.allclose(q.sum(-1),q.new_ones(len(q)),atol=2e-7,rtol=1e-6))
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'support_count':len(soft.boxes),'support_ids':list(range(len(soft.boxes))),
            'supports':supports,'supports_sha256':hashlib.sha256(json.dumps(supports,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'weights':hard.weights.cpu().tolist(),'original_logits':soft.original_logits.cpu().tolist(),
            'q':q.cpu().tolist(),'objectives':values,'jacobian':jac,'support_match':support_match,
            'target_valid':target_valid,'isolation':isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())}
        write(output/f'record_{index:03d}.json',row)
        assert row['isolation'] and support_match and target_valid and all(v['parity']['passed'] for v in values.values())
        rows.append(row);print(json.dumps({'stage':'candidate','episode':index,'integrity':True}),flush=True)
    assert len(rows)==120 and state_hash(source)==env['source_state_sha256']
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','reference','source_meta','clip_semantic','memory'))]
    assert not forbidden,forbidden
    write(output/'records.json',rows)
    write(output/'completion.json',{'status':'candidate_complete','episodes':120,'seconds':time.perf_counter()-started,
        'source_hash_after':state_hash(source),'forbidden_modules_loaded':forbidden,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)
