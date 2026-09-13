"""T027-A GT-free global pseudo-gradient consensus, analysis only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

from taisp import DifferentiableISP
from taisp.losses.detector_native import DetectorNativeLoss
from .roi_equivariance import (sha,write,setup,images,common_jvp,isolated,state_hash,select_predictions)


EPS=1e-12
STATES=[[0.]*8,[.1,-.2,.3,-.1,.2,-.3,.05,.15],[.2]*8,[-.15]*8]


def norm(x):
    return math.sqrt(math.fsum(v*v for v in x))


def consensus(go,gf):
    no,nf=norm(go),norm(gf)
    if no<=EPS or nf<=EPS:
        return {'norm_original':no,'norm_flip':nf,'agreement':-1.,'g_cons':[0.]*8,
                'both_nonzero':False,'abstain':True,'reason':'zero_view','sum_unit_norm':None}
    uo,uf=[v/no for v in go],[v/nf for v in gf]
    agreement=math.fsum(a*b for a,b in zip(uo,uf));q=[a+b for a,b in zip(uo,uf)];nq=norm(q)
    return {'norm_original':no,'norm_flip':nf,'agreement':agreement,
        'g_cons':[v/nq for v in q] if nq>EPS else [0.]*8,'both_nonzero':True,
        'abstain':nq<=EPS,'reason':'antiparallel' if nq<=EPS else None,'sum_unit_norm':nq}


def preflight(device='cuda:0'):
    torch.manual_seed(2701);torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    isp=DifferentiableISP().to(device);rows=[]
    for shape in [(1,3,31,37),(1,3,16,18)]:
        x=torch.rand(shape,device=device);x[...,0,0]=0.;x[...,-1,-1]=1.
        for state in STATES:
            p=x.new_tensor(state)
            a=isp(x,p).flip(-1);b=isp(x.flip(-1),p)
            rows.append({'shape':shape,'phi':state,'passed':bool(torch.allclose(a,b,atol=2e-7,rtol=1e-6)),
                'max_abs_error':(a-b).abs().max().item(),'max_error_over_bound':((a-b).abs()/(2e-7+1e-6*b.abs())).max().item()})
    return {'passed':all(r['passed'] for r in rows),'records':rows,'device':device,'seed':2701,
        'atol':2e-7,'rtol':1e-6,'model_calls':0,'code_sha256_LF':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}


def view_gradient(source,isp,image):
    with torch.no_grad():support=select_predictions(source(image)[0],.5,20)
    native=DetectorNativeLoss(source,{'base':support},'det_pseudo')
    y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
    loss=native(image,y);c=torch.autograd.grad(loss,y)[0].detach()
    assert torch.isfinite(loss) and torch.isfinite(c).all()
    refs,jacobian=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'pseudo':c})
    g=refs['pseudo']['global'];phi=image.new_zeros(8,requires_grad=True)
    direct=torch.autograd.grad((isp(image,phi)*c).sum(),phi)[0].cpu().tolist()
    ng,nd=norm(g),norm(direct)
    cosine=math.fsum(a*b for a,b in zip(g,direct))/(ng*nd) if ng and nd else (1. if ng==nd==0 else 0.)
    rel=norm([a-b for a,b in zip(g,direct)])/max(ng,EPS)
    supports={k:v.cpu().tolist() for k,v in support.items()}
    receipt={'supports':supports,'support_count':len(support['boxes']),
        'supports_sha256':hashlib.sha256(json.dumps(supports,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'loss':loss.item(),'gradient':g,'jacobian':jacobian,'direct_reverse':direct,
        'parity':{'cosine':cosine,'relative_l2':rel,'passed':cosine>=.999999 and rel<=1e-5},
        'cotangent_l2':c.double().norm().item()}
    return receipt


def run(manifest_path,preflight_path,output):
    assert json.loads(preflight_path.read_text())['passed']
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source,isp,env=setup();env.update({'manifest_sha256':sha(manifest_path),'preflight_sha256':sha(preflight_path)})
    write(output/'environment.json',env);rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        original=view_gradient(source,isp,image);flipped=view_gradient(source,isp,image.flip(-1))
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'original':original,'flipped':flipped,**consensus(original['gradient'],flipped['gradient']),
            'isolation':isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())}
        write(output/f'record_{index:02d}.json',row)
        assert row['isolation'] and original['parity']['passed'] and flipped['parity']['passed'],'Candidate numerical/isolation blocker; stop.'
        rows.append(row);print(json.dumps({'stage':'candidate','episode':index,'abstain':row['abstain']}),flush=True)
    assert len(rows)==48 and state_hash(source)==env['source_state_sha256']
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('oracle','source_meta','common_jacobian','_reference','memory'))]
    assert not forbidden,forbidden
    write(output/'records.json',rows)
    write(output/'agreement_split.json',{'overall_median_agreement':statistics.median(r['agreement'] for r in rows),
        'high_rule':'>= overall pinned median','low_rule':'< overall pinned median','episodes':48,'GT_loaded':False})
    write(output/'completion.json',{'status':'candidate_complete','episodes':48,'source_hash_after':state_hash(source),
        'frozen_eval_grad_none':isolated(source),'forbidden_modules_loaded':forbidden,'seconds':time.perf_counter()-started})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='mode',required=True)
    pf=sp.add_parser('preflight');pf.add_argument('--output',type=Path,required=True)
    ca=sp.add_parser('candidate');ca.add_argument('--manifest',type=Path,required=True)
    ca.add_argument('--preflight-receipt',type=Path,required=True);ca.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='preflight':
        result=preflight();write(a.output,result);assert result['passed'],'ISP flip commutation failed; BLOCKED.'
    else:run(a.manifest,a.preflight_receipt,a.output)
