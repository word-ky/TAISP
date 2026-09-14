"""R045 annotation-free pseudo-native sum and diagnostic component gradients."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
from taisp.losses.detector_native import DetectorNativeLoss
from .native_task_signal import KEYS,pseudo_targets,pseudo_native_loss
from .roi_equivariance import setup,images,select_predictions,common_jvp,sha,write,isolated,state_hash


def norm(g):return math.sqrt(math.fsum(v*v for v in g))


def component_sum_check(values):
    total=values['native']['gradient']
    summed=[math.fsum(values[k]['gradient'][i] for k in KEYS) for i in range(8)]
    errors=[abs(a-b) for a,b in zip(total,summed)]
    bounds=[1e-7+1e-4*math.fsum(abs(values[k]['gradient'][i]) for k in KEYS) for i in range(8)]
    return {'passed':all(e<=b for e,b in zip(errors,bounds)),'absolute_errors':errors,
            'bounds':bounds,'max_error_over_bound':max(e/b for e,b in zip(errors,bounds)),
            'atol':1e-7,'rtol_sum_component_magnitudes':1e-4}


def gradients(image,isp,source,hard,targets):
    y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
    hard_loss=hard(image,y);ch=torch.autograd.grad(hard_loss,y)[0].detach()
    y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
    before_cpu=torch.random.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state(image.device).clone() if image.is_cuda else None
    total,parts=pseudo_native_loss(source,y,targets)
    rng_ok=torch.equal(before_cpu,torch.random.get_rng_state()) and (before_cuda is None or torch.equal(before_cuda,torch.cuda.get_rng_state(image.device)))
    assert set(parts)==set(KEYS)
    scalars={'native':total,**{k:parts[k] for k in KEYS}}
    cotangents={'hard':ch};losses={'hard':hard_loss.item()}
    for j,(key,loss) in enumerate(scalars.items()):
        cotangents[key]=torch.autograd.grad(loss,y,retain_graph=j<len(scalars)-1)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(cotangents[key]).all()
        losses[key]=loss.item()
    refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),cotangents)
    values={}
    for key,ct in cotangents.items():
        g=refs[key]['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist()
        n,nd=norm(g),norm(direct)
        cos=math.fsum(a*b for a,b in zip(g,direct))/(n*nd) if n and nd else (1. if n==nd==0 else 0.)
        rel=norm([a-b for a,b in zip(g,direct)])/max(n,1e-12)
        values[key]={'gradient':g,'norm':n,'loss':losses[key],'direct_reverse':direct,
            'parity':{'cosine':cos,'relative_l2':rel,'passed':cos>=.999999 and rel<=1e-5}}
    return values,jac,{'restored':rng_ok,'sampling_seed':20260930,
        'cpu_rng_before_sha256':hashlib.sha256(before_cpu.cpu().numpy().tobytes()).hexdigest(),
        'cuda_rng_before_sha256':hashlib.sha256(before_cuda.cpu().numpy().tobytes()).hexdigest() if before_cuda is not None else None}


def run(manifest_path,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source,isp,env=setup();env.update(manifest_sha256=sha(manifest_path),native_sampling_seed=20260930,components=KEYS)
    write(output/'environment.json',env);rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        with torch.no_grad():support=select_predictions(source(image)[0],.5,20)
        hard=DetectorNativeLoss(source,{'base':support},'det_pseudo');targets=pseudo_targets(hard)
        values,jac,rng=gradients(image,isp,source,hard,targets)
        target_match=torch.equal(targets[0]['boxes'],hard.boxes) and torch.equal(targets[0]['labels'],hard.labels) and all(not v.requires_grad for v in targets[0].values())
        support_match=torch.equal(hard.boxes,support['boxes']) and torch.equal(hard.labels,support['labels'])
        supports={k:v.cpu().tolist() for k,v in support.items()};sumcheck=component_sum_check(values)
        gh,gn=values['hard']['gradient'],values['native']['gradient'];nh,nn=norm(gh),norm(gn)
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'support_count':len(hard.boxes),'supports':supports,
            'supports_sha256':hashlib.sha256(json.dumps(supports,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'pseudo_targets':{k:v.cpu().tolist() for k,v in targets[0].items()},'hard_weights':hard.weights.cpu().tolist(),
            'objectives':values,'jacobian':jac,'rng':rng,'component_sum':sumcheck,
            'hard_native_cosine':math.fsum(a*b for a,b in zip(gh,gn))/(nh*nn) if nh and nn else None,
            'target_match':target_match,'support_match':support_match,
            'isolation':isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())}
        write(output/f'record_{index:03d}.json',row)
        assert target_match and support_match and row['isolation'] and rng['restored'] and sumcheck['passed'] and all(v['parity']['passed'] for v in values.values())
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
