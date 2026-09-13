"""R034 pre-AP same-cotangent equal-state pseudo/CLIP numerical receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from taisp import DifferentiableISP
from taisp.isp.spatial import compose,regional_gradients,support_mask
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions
from taisp.tta.spatial_dose import dose_step
from taisp.tta.trust_radius import transfer_norm
from .deterministic_replay import state_hash
from .differential_subspace import write_json
from .run_t002 import environment_metadata
from .run_t018a import load_image,CASES_ALL,corrupt
from .source_meta_smoke import frozen_unchanged


def vector_parity(value,reference):
    a,b=value.detach().double(),reference.detach().double()
    na,nb=a.norm().item(),b.norm().item()
    relative=(a-b).norm().item()/max(nb,1e-12)
    cosine=(a @ b).item()/(na*nb) if na and nb else (1. if na==nb==0 else 0.)
    return dict(passed=relative<=1e-5 and cosine>=.999999,relative_l2=relative,cosine=cosine,
                max_absolute=(a-b).abs().max().item(),value=a.tolist(),reference=b.tolist())


def clip_isolation(clip,initial_hash):
    return dict(clip_parameters_frozen_grad_none=all(not p.requires_grad and p.grad is None for p in clip.parameters()),
                clip_all_modules_eval=all(not m.training for m in clip.modules()),
                clip_state_hash_unchanged=state_hash(clip)==initial_hash)


def collect(config,manifest_path,output):
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed']);torch.set_num_threads(config['threads']);torch.backends.cudnn.benchmark=False
    manifest=json.loads(manifest_path.read_text());source=load_detector(config['device'])
    clip=load_clip_guidance(config['device'],local_files_only=True).eval().requires_grad_(False);isp=DifferentiableISP().to(config['device'])
    sh,ch=state_hash(source),state_hash(clip);states=[{k:v.clone() for k,v in source.state_dict().items()}]
    assert sh=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    meta=environment_metadata(config)
    old=json.loads(Path('/home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000/artifacts/study/environment.json').read_text())
    for k in ('clip_model','clip_revision','clip_sha256','detector_sha256','positive_prompts','negative_prompts'):
        assert (list(meta[k]) if k.endswith('prompts') else meta[k])==old[k]
    meta.update(source_state_sha256=sh,clip_state_sha256=ch,cohort_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                protocol='first2images/all7conditions/equalstates0and.01; same pseudo/CLIP image cotangents; no annotations/AP')
    write_json(output/'environment.json',meta);records=[];failed=None
    for info in manifest['images'][:2]:
        image0=load_image(info,config['device'])
        for family,severity in CASES_ALL:
            image=image0 if family=='clean' else corrupt(image0,family,severity)
            with torch.no_grad():chosen=select_predictions(source(image)[0],config['support_threshold'],config['support_topk'])
            mask,rects=support_mask(image,chosen['boxes']);native=DetectorNativeLoss(source,{'base':chosen},'det_pseudo').eval().requires_grad_(False)
            for initial in (0.,.01):
                phi=image.new_full((8,),initial,requires_grad=True);y=isp(image,phi)
                pseudo,cl= native(image,y),clip(image,y)
                cp,gp=torch.autograd.grad(pseudo,(y,phi),retain_graph=True)
                cc,gc=torch.autograd.grad(cl,(y,phi))
                regional=regional_gradients(image,isp,mask,torch.stack((phi.detach(),phi.detach())),{'pseudo':cp,'clip':cc})
                full=regional_gradients(image,isp,torch.ones_like(mask),torch.stack((phi.detach(),phi.detach())),{'pseudo':cp,'clip':cc})
                checks={};partitions={}
                for name,reference in [('pseudo',gp),('clip',gc)]:
                    go,gb=regional[name]
                    checks[name]=vector_parity(go.to(image)+gb.to(image),reference)
                    error=(go+gb-full[name][0]).abs();bound=1e-12+1e-10*(go.abs()+gb.abs())
                    partitions[name]=dict(passed=bool((error<=bound).all()),max_absolute=error.max().item(),
                        object=go.tolist(),background=gb.tolist(),global_jvp=full[name][0].tolist())
                delta,d=dose_step(*regional['pseudo'].to(image),*regional['clip'].to(image),lr=config['semantic_lr'],eps=config['radius_eps'])
                global_update,_=transfer_norm(gp.detach(),gc.detach(),config['radius_eps'])
                checks['mean_update_c0_limit']=vector_parity(delta.mean(0),-config['semantic_lr']*global_update)
                spatial=compose(image,isp,mask,phi.detach(),phi.detach())
                image_error=(spatial-y.detach()).abs()
                image_pass=bool((image_error<=2e-7+1e-6*y.detach().abs()).all())
                clip_norm_error=abs((regional['clip'].to(image).sum(0)).norm().item()-gc.norm().item())/max(gc.norm().item(),1e-12)
                isolation=dict(source_unchanged_frozen_eval_grad_none=frozen_unchanged((source,),states),
                    clip_frozen_eval_grad_none=all(not p.requires_grad and p.grad is None for p in clip.parameters()) and all(not m.training for m in clip.modules()),
                    isp_identity_grad_none=bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None))
                isolation.update(clip_isolation(clip,ch))
                passed=all(x['passed'] for x in checks.values()) and all(x['passed'] for x in partitions.values()) and image_pass and clip_norm_error<=1e-5 and all(isolation.values())
                row=dict(image_id=info['image_id'],case=f'{family}_s{severity}',equal_state=initial,support_count=len(chosen['boxes']),
                    mask_rectangles=rects,mask_area=mask.double().mean().item(),checks=checks,partitions=partitions,
                    image_max_absolute=image_error.max().item(),image_passed=image_pass,clip_norm_relative=clip_norm_error,
                    dose=d,isolation=isolation,passed=passed)
                records.append(row);write_json(output/f'record_{len(records):02d}.json',row)
                print(f'parity {len(records)}/28 image={info["image_id"]} case={family}_s{severity} state={initial} passed={passed}',flush=True)
                if not passed:failed=row;break
            if failed:break
        if failed:break
    hashes=dict(source_hash_unchanged=state_hash(source)==sh,clip_hash_unchanged=state_hash(clip)==ch)
    final_clip_isolation=clip_isolation(clip,ch)
    hashes.update(final_clip_isolation)
    completion=dict(status='blocked' if failed or not all(hashes.values()) else 'completed',records=len(records),
                    expected_records=28,all_passed=not failed and all(hashes.values()),hashes=hashes,evaluations=0)
    write_json(output/'completion.json',completion)
    if failed or not all(hashes.values()):
        write_json(output/'blocker.json',dict(reason='Predeclared T022-A numerical parity failed; no AP or tolerance changes',failed_record=failed,hashes=hashes))
        raise RuntimeError('T022-A pre-AP numerical blocker; stop for research review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/t022a.yaml'))
    p.add_argument('--manifest',type=Path,default=Path('research_log/T022A_train_cohort.json'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();collect(yaml.safe_load(a.config.read_text()),a.manifest,a.output)
