"""R036 analysis-only attribution; exact frozen runtime, zero AP."""
import argparse
import gzip
import hashlib
import itertools
import json
import traceback
from pathlib import Path

import torch
import yaml

from taisp import DifferentiableISP
from taisp.isp.spatial import compose,regional_gradients,support_mask
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.tta.spatial_dose import adapt_spatial_dose,dose_step
from taisp.tta.trust_radius import adapt_clip_radius
from .deterministic_replay import state_hash
from .differential_subspace import write_json
from .run_t002 import environment_metadata
from .run_t018a import load_image,corrupt


def compare(a,b):
    a,b=a.detach().double().flatten(),b.detach().double().flatten()
    na,nb=a.norm().item(),b.norm().item();delta=a-b
    return dict(exact=torch.equal(a,b),max_absolute=delta.abs().max().item(),
        relative_l2=delta.norm().item()/max(na,1e-12),
        cosine=(a@b).item()/(na*nb) if na and nb else (1. if na==nb==0 else None))


def pairs(repeats):
    return [dict(first=i,second=j,fields={k:compare(a[k],b[k]) for k in a})
            for (i,a),(j,b) in itertools.combinations(enumerate(repeats),2)]


def save_tensors(path,data):
    with gzip.open(path,'wb',compresslevel=1) as f:
        torch.save({k:v.detach().cpu() for k,v in data.items()},f)


class CotangentRecorder:
    """Observe existing loss backwards with identity hooks; no extra model calls."""
    def __init__(self,pseudo,clip):
        self.handles=[];self.images=[];self.records={'pseudo':[],'clip':[]};self.active=None
        for name,loss in [('pseudo',pseudo),('clip',clip)]:
            def forward(module,args,output,name=name):
                y=args[1]
                if not any(y is x for x in self.images):
                    self.images.append(y)
                    def capture(gradient):
                        self.records[self.active].append(gradient.detach().clone())
                    self.handles.append(y.register_hook(capture))
                def mark(gradient):self.active=name
                self.handles.append(output.register_hook(mark))
            self.handles.append(loss.register_forward_hook(forward))
    def close(self):
        for h in self.handles:h.remove()
        self.images.clear()


def cotangents(image,y,pseudo,clip):
    probe=y.detach().requires_grad_(True)
    lp,lc=pseudo(image,probe),clip(image,probe)
    cp=torch.autograd.grad(lp,probe,retain_graph=True)[0].detach()
    cc=torch.autograd.grad(lc,probe)[0].detach()
    return dict(pseudo_loss=lp.detach(),clip_loss=lc.detach(),cp=cp,cc=cc)


def isolation(source,clip,isp,hashes):
    return dict(source_hash_unchanged=state_hash(source)==hashes[0],clip_hash_unchanged=state_hash(clip)==hashes[1],
        all_eval=all(not m.training for model in (source,clip) for m in model.modules()),
        all_frozen_grad_none=all(not p.requires_grad and p.grad is None for model in (source,clip) for p in model.parameters()),
        isp_identity_grad_none=bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None))


def setup(config,project,output):
    torch.manual_seed(config['seed']);torch.set_num_threads(config['threads']);torch.backends.cudnn.benchmark=False
    pins=json.loads(Path('research_log/T022A2_pins.json').read_text());code={}
    for name,want in pins['protected_modules_LF'].items():
        raw=Path(name).read_bytes();code[name]=dict(raw_sha256=hashlib.sha256(raw).hexdigest(),LF_sha256=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest())
        assert code[name]['LF_sha256']==want,name
    cohort=Path('research_log/T022A_train_cohort.json');assert hashlib.sha256(cohort.read_bytes()).hexdigest()==pins['cohort_sha256']
    for name,want in pins['saved_receipts'].items():assert hashlib.sha256((project/name).read_bytes()).hexdigest()==want,name
    base=project/'research_log/remote_runs/20260914-001557-taisp-t022a1-runtime-smoke/artifacts/study'
    saved=json.loads((base/'samples.jsonl').read_text().splitlines()[0]);edges=json.loads((base/'spatial_edges.json').read_text())
    assert saved['image_id']==160585 and saved['case']=='gamma_s1'
    info=next(i for i in json.loads(cohort.read_text())['images'] if i['image_id']==160585)
    image=corrupt(load_image(info,config['device']),'gamma',1)
    selected={k:torch.tensor(v,device=image.device,dtype=torch.long if k=='labels' else image.dtype) for k,v in saved['support'].items()}
    source=load_detector(config['device']).eval().requires_grad_(False)
    clip=load_clip_guidance(config['device'],local_files_only=True).eval().requires_grad_(False)
    isp=DifferentiableISP().to(config['device']);loss=DetectorNativeLoss(source,{'base':selected},'det_pseudo').eval().requires_grad_(False)
    hashes=(state_hash(source),state_hash(clip));old=json.loads((base/'environment.json').read_text())
    assert hashes==(old['source_state_sha256'],old['clip_state_sha256'])
    mask,rects=support_mask(image,selected['boxes'])
    trajectory=next(r for r in edges['records'] if r['case']=='empty_mask')['diagnostics']
    states=[image.new_tensor(d['phi']) for d in trajectory]
    meta=environment_metadata(config);meta.update(code_hashes=code,model_state_hashes=hashes,saved_support=saved['support'],mask_rectangles=rects,
        mask_area=mask.double().mean().item(),mask_sha256=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
        cohort_sha256=pins['cohort_sha256'],image_id=160585,case='gamma_s1',saved_states=[s.tolist() for s in states],evaluations=0)
    write_json(output/'environment.json',meta)
    return image,selected,source,clip,isp,loss,hashes,mask,states


def audit(config,project,output,deterministic=False):
    output.mkdir(parents=True,exist_ok=True)
    image,selected,source,clip,isp,loss,hashes,mask,states=setup(config,project,output)
    if deterministic:
        torch.use_deterministic_algorithms(True,warn_only=False)
        result=dict(enabled=True,warn_only=False,evaluations=0)
        try:
            c=cotangents(image,isp(image,image.new_zeros(8)).detach(),loss,clip)
            save_tensors(output/'deterministic_cotangents.pt.gz',c);result['status']='completed'
        except RuntimeError as error:
            result.update(status='raised',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        result['isolation']=isolation(source,clip,isp,hashes)
        write_json(output/'deterministic.json',result);print(json.dumps(result),flush=True);return
    assert not torch.are_deterministic_algorithms_enabled()
    fixed=[];b=[]
    for step,state in enumerate(states):
        y=compose(image,isp,torch.zeros_like(mask),*state).detach();repeats=[]
        save_tensors(output/f'fixed_state{step}.pt.gz',dict(image=image,enhanced=y,states=state,mask=mask))
        for repeat in range(5):
            c=cotangents(image,y,loss,clip);save_tensors(output/f'cotangent_s{step}_r{repeat}.pt.gz',c);repeats.append(c)
        fixed.append(repeats[0]);b.append(dict(step=step,pairs=pairs(repeats)))
        write_json(output/'fixed_cotangent_pairs.json',b)
        print('fixed cotangents state '+str(step)+' complete',flush=True)
    iso=isolation(source,clip,isp,hashes);write_json(output/'fixed_cotangent_isolation.json',iso);assert all(iso.values())
    crows=[];tuple0=None
    for step,state in enumerate(states):
        for name,m in [('real',mask),('empty',torch.zeros_like(mask)),('full',torch.ones_like(mask))]:
            repeats=[]
            for repeat in range(5):
                g=regional_gradients(image,isp,m,state,{'pseudo':fixed[step]['cp'],'clip':fixed[step]['cc']})
                repeats.append(g);save_tensors(output/f'jvp_s{step}_{name}_r{repeat}.pt.gz',g)
                if tuple0 is None:tuple0=(*g['pseudo'].to(image),*g['clip'].to(image))
            row=dict(step=step,mask=name,pairs=pairs(repeats));crows.append(row)
            write_json(output/'jvp_pairs.json',crows)
            if not all(v['exact'] for pair in row['pairs'] for v in pair['fields'].values()):
                write_json(output/'completion.json',dict(status='spatial_fixed_input_blocker',stage='JVP',evaluations=0));return
    doses=[]
    for _ in range(20):
        delta,d=dose_step(*tuple0,lr=config['semantic_lr'],eps=config['radius_eps'])
        doses.append(dict(delta=delta,c=image.new_tensor(d['c']),multipliers=image.new_tensor(d['multipliers']),u=image.new_tensor(d['shared_direction']),v=image.new_tensor(d['base_delta'])))
    d_pairs=pairs(doses);write_json(output/'dose_pairs.json',d_pairs)
    save_tensors(output/'dose_repeats.pt.gz',{f'r{i}_{k}':v for i,r in enumerate(doses) for k,v in r.items()})
    if not all(v['exact'] for pair in d_pairs for v in pair['fields'].values()):
        write_json(output/'completion.json',dict(status='spatial_fixed_input_blocker',stage='dose',evaluations=0));return
    print('Fixed-input JVP and dose exact; starting25runtime repeats',flush=True)
    empty={k:v[:0] for k,v in selected.items()};runtime=[]
    for name,m in [('current',None),('real',mask),('empty',torch.zeros_like(mask)),('full',torch.ones_like(mask)),('empty_support',torch.zeros_like(mask))]:
        objective=DetectorNativeLoss(source,{'base':empty if name=='empty_support' else selected},'det_pseudo')
        repeats=[];diagnostics=[];frozen_mask=m.clone() if m is not None else None
        for repeat in range(5):
            recorder=CotangentRecorder(objective,clip)
            try:
                kwargs=dict(steps=config['semantic_steps'],lr=config['semantic_lr'],eps=config['radius_eps'])
                result=adapt_clip_radius(image,isp,objective,clip,**kwargs) if m is None else adapt_spatial_dose(image,isp,objective,clip,m,**kwargs)
            finally:recorder.close()
            assert len(recorder.records['pseudo'])==len(recorder.records['clip'])==4
            tensors=dict(final_state=result.phi,final_image=result.enhanced)
            for step,d in enumerate(result.diagnostics):
                tensors.update({f's{step}_cp':recorder.records['pseudo'][step],f's{step}_cc':recorder.records['clip'][step],
                    f's{step}_pseudo_gradient':image.new_tensor(d['detector_gradient']),
                    f's{step}_clip_gradient':image.new_tensor(d['clip_gradient'] if name=='current' else d['common_clip_gradient']),
                    f's{step}_state':image.new_tensor(d['phi'])})
            save_tensors(output/f'runtime_{name}_r{repeat}.pt.gz',tensors)
            checks=isolation(source,clip,isp,hashes);checks.update(reset_zero=bool(torch.count_nonzero(result.phi0)==0),mask_unchanged=m is None or torch.equal(m,frozen_mask))
            if name=='empty_support':checks['exact_zero_state']=bool(torch.count_nonzero(result.phi)==0)
            diagnostics.append(dict(repeat=repeat,steps=result.diagnostics,isolation=checks));repeats.append(tensors)
            write_json(output/f'runtime_{name}_diagnostics.json',diagnostics)
            assert all(checks.values()),checks
            print(f'runtime {name} repeat{repeat} complete',flush=True)
        row=dict(case=name,pairs=pairs(repeats));runtime.append(row);write_json(output/'runtime_pairs.json',runtime)
        if name=='empty_support' and not all(v['exact'] for pair in row['pairs'] for v in pair['fields'].values()):
            write_json(output/'completion.json',dict(status='empty_support_control_blocker',evaluations=0));return
    write_json(output/'completion.json',dict(status='completed_analysis_needs_review',fixed_states=4,cotangent_repeats=20,
        regional_calls=60,dose_calls=20,runtime_episodes=25,isolation=isolation(source,clip,isp,hashes),evaluations=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/t022a.yaml'))
    p.add_argument('--project',type=Path,default=Path('/home/liujianhua/wjq/TAISP'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deterministic',action='store_true');a=p.parse_args()
    audit(yaml.safe_load(a.config.read_text()),a.project,a.output,a.deterministic)
