"""Prospective R037 paired output confirmation, no AP or method changes."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
import yaml
from taisp import DifferentiableISP
from taisp.isp.spatial import support_mask
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions
from taisp.tta.trust_radius import adapt_clip_radius
from taisp.tta.spatial_dose import adapt_spatial_dose
from .deterministic_replay import state_hash
from .differential_subspace import write_json
from .run_t018a import load_image,corrupt
from .run_t002 import environment_metadata
from .spatial_repeatability import pairs,save_tensors,isolation


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def mask_hash(mask):return hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()


def decision(tuples,valid):
    cur=[r['d_cur'] for r in tuples];sp=[r['d_sp'] for r in tuples]
    flags=dict(all80episodes_valid=valid and len(tuples)==8,
        at_least7of8=sum(b<=2*a+1e-6 for a,b in zip(cur,sp))>=7,
        median_within125=statistics.median(sp)<=1.25*statistics.median(cur)+1e-6,
        no_tuple_above5=all(b<=5*a+1e-6 for a,b in zip(cur,sp)))
    return dict(flags=flags,passed=all(flags.values()),passing_tuples=sum(b<=2*a+1e-6 for a,b in zip(cur,sp)),
        median_current=statistics.median(cur),median_spatial=statistics.median(sp))


def setup(config,output):
    output.mkdir(parents=True,exist_ok=True);torch.manual_seed(config['seed']);torch.set_num_threads(config['threads']);torch.backends.cudnn.benchmark=False
    assert not torch.are_deterministic_algorithms_enabled()
    plan=json.loads(Path('research_log/T022A3_plan.json').read_text());checks={}
    for name,want in plan['protected_modules_LF'].items():
        b=Path(name).read_bytes();checks[name]=dict(raw=hashlib.sha256(b).hexdigest(),LF=hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest())
        assert checks[name]['LF']==want,name
    cp=Path('research_log/T022A_train_cohort.json');assert hashlib.sha256(cp.read_bytes()).hexdigest()==plan['cohort_sha256']
    cohort=json.loads(cp.read_text());assert cohort['image_ids'][:4]==plan['image_ids']
    source=load_detector(config['device']).eval().requires_grad_(False)
    assert state_hash(source)=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    write_json(output/'code_hashes.json',checks)
    return plan,cohort,source


def condition_image(info,case,device):
    x=load_image(info,device);return x if case=='clean_s0' else corrupt(x,'contrast',2)


def prepare(config,output):
    plan,cohort,source=setup(config,output);before=state_hash(source);rows=[]
    for info in cohort['images'][:4]:
        for case in plan['conditions']:
            x=condition_image(info,case,config['device'])
            with torch.no_grad():s=select_predictions(source(x)[0],config['support_threshold'],config['support_topk'])
            values={k:v.cpu().tolist() for k,v in s.items()};mask,rects=support_mask(x,s['boxes'])
            rows.append(dict(image_id=info['image_id'],case=case,support=values,support_sha256=json_hash(values),mask_sha256=mask_hash(mask),mask_area=mask.double().mean().item(),mask_rectangles=rects))
    assert state_hash(source)==before
    write_json(output/'supports.json',dict(plan_sha256=hashlib.sha256(Path('research_log/T022A3_plan.json').read_bytes()).hexdigest(),rows=rows,teacher_forwards=8,source_hash=before,evaluations=0))
    write_json(output/'environment.json',environment_metadata(config));print('Eight supports prepared, zero adaptation/AP; commit before repeats.',flush=True)


def run(config,supports_path,output):
    plan,cohort,source=setup(config,output);saved=json.loads(supports_path.read_text())
    assert saved['plan_sha256']==hashlib.sha256(Path('research_log/T022A3_plan.json').read_bytes()).hexdigest()
    assert [(r['image_id'],r['case']) for r in saved['rows']]==[(i,c) for i in plan['image_ids'] for c in plan['conditions']]
    clip=load_clip_guidance(config['device'],local_files_only=True).eval().requires_grad_(False)
    isp=DifferentiableISP().to(config['device']);hashes=(state_hash(source),state_hash(clip))
    assert hashes[1]=='6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c'
    meta=environment_metadata(config);meta.update(model_hashes=hashes,supports_file_sha256=hashlib.sha256(supports_path.read_bytes()).hexdigest(),plan=plan,evaluations=0)
    write_json(output/'environment.json',meta);summaries=[];all_valid=True;episode_count=0
    for index,row in enumerate(saved['rows']):
        info=next(i for i in cohort['images'] if i['image_id']==row['image_id']);x=condition_image(info,row['case'],config['device'])
        support={k:torch.tensor(v,device=x.device,dtype=torch.long if k=='labels' else x.dtype) for k,v in row['support'].items()}
        if not row['support']['boxes']:support['boxes']=support['boxes'].reshape(0,4)
        assert json_hash(row['support'])==row['support_sha256'];mask,_=support_mask(x,support['boxes']);assert mask_hash(mask)==row['mask_sha256']
        method_rows={};dispersions={}
        for method in plan['methods']:
            loss=DetectorNativeLoss(source,{'base':support},'det_pseudo');repeats=[];receipts=[]
            for repeat in range(5):
                torch.cuda.synchronize();start=time.perf_counter()
                kwargs=dict(steps=config['semantic_steps'],lr=config['semantic_lr'],eps=config['radius_eps'])
                result=adapt_clip_radius(x,isp,loss,clip,**kwargs) if method=='current_ours' else adapt_spatial_dose(x,isp,loss,clip,mask,**kwargs)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                tensors=dict(final_image=result.enhanced,final_state=result.phi)
                for step,d in enumerate(result.diagnostics):
                    tensors[f's{step}_state']=x.new_tensor(d['phi'])
                    if step<3:tensors[f's{step}_update']=x.new_tensor(d['state_delta']) if method=='spatial_dose_ours' else -config['semantic_lr']*x.new_tensor(d['gradient_per_coordinate'])
                save_tensors(output/f't{index}_{method}_r{repeat}.pt.gz',tensors)
                checks=isolation(source,clip,isp,hashes);checks.update(reset_zero=bool(torch.count_nonzero(result.phi0)==0),
                    finite=all(bool(torch.isfinite(v).all()) for v in tensors.values()) and all(torch.isfinite(torch.tensor([d['total'],d['clip_loss']])).all().item() for d in result.diagnostics),
                    support_unchanged=json_hash({k:v.cpu().tolist() for k,v in support.items()})==row['support_sha256'],mask_unchanged=mask_hash(mask)==row['mask_sha256'])
                if not len(support['boxes']):checks['empty_support_zero']=bool(torch.count_nonzero(result.phi)==0)
                if method=='spatial_dose_ours':checks['dose_valid']=all(-1-1e-6<=d['c']<=1+1e-6 and all(.5-1e-6<=v<=1.5+1e-6 for v in d['multipliers']) and abs(sum(d['multipliers'])/2-1)<=1e-6 for d in result.diagnostics)
                all_valid=all_valid and all(checks.values());episode_count+=1
                receipts.append(dict(repeat=repeat,seconds=elapsed,updated=bool(torch.count_nonzero(result.phi)),support_sha256=row['support_sha256'],mask_sha256=row['mask_sha256'],mask_area=row['mask_area'],isolation=checks,diagnostics=result.diagnostics))
                write_json(output/f't{index}_{method}_receipts.json',receipts);repeats.append(tensors)
            comparisons=pairs(repeats);write_json(output/f't{index}_{method}_pairs.json',comparisons)
            dispersions[method]=max(p['fields']['final_image']['relative_l2'] for p in comparisons)
            method_rows[method]=dict(seconds=[r['seconds'] for r in receipts],zero_updates=sum(not r['updated'] for r in receipts))
        dc,ds=(dispersions[m] for m in plan['methods']);summaries.append(dict(image_id=row['image_id'],case=row['case'],d_cur=dc,d_sp=ds,ratio=ds/dc if dc else None,within2=ds<=2*dc+1e-6,above5=ds>5*dc+1e-6,methods=method_rows))
        write_json(output/'tuple_summary.json',summaries);print(f'confirmation tuple{index+1}/8 current={dc} spatial={ds}',flush=True)
    gate=decision(summaries,all_valid and episode_count==80);write_json(output/'completion.json',dict(status='passed' if gate['passed'] else 'blocked',episodes=episode_count,gate=gate,evaluations=0))
    print(json.dumps(gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/t022a.yaml'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--supports',type=Path)
    a=p.parse_args();config=yaml.safe_load(a.config.read_text())
    if a.prepare:prepare(config,a.output)
    else:run(config,a.supports,a.output)
