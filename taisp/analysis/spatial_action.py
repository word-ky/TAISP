"""T014-A analysis-only object/background ISP Jacobian audit at identity."""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from taisp import DifferentiableISP
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from .deterministic_replay import state_hash
from .oracle import detector_task_loss
from .predictor_factorization import write_json
from .source_meta_smoke import load_episodes, frozen_unchanged

EPS = 1e-12
METRICS = ('task_regional_cosine','pseudo_regional_cosine','task_cancellation','R_task',
           'A_global','A_spatial','Delta_A','D_global','D_spatial','Delta_D')


def support_mask(image, boxes):
    height,width = image.shape[-2:]
    mask = image.new_zeros((1,1,height,width))
    rectangles = []
    for x1,y1,x2,y2 in boxes.detach().cpu().tolist():
        left,top = math.floor(max(0,min(width,x1))),math.floor(max(0,min(height,y1)))
        right,bottom = math.ceil(max(0,min(width,x2))),math.ceil(max(0,min(height,y2)))
        mask[:,:,top:bottom,left:right] = 1
        rectangles.append([left,top,right,bottom])
    return mask.detach(),rectangles


def compose(image, isp, mask, phi_obj, phi_bg):
    mask = mask.detach()
    return mask*isp(image,phi_obj)+(1-mask)*isp(image,phi_bg)


def tensor_check(value, reference, *, atol, rtol):
    difference = (value.detach().double()-reference.detach().double()).abs()
    bound = atol+rtol*reference.detach().double().abs()
    return {'passed':bool(torch.all(difference<=bound)), 'max_absolute_error':difference.max().item(),
            'relative_l2_error':(difference.norm()/(reference.detach().double().norm()+EPS)).item(),
            'max_error_over_bound':(difference/bound).max().item(), 'atol':atol,'rtol':rtol}


def identity_gradients(image, isp, mask, loss_fn):
    phi = image.new_zeros(8,requires_grad=True)
    global_image = isp(image,phi)
    loss,components = loss_fn(global_image)
    cotangent,direct = torch.autograd.grad(loss,(global_image,phi))
    obj,bg = image.new_zeros(8,requires_grad=True),image.new_zeros(8,requires_grad=True)
    regional_image = compose(image,isp,mask,obj,bg)
    go,gb = torch.autograd.grad(regional_image,(obj,bg),grad_outputs=cotangent.detach())
    assert all(torch.isfinite(v).all() for v in (loss,cotangent,direct,go,gb)), 'Non-finite identity gradient.'
    return {'loss':loss.item(),'components':{k:v.item() for k,v in components.items()},
            'global':direct.detach().cpu().tolist(),'object':go.detach().cpu().tolist(),'background':gb.detach().cpu().tolist(),
            'image_cotangent_norm':cotangent.norm().item(),
            'saturation_at_identity':((global_image<=1e-4)|(global_image>=1-1e-4)).float().mean().item(),
            'checks':{'identity_image':tensor_check(global_image,image,atol=2e-7,rtol=1e-6),
                      'regional_global_output':tensor_check(regional_image,global_image,atol=2e-7,rtol=1e-6),
                      'regional_gradient_sum':tensor_check(go+gb,direct,atol=1e-7,rtol=1e-5)}}


def cosine(x,y):
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator else 0.


def geometry(task_obj,task_bg,pseudo_obj,pseudo_bg):
    to,tb,po,pb = [np.asarray(v,dtype=np.float64) for v in (task_obj,task_bg,pseudo_obj,pseudo_bg)]
    ts,ps = to+tb,po+pb
    tc,pc = np.concatenate((to,tb)),np.concatenate((po,pb))
    nt,nb,np_o,np_b = [np.linalg.norm(v) for v in (to,tb,po,pb)]
    ag,asp = cosine(ps,ts),cosine(pc,tc)
    dg = float(ts@ps/(math.sqrt(2)*np.linalg.norm(ps)+EPS))
    ds = float(tc@pc/(np.linalg.norm(pc)+EPS))
    return {'task_regional_cosine':cosine(to,tb),'pseudo_regional_cosine':cosine(po,pb),
            'task_cancellation':float(1-np.linalg.norm(ts)/(nt+nb+EPS)),
            'R_task':float(math.sqrt(2)*np.linalg.norm(tc)/(np.linalg.norm(ts)+EPS)),
            'A_global':ag,'A_spatial':asp,'Delta_A':asp-ag,
            'D_global':dg,'D_spatial':ds,'Delta_D':ds-dg,
            'task_object_norm':float(nt),'task_background_norm':float(nb),
            'pseudo_object_norm':float(np_o),'pseudo_background_norm':float(np_b),
            'zero_task_regions':bool(nt==0 and nb==0),'zero_pseudo_regions':bool(np_o==0 and np_b==0)}


def describe(rows):
    if not rows:
        return {'n':0,'metrics':{}}
    return {'n':len(rows),'empty_support_count':sum(r['support_count']==0 for r in rows),
            'mask_area_mean':float(np.mean([r['mask_area_fraction'] for r in rows])),
            'metrics':{key:{'mean':float(np.mean(v)),'median':float(np.median(v)),
                            'min':float(np.min(v)),'max':float(np.max(v)),
                            'positive_count':sum(x>0 for x in v)}
                       for key in METRICS for v in [[r['geometry'][key] for r in rows]]}}


def summarize(rows):
    groups = {'overall':rows,'clean':[r for r in rows if r['case']=='clean_s0'],
              'corrupted':[r for r in rows if r['case']!='clean_s0']}
    groups.update({f'block{i}':[r for r in rows if r['block']==i] for i in range(4)})
    groups.update({'case/'+case:[r for r in rows if r['case']==case] for case in sorted({r['case'] for r in rows})})
    groups['empty_support'] = [r for r in rows if r['support_count']==0]
    groups['nonempty_support'] = [r for r in rows if r['support_count']>0]
    strata = {'mask_zero':lambda a:a==0,'mask_small':lambda a:0<a<=.05,
              'mask_middle':lambda a:.05<a<.95,'mask_large':lambda a:.95<=a<1,'mask_full':lambda a:a==1}
    groups.update({name:[r for r in rows if predicate(r['mask_area_fraction'])] for name,predicate in strata.items()})
    result = {name:describe(selected) for name,selected in groups.items()}
    return {'scopes':result,'gate':gate(result)}


def gate(scopes):
    all_m,corr = scopes['overall']['metrics'],scopes['corrupted']['metrics']
    blocks = sum(scopes[f'block{i}']['metrics']['Delta_D']['median']>0 for i in range(4))
    flags = {'overall_median_R_at_least1_20':all_m['R_task']['median']>=1.20,
             'corrupted_median_R_at_least1_15':corr['R_task']['median']>=1.15,
             'overall_positive_Delta_D_at_least20':all_m['Delta_D']['positive_count']>=20,
             'corrupted_positive_Delta_D_at_least10':corr['Delta_D']['positive_count']>=10,
             'overall_median_Delta_D_positive':all_m['Delta_D']['median']>0,
             'at_least3_positive_block_medians':blocks>=3}
    return {'flags':flags,'positive_block_medians':blocks,'passed':all(flags.values()),
            'decision':'spatial_partition_worthy_of_later_fixed_prototype_review' if all(flags.values()) else 'do_not_implement_spatial_ISP_from_this_partition',
            'stop_for_research_review':True}


def run(manifest_path, prior_root, output):
    pins = {'supports.json':'bf8f688741fcb4956c7ed40f52cc5f0a63d3030cad3ada968f1457a38c2cf747',
            'environment.json':'d0635da6c52a1f41a86181ace053c6c6fe1f38f342530b6fd9d806e8ee98cc42',
            'cohort.json':'99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357'}
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='a4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e'
    for name,pin in pins.items():
        assert hashlib.sha256((prior_root/name).read_bytes()).hexdigest()==pin,name
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    source = load_detector(device)
    isp = DifferentiableISP().to(device)
    source_hash = state_hash(source)
    assert source_hash=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    weights = Path(torch.hub.get_dir())/'checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert weight_hash=='258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    states = [{k:v.clone() for k,v in source.state_dict().items()}]
    assert frozen_unchanged((source,),states)
    write_json(output/'environment.json',{'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
               'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
               'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'source_state_sha256':source_hash,
               'source_weight_sha256':weight_hash,'parent_sha256':pins,'seed':20260913,'eps':EPS,
               'deterministic_algorithms':False,'cudnn_benchmark':False,'cublas_workspace_config':None,
               'gradient_method':'One source loss/image cotangent per objective; direct global gradient and independent regional ISP VJP at identical output.',
               'clip_calls':0,'optimizer_steps':0,'initial_isolation_passed':True})
    shutil.copyfile(manifest_path,output/'cohort.json')
    manifest = json.loads(manifest_path.read_text())
    all_supports = json.loads((prior_root/'supports.json').read_text())
    selected_supports = [all_supports[j] for i in manifest['parent_image_indices'] for j in (2*i,2*i+1)]
    write_json(output/'supports.json',selected_supports)
    episodes = load_episodes(manifest,device)
    assert len(episodes)==len(selected_supports)==32
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index,(episode,saved) in enumerate(zip(episodes,selected_supports,strict=True)):
        assert (episode['image_id'],episode['case'])==(saved['image_id'],saved['case'])
        selected = {k:torch.tensor(saved[k],device=device,dtype=torch.long if k=='labels' else torch.float32)
                    for k in ('boxes','scores','labels')}
        assert len(selected['scores'])<=20 and bool(torch.all(selected['scores']>=.5))
        mask,rectangles = support_mask(episode['image'],selected['boxes'])
        native = DetectorNativeLoss(source,{'base':selected},'det_pseudo')
        task = identity_gradients(episode['image'],isp,mask,
                                  lambda y:detector_task_loss(source,y,episode['targets'],seed=20260913))
        pseudo = identity_gradients(episode['image'],isp,mask,lambda y:(native(episode['image'],y),{}))
        current_hash = state_hash(source)
        isolation = {'source_unchanged_frozen_eval_grad_none':frozen_unchanged((source,),states),
                     'source_hash_unchanged':current_hash==source_hash,
                     'isp_unchanged_grad_none':bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None)}
        row = {'episode_index':index,'parent_episode_index':saved['episode_index'],'image_id':episode['image_id'],
               'case':episode['case'],'block':index//8,'support_count':len(selected['boxes']),
               'mask_shape':list(mask.shape),'mask_rectangles':rectangles,'mask_area_fraction':mask.double().mean().item(),
               'mask_uint8_sha256':hashlib.sha256(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
               'source_state_sha256':current_hash,'isolation':isolation,'task':task,'pseudo':pseudo}
        write_json(output/f'record_{index:02d}.json',row)
        assert all(isolation.values()) and all(c['passed'] for objective in (task,pseudo) for c in objective['checks'].values()), 'Identity/reconstruction/isolation blocker; retain record, no tolerance change.'
        rows.append(row)
        print(json.dumps({'episode':index+1,'of':32,'reconstruction_passed':True,'isolation_passed':True}),flush=True)
    # Scientific comparisons start only after the complete primary record/check stage.
    for row in rows:
        row['geometry'] = geometry(row['task']['object'],row['task']['background'],row['pseudo']['object'],row['pseudo']['background'])
    write_json(output/'records.json',{'records':rows})
    result = summarize(rows)
    write_json(output/'summary.json',result)
    write_json(output/'completion.json',{'status':'completed','source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
               'episodes':32,'source_loss_calls':64,'clip_calls':0,'optimizer_steps':0,'gate':result['gate'],
               'all_reconstructions_and_isolation_passed':True,'elapsed_seconds':time.perf_counter()-started,
               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps({'gate':result['gate'],'overall':result['scopes']['overall']}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','prior-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    run(args.manifest,args.prior_root,args.output)
