"""Fixed paired spatial CLIP study; oracle directions and losses stay here."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.conditioned_clip import load_conditioner, NEGATIVE_BANKS
from taisp.losses.semantic import SemanticDirectionLoss
from taisp.losses.spatial_clip import PatchDirectionLoss
from taisp.models.detector import load_detector
from taisp.models.region_weights import original_region_weights
from taisp.tta import AdaptConfig, adapt
from .coco import COCOSubset, prediction_records, subset_ap
from .corruptions import CASES, corrupt
from .oracle import detector_task_loss
from .run_t002 import environment_metadata, physical_vector
from .run_t003 import FAMILY_INDEX
from .spatial_diagnostics import norm_match, partition_gradients, partition_losses


def make_guidance(variant, generic, banks, family, region):
    kind, text = variant.split('_')
    direction = generic.direction if text == 'generic' else banks.positive-banks.negatives[FAMILY_INDEX[family]]
    if kind == 'global':
        return generic if text == 'generic' else SemanticDirectionLoss(generic.encoder,direction)
    return PatchDirectionLoss(generic.encoder,direction,region['weights'] if kind == 'region' else None)


def run(config,data_root,baseline,output,limit=None):
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark=False
    device=torch.device(config['device'])
    data=COCOSubset(data_root)
    ids=data.ids[:limit] if limit is not None else data.ids[:config['count']]
    generic=load_clip_guidance(device,local_files_only=True)
    banks=load_conditioner(generic,local_files_only=True)  # Text-bank reuse only; prepare() never called.
    detector,isp=load_detector(device),DifferentiableISP().to(device)
    cfg=AdaptConfig(steps=config['semantic_steps'],lr=config['semantic_lr'],
                    consistency_weight=0,regularization_weight=0)
    meta=environment_metadata(config)
    meta.update(evaluated_image_ids=ids,smoke_limit=limit,annotation_sha256=data.manifest['annotation_sha256'],
                negative_concept_banks=NEGATIVE_BANKS,feature='last_hidden_state[:,1:] -> frozen post_layernorm -> visual_projection -> token L2 normalization;49x512',
                region='original frozen inference;score>=0.5,top20;score-weighted box/patch overlap fraction;normalized;uniform if empty',
                baseline_study=str(baseline),norm_match_reference='same-image/case global_generic gradient at phi0',
                directions={'generic':generic.direction.tolist(), 'oracle':{f:torch.nn.functional.normalize(banks.positive-banks.negatives[i],dim=-1).tolist() for f,i in FAMILY_INDEX.items()}},
                gradient_definition='all eight coordinates; no masks; raw semantic gradient is actual update',
                timing='adapt_only excludes analysis; deploy_seconds adds original detector inference/map only for region variants',
                partition='fixed predicted object-support vs complement; background region loss is zero by design except uniform fallback')
    (output/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
    shutil.copyfile(data.root/'subset.json',output/'subset.json')
    predictions={f'{f}_s{s}_{v}{k}':[] for f,s in CASES for v in config['variants'] for k in (1,3)}
    rows=[]
    initial_physical=physical_vector(isp,torch.zeros(8,device=device))
    started=time.time()
    with (output/'samples.jsonl').open('w',encoding='utf-8') as samples:
        for index,image_id in enumerate(ids):
            clean,targets=data.load(image_id,device)
            for family,severity in CASES:
                x=corrupt(clean,family,severity)
                seed=config['seed']+image_id
                torch.cuda.synchronize()
                start=time.perf_counter()
                region=original_region_weights(detector,x,threshold=config['region_threshold'],topk=config['region_topk'])
                torch.cuda.synchronize()
                region_seconds=time.perf_counter()-start
                region_record={k:v.tolist() if torch.is_tensor(v) else v for k,v in region.items()}
                p0=torch.zeros(8,device=device,requires_grad=True)
                before,_=detector_task_loss(detector,isp(x,p0),targets,seed=seed)
                g_det=torch.autograd.grad(before,p0)[0].detach()
                before_value=before.item()
                del before
                global_gradient=None
                for variant in config['variants']:
                    kind=variant.split('_')[0]
                    guidance=make_guidance(variant,generic,banks,family,region)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    result=adapt(x,isp,guidance,config=cfg)
                    torch.cuda.synchronize()
                    adapt_seconds=time.perf_counter()-start
                    peak=torch.cuda.max_memory_allocated()/1024**2
                    g=x.new_tensor(result.diagnostics[0]['gradient_per_coordinate'])
                    if variant=='global_generic':
                        global_gradient=g
                    matched=norm_match(g,global_gradient)
                    phi1=x.new_tensor(result.diagnostics[1]['phi'])
                    y1=isp(x,phi1).detach()
                    pm=-cfg.lr*matched
                    ym=isp(x,pm).detach()
                    partition={}
                    if kind!='global':
                        go,gb=partition_gradients(x,isp,guidance,region['support'])
                        partition={'g_object':go.tolist(),'g_background':gb.tolist(),
                            'sum_gradient_residual_norm':(go+gb-g).norm().item(),
                            'step1':partition_losses(x,y1,guidance,region['support']),
                            'step3':partition_losses(x,result.enhanced,guidance,region['support'])}
                    with torch.no_grad():
                        loss1,_=detector_task_loss(detector,y1,targets,seed=seed)
                        loss3,_=detector_task_loss(detector,result.enhanced,targets,seed=seed)
                        lossm,_=detector_task_loss(detector,ym,targets,seed=seed)
                        common1=generic(x,y1).item()
                        common3=generic(x,result.enhanced).item()
                        matched_semantic=guidance(x,ym).item()
                        for k,y in ((1,y1),(3,result.enhanced)):
                            predictions[f'{family}_s{severity}_{variant}{k}'].extend(prediction_records(image_id,detector(y)[0]))
                    norm=(g.norm()*g_det.norm()).item()
                    weights=guidance.weights if kind!='global' else None
                    row={'image_id':image_id,'family':family,'severity':severity,'variant':variant,
                        'g_sem':g.tolist(),'g_sem_raw':g.tolist(),'g_det':g_det.tolist(),
                        'g_sem_norm':g.norm().item(),'g_det_norm':g_det.norm().item(),
                        'gradient_cosine':(torch.dot(g,g_det)/norm).item() if norm>0 else None,
                        'coordinate_gate':[1.]*8,'det_loss_before':before_value,
                        'det_loss_delta_sem1':loss1.item()-before_value,'det_loss_delta_sem3':loss3.item()-before_value,
                        'semantic_loss_before':result.diagnostics[0]['semantic'],
                        'semantic_loss_1':result.diagnostics[1]['semantic'],'semantic_loss_3':result.diagnostics[-1]['semantic'],
                        'generic_semantic_loss_1':common1,'generic_semantic_loss_3':common3,
                        'saturation_before':result.diagnostics[0]['saturation_rate'],
                        'saturation_1':result.diagnostics[1]['saturation_rate'],'saturation_3':result.diagnostics[-1]['saturation_rate'],
                        'physical_change_1':(physical_vector(isp,phi1)-initial_physical).tolist(),
                        'physical_change_3':(physical_vector(isp,result.phi)-initial_physical).tolist(),
                        'adapt_seconds_3':adapt_seconds,'region_setup_seconds':region_seconds,
                        'deploy_seconds_3':adapt_seconds+(region_seconds if kind=='region' else 0),
                        'peak_allocated_mb':peak,'diagnostics':result.diagnostics,
                        'norm_matched':{'gradient':matched.tolist(),'gradient_norm':matched.norm().item(),
                            'target_norm':global_gradient.norm().item(),'zero_gradient':g.norm().item()==0,
                            'phi':pm.tolist(),'det_loss_delta':lossm.item()-before_value,
                            'linear_prediction':(-cfg.lr*torch.dot(matched,g_det)).item(),
                            'semantic_loss':matched_semantic,
                            'saturation':((ym<=1e-4)|(ym>=1-1e-4)).float().mean().item()},
                        'region':region_record,'partition':partition,
                        'patch_weights':weights.tolist() if weights is not None else None,
                        'effective_patch_count':(1/weights.square().sum()).item() if weights is not None else None,
                        'weighted_patch_fraction':(weights>0).float().mean().item() if weights is not None else None}
                    samples.write(json.dumps(row,allow_nan=False)+'\n')
                    samples.flush()
                    rows.append(row)
            print(f'completed {index+1}/{len(ids)} image_id={image_id} elapsed={time.time()-started:.1f}s',flush=True)
    base_metrics=json.loads((baseline/'metrics.json').read_text())
    if limit is not None:
        base_metrics={n:subset_ap(data.coco,ids,[p for p in json.loads((baseline/'predictions'/f'{n}.json').read_text()) if p['image_id'] in ids]) for n in base_metrics}
    (output/'baseline_metrics.json').write_text(json.dumps(base_metrics,indent=2)+'\n')
    pd=output/'predictions'
    pd.mkdir(exist_ok=True)
    metrics={}
    for name,records in predictions.items():
        (pd/f'{name}.json').write_text(json.dumps(records)+'\n')
        print(f'Evaluating {name}',flush=True)
        metrics[name]=subset_ap(data.coco,ids,records)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    summary={v:{f'{f}_s{s}':{'count':sum(r['variant']==v and r['family']==f and r['severity']==s for r in rows)} for f,s in CASES} for v in config['variants']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'images':len(ids),'samples':len(rows),'elapsed_seconds':time.time()-started})+'\n')
    print(f'Finished {len(ids)} images / {len(rows)} variant observations',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/t004.yaml'))
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    run(yaml.safe_load(a.config.read_text()),a.data_root,a.baseline,a.output,a.limit)


if __name__=='__main__':
    main()
