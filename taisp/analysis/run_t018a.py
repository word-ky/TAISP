"""R029 source100 comparison; annotations enter official evaluation only."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions
from taisp.tta.trust_radius import adapt_clip_radius
from .coco import prediction_records
from .corruptions import CASES, corrupt
from .deterministic_replay import state_hash
from .differential_subspace import write_json
from .native_pseudo_target import NativePseudoTargetLoss
from .replication import replication_ap
from .run_t002 import environment_metadata
from .source_meta_smoke import frozen_unchanged

METHODS = ('no_adapt', 'current_ours', 'nativePT_ours')
CASES_ALL = list(CASES)+[('clean',0)]
CASE_NAMES = [f'{f}_s{s}' for f,s in CASES_ALL]


def load_image(info, device):
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    path=Path(info['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    with Image.open(path) as im:
        return pil_to_tensor(im.convert('RGB')).float().div(255).unsqueeze(0).to(device)


def advancement(metrics):
    corrupt_cases=CASE_NAMES[:-1]
    groups={}
    for group,values in metrics.items():
        macro={m:100*statistics.mean(values[f'{c}_{m}']['AP'] for c in corrupt_cases) for m in METHODS}
        cases={c:100*(values[f'{c}_nativePT_ours']['AP']-values[f'{c}_current_ours']['AP']) for c in corrupt_cases}
        groups[group]={'macro_AP':macro,'candidate_minus_current':macro['nativePT_ours']-macro['current_ours'],
                       'candidate_minus_no_adapt':macro['nativePT_ours']-macro['no_adapt'],
                       'condition_deltas':cases,'positive_conditions':sum(v>0 for v in cases.values()),
                       'clean_AP':{m:100*values[f'clean_s0_{m}']['AP'] for m in METHODS},
                       'clean_delta':100*(values['clean_s0_nativePT_ours']['AP']-values['clean_s0_current_ours']['AP'])}
    a=groups['aggregate']
    blocks=sum(groups[f'block{i}']['candidate_minus_current']>0 for i in range(4))
    flags={'corruption_macro_above_current':a['candidate_minus_current']>0,
           'corruption_macro_delta_at_least_point10':a['candidate_minus_current']>=.10,
           'at_least3_positive_block_macros':blocks>=3,
           'at_least4_positive_corruption_conditions':a['positive_conditions']>=4,
           'corruption_macro_above_no_adapt':a['candidate_minus_no_adapt']>0,
           'clean_delta_at_least_minus_point10':a['clean_delta']>=-.10}
    return {'groups':groups,'gate':{'flags':flags,'positive_blocks':blocks,'passed':all(flags.values()),
            'decision':'promising_source_upgrade_pending_confirmatory_review' if all(flags.values()) else
                       'insufficient_or_worse_fixed_native_pseudo_target_formulation',
            'stop_for_research_review':True}}


def diagnostic_summary(rows, methods=METHODS):
    result={}
    for method in methods[1:]:
        for group in ('clean','corrupted'):
            chosen=[r for r in rows if r['method']==method and (r['case']=='clean_s0')==(group=='clean')]
            if not chosen:
                continue
            entry={'n':len(chosen),'update_fraction':statistics.mean(r['updated'] for r in chosen),
                   'empty_support_fraction':statistics.mean(r['support_count']==0 for r in chosen)}
            for key in ('phi3_norm','adapt_seconds','deploy_seconds','support_count','prediction_count','peak_allocated_bytes'):
                v=[r[key] for r in chosen]
                entry[key]={'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
            result[method+'/'+group]=entry
    return result


def confirmation(metrics, isolation_passed):
    """R030 source500 criteria; AP50/AP75 remain diagnostics, never gates."""
    groups=advancement(metrics)['groups']
    a=groups['aggregate']
    positive_blocks=sum(groups[f'block{i}']['candidate_minus_current']>0 for i in range(5))
    flags={'macro_AP_delta_at_least_point10':a['candidate_minus_current']>=.10,
           'at_least4_positive_block_macros':positive_blocks>=4,
           'at_least4_positive_corruption_conditions':a['positive_conditions']>=4,
           'corruption_macro_above_no_adapt':a['candidate_minus_no_adapt']>0,
           'clean_delta_at_least_minus_point10':a['clean_delta']>=-.10,
           'no_isolation_or_reproducibility_blocker':isolation_passed}
    for group,values in metrics.items():
        macro={k:{m:100*statistics.mean(values[f'{c}_{m}'][k] for c in CASE_NAMES[:-1]) for m in METHODS}
               for k in ('AP50','AP75')}
        groups[group]['additional_macro_metrics']=macro
        groups[group]['additional_macro_deltas']={k:v['nativePT_ours']-v['current_ours'] for k,v in macro.items()}
    return {'groups':groups,'gate':{'flags':flags,'positive_blocks':positive_blocks,'passed':all(flags.values()),
            'decision':'source_confirmed_pending_cross_detector_review' if all(flags.values()) else 'not_source_confirmed',
            'stop_for_research_review':True},
            'AP75_nonnegative_diagnostic':groups['aggregate']['additional_macro_deltas']['AP75']>=0}


def run(config, manifest_path, output, smoke=False):
    manifest=json.loads(manifest_path.read_text())
    images=manifest['images'][:2] if smoke else manifest['images']
    study=config.get('study','T018-A')
    methods=('no_adapt',*config['variants'])
    assert len(manifest['images'])==config['count'] and manifest['overlap_prior_source']==manifest['overlap_val']==0
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark=False
    source=load_detector(config['device'])
    clip=load_clip_guidance(config['device'],local_files_only=True)
    isp=DifferentiableISP().to(config['device'])
    source_hash,clip_hash=state_hash(source),state_hash(clip)
    source_states=[{k:v.clone() for k,v in source.state_dict().items()}]
    assert source_hash=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    meta=environment_metadata(config)
    old=json.loads(Path('/home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000/artifacts/study/environment.json').read_text())
    for key in ('clip_model','clip_revision','clip_sha256','detector_sha256','positive_prompts','negative_prompts'):
        value=list(meta[key]) if key.endswith('prompts') else meta[key]
        assert value==old[key],key
    for key in ('seed','device','threads','semantic_lr','semantic_steps','support_threshold','support_topk','radius_eps'):
        assert config[key]==old['config'][key],key
    interpretation=('T018-B independent train2017 source confirmation of unchanged T018-A candidate; no cross-detector claim'
                    if study=='T018-B' else 'T018-A developmental train2017 source-only fixed candidate; no confirmatory or cross-detector claim')
    if study=='T019-A':
        interpretation='T019-A predeclared native component development study; no confirmation or cross-detector claim'
    meta.update(interpretation=interpretation,
                smoke=smoke,cases=CASE_NAMES,methods=methods,source_state_sha256=source_hash,clip_state_sha256=clip_hash,
                cohort_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                evaluated_image_ids=[r['image_id'] for r in images],teacher='one original condition prediction,detachedscore>=.5top20',
                ground_truth_boundary='COCO annotations loaded only after all adaptation/predictions, for official AP',
                timing='synchronized steps0..3 diagnostics with K3updates; deploy adds sharedteacher setup; evaluation excluded')
    write_json(output/'environment.json',meta)
    write_json(output/'cohort.json',manifest)
    predictions={f'{c}_{m}':[] for c in CASE_NAMES for m in methods}
    rows=[]
    started=time.perf_counter()
    with (output/'samples.jsonl').open('w',encoding='utf-8') as handle:
        for i,info in enumerate(images):
            clean=load_image(info,config['device'])
            for family,severity in CASES_ALL:
                case=f'{family}_s{severity}'
                image=clean if family=='clean' else corrupt(clean,family,severity)
                torch.cuda.synchronize()
                begin=time.perf_counter()
                with torch.no_grad():
                    original=source(image)[0]
                    selected=select_predictions(original,config['support_threshold'],config['support_topk'])
                torch.cuda.synchronize()
                setup=time.perf_counter()-begin
                predictions[f'{case}_no_adapt'].extend(prediction_records(info['image_id'],original))
                frozen_support={k:v.clone() for k,v in selected.items()}
                counts={}
                for method in methods[1:]:
                    loss=DetectorNativeLoss(source,{'base':selected},'det_pseudo') if method=='current_ours' else NativePseudoTargetLoss(
                        source,selected,seed=config['seed'],component_set='full' if method=='nativePT_ours' else method)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin=time.perf_counter()
                    result=adapt_clip_radius(image,isp,loss,clip,steps=config['semantic_steps'],lr=config['semantic_lr'],eps=config['radius_eps'])
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-begin
                    peak=torch.cuda.max_memory_allocated()
                    isolation={'source_unchanged_frozen_eval_grad_none':frozen_unchanged((source,),source_states),
                               'isp_identity_grad_none':bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None),
                               'teacher_support_unchanged':all(torch.equal(v,frozen_support[k]) for k,v in selected.items()),
                               'finite_phi_and_gradients':bool(torch.isfinite(result.phi).all()) and
                                   all(torch.isfinite(torch.tensor(d['detector_gradient'])).all().item() for d in result.diagnostics)}
                    assert all(isolation.values()),isolation
                    if not len(selected['boxes']):
                        assert torch.equal(result.phi,torch.zeros_like(result.phi))
                        assert torch.equal(result.enhanced,isp(image,torch.zeros_like(result.phi)))
                    with torch.no_grad():
                        prediction=source(result.enhanced)[0]
                    predictions[f'{case}_{method}'].extend(prediction_records(info['image_id'],prediction))
                    counts[method]=len(prediction['boxes'])
                    row={'image_id':info['image_id'],'block':info['block'],'case':case,'method':method,
                         'diagnostics':result.diagnostics,'native_components':loss.loss_history if isinstance(loss,NativePseudoTargetLoss) else None,
                         'support':{k:v.detach().cpu().tolist() for k,v in selected.items()},'support_count':len(selected['boxes']),
                         'phi3_norm':result.phi.norm().item(),'updated':bool(torch.count_nonzero(result.phi)),
                         'adapt_seconds':elapsed,'teacher_seconds':setup,'deploy_seconds':elapsed+setup,
                         'peak_allocated_bytes':peak,'prediction_count':counts[method],
                         'original_prediction_count':len(original['boxes']),'isolation':isolation}
                    if isinstance(loss,NativePseudoTargetLoss):
                        row['active_component_keys']=list(loss.active_keys)
                        row['paired_prediction_count_delta']=counts[method]-counts['current_ours']
                    handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();rows.append(row)
            print(f'completed {i+1}/{len(images)} image_id={info["image_id"]} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    pd=output/'predictions';pd.mkdir()
    for name,records in predictions.items():
        write_json(pd/(name+'.json'),records)
    final_isolation={'source_hash_unchanged':state_hash(source)==source_hash,'clip_hash_unchanged':state_hash(clip)==clip_hash,
                     'source_frozen_eval_grad_none':frozen_unchanged((source,),source_states),
                     'clip_frozen_eval_grad_none':all(not p.requires_grad and p.grad is None for p in clip.parameters()) and
                                                  all(not m.training for m in clip.modules())}
    write_json(output/'isolation.json',final_isolation)
    assert all(final_isolation.values())
    write_json(output/'diagnostics.json',diagnostic_summary(rows,methods))
    metrics={}
    summary=None
    if not smoke:
        from pycocotools.coco import COCO
        annotation=Path(manifest['annotation_file'])
        assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['annotation_sha256']
        coco=COCO(str(annotation))
        ids=manifest['image_ids']
        for name,records in predictions.items():
            print('Official aggregate/block evaluation '+name,flush=True)
            for group,values in replication_ap(coco,ids,records,manifest['replication_blocks']).items():
                metrics.setdefault(group,{})[name]=values
        write_json(output/'metrics.json',metrics)
        if study=='T019-A':
            from .native_component_study import component_selection
            summary=component_selection(metrics,all(final_isolation.values()))
        else:
            summary=confirmation(metrics,all(final_isolation.values())) if study=='T018-B' else advancement(metrics)
        write_json(output/'summary.json',summary)
    write_json(output/'completion.json',{'status':'completed','smoke':smoke,'images':len(images),'adaptive_samples':len(rows),
               'teacher_forwards':len(images)*7,'evaluations':sum(len(v) for v in metrics.values()),
               'elapsed_seconds':time.perf_counter()-started,'all_isolation_passed':True,
               'gate':summary['gate'] if summary else None})
    print('Finished '+study+' '+('runtime smoke (no AP)' if smoke else json.dumps(summary['gate'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/t018a.yaml'))
    p.add_argument('--manifest',type=Path,default=Path('research_log/T018A_train_cohort.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    run(yaml.safe_load(args.config.read_text()),args.manifest,args.output,args.smoke)
