"""T013-F: immutable T013-C checkpoints, eight matched cycles, no optimizer."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.parameter_predictor import ParameterPredictor
from .gradient_conflict import evaluate, loss_means, write_json
from .source_meta_smoke import load_episodes, frozen_unchanged, parameter_vector
from .repeatability import PINS
from .deterministic_replay import state_hash

CHECKPOINTS = {
    'original': ('original_predictor.pt', PINS['original']),
    'joint': ('joint_one_step_predictor.pt', 'fec4271dcfda344023521c17070d117d08082664fdb6ddd6934adb47e727c634'),
    'clean': ('clean_one_step_predictor.pt', '06e424abceb8004ac35538e41636814cd5ba5dc85bc4d191429a93dbd64b9cb8'),
    'corrupted': ('corrupted_one_step_predictor.pt', 'b1edfca935f8e8dc90206f11b123e6d40524588cc4351f767d4af79dc8d5ba4e'),
}


def schedule():
    pairs = ['null','joint','clean','corrupted']
    return [{'cycle':i+1, 'pairs':pairs[i%4:]+pairs[:i%4],
             'roles':['baseline','probe'] if i%2 == 0 else ['probe','baseline']} for i in range(8)]


def distribution(values):
    return {'values':values,'median':statistics.median(values),'min':min(values),'max':max(values),
            'range':max(values)-min(values),'negative_count':sum(v < 0 for v in values),
            'zero_count':sum(v == 0 for v in values),'positive_count':sum(v > 0 for v in values)}


def resolved_effect(corrected, null):
    summary = distribution(corrected)
    median = summary['median']
    count = sum((v > 0)-(v < 0) == (median > 0)-(median < 0) for v in corrected)
    floor = max(abs(v) for v in null)
    resolved = count >= 7 and abs(median) > floor
    return {**summary,'median_sign_count':count,'max_absolute_null':floor,
            'resolved':resolved,'direction':('improvement' if median < 0 else 'worsening') if resolved else 'unresolved'}


def decision(effects):
    def has(probe, group, direction):
        return effects[probe][group]['corrected']['direction'] == direction
    cross_clean = has('clean','clean','improvement') and has('clean','corrupted','worsening')
    cross_corrupt = has('corrupted','corrupted','improvement') and has('corrupted','clean','worsening')
    both = {p:all(has(p,g,'improvement') for g in ('clean','corrupted')) for p in effects}
    if cross_clean or cross_corrupt:
        rule = 'functionally_active_source_objective_conflict'
    elif both['joint'] or (both['clean'] and both['corrupted']):
        rule = 'resolved_both_group_improvement_not_immediate_finite_step_bottleneck'
    else:
        rule = 'measurement_limited_unresolved_or_mixed'
    return {'rule':rule,'clean_only_cross_harm':cross_clean,'corrupt_only_cross_harm':cross_corrupt,
            'resolved_both_group_improvement':both,'next_action':'Stop for research review; no T013-G or method change.'}


def summarize(cycles):
    per_cycle = []
    for cycle in cycles:
        paired = {}
        for pair in cycle['pairs']:
            by_role = {e['role']:e['episodes'] for e in pair['evaluations']}
            deltas = [{'outer_loss':b['outer_loss']-a['outer_loss']}
                      for a,b in zip(by_role['baseline'],by_role['probe'],strict=True)]
            paired[pair['pair']] = {'episode_loss_deltas':[r['outer_loss'] for r in deltas],
                                   'groups':loss_means(deltas)}
        corrected = {p:{g:paired[p]['groups'][g]-paired['null']['groups'][g]
                        for g in ('joint','clean','corrupted')} for p in ('joint','clean','corrupted')}
        per_cycle.append({'cycle':cycle['cycle'],'paired':paired,'corrected':corrected})
    nulls = {g:[c['paired']['null']['groups'][g] for c in per_cycle] for g in ('joint','clean','corrupted')}
    effects = {p:{g:{'paired':distribution([c['paired'][p]['groups'][g] for c in per_cycle]),
                       'corrected':resolved_effect([c['corrected'][p][g] for c in per_cycle],nulls[g])}
                  for g in nulls} for p in ('joint','clean','corrupted')}
    return {'cycles':per_cycle,'null':{g:{'signed':distribution(v),'absolute':distribution([abs(x) for x in v])}
                                      for g,v in nulls.items()},'effects':effects,'decision':decision(effects),
            'interpretation':'Eight dependent matched cycles on four images; descriptive noise-floor rule, no population significance or confidence intervals.'}


def run(manifest_path, supports_path, prior_root, output):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    for key,path in [('manifest',manifest_path),('supports',supports_path),('prior',prior_root/'receipt.json')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[key], key
    for filename,checksum in CHECKPOINTS.values():
        assert hashlib.sha256((prior_root/filename).read_bytes()).hexdigest() == checksum
    prior = json.loads((prior_root/'receipt.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    source,clip = load_detector(device),load_clip_guidance(device,local_files_only=True)
    clip.eval()
    isp,original = DifferentiableISP().to(device),ParameterPredictor().to(device)
    original_state = torch.load(prior_root/CHECKPOINTS['original'][0],map_location=device,weights_only=True)
    assert all(torch.equal(v,original_state[k]) for k,v in original.state_dict().items())
    templates = {'original':original}
    freeze = {}
    for name,(filename,checksum) in CHECKPOINTS.items():
        shutil.copyfile(prior_root/filename,output/filename)
        if name != 'original':
            predictor = copy.deepcopy(original)
            predictor.load_state_dict(torch.load(prior_root/filename,map_location=device,weights_only=True))
            templates[name] = predictor
            changed = [k for k,v in predictor.state_dict().items() if not torch.equal(v,original_state[k])]
            assert changed and all(k.startswith('head.') for k in changed)
            direction = torch.tensor(prior['probes'][name]['direction'],device=device)
            literal = torch.zeros_like(direction).add_(direction,alpha=-1e-3)
            head = torch.cat((predictor.head.weight.detach().flatten(),predictor.head.bias.detach().flatten()))
            assert torch.equal(head,literal)
            freeze[name] = {'file_sha256':checksum,'direction':direction.cpu().tolist(),'changed_state_names':changed,
                            'parameter_delta_norm':(parameter_vector(predictor)-parameter_vector(original)).norm().item(),
                            'exact_saved_step_verified':True,'state_sha256':state_hash(predictor)}
        else:
            freeze[name] = {'file_sha256':checksum,'state_sha256':state_hash(original),'parameter_delta_norm':0.}
    models = (source,clip)
    states = [{k:v.clone() for k,v in model.state_dict().items()} for model in models]
    frozen_hashes = [state_hash(model) for model in models]
    assert frozen_unchanged(models,states)
    assert all(p.grad is None for m in templates.values() for p in m.parameters())
    from .run_t002 import environment_metadata
    environment = environment_metadata({'seed':20260913,'inner_steps':3,'inner_lr':.1,'saved_outer_coefficient':1e-3,'cycles':8})
    assert environment['detector_sha256'] == '258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    assert environment['clip_sha256'] == 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    environment.update(interpretation='T013-F fixed train2017 matched checkpoint replay; no training or AP.',
                       deterministic_algorithms=False,cudnn_benchmark=False,cublas_workspace_config=None,
                       source_state_sha256=frozen_hashes[0],clip_state_sha256=frozen_hashes[1])
    write_json(output/'environment.json',environment)
    write_json(output/'checkpoint_freeze.json',{'checkpoints':freeze,'original_gradient_audit':prior['gradient_audit'],
              'optimizer_steps_in_t013f':0,'saved_coefficient':1e-3,'schedule':schedule(),
              'source_state_sha256':frozen_hashes[0],'clip_state_sha256':frozen_hashes[1],'initial_frozen_grad_none':True})
    episodes = load_episodes(json.loads(manifest_path.read_text()),device)
    for episode,saved in zip(episodes,json.loads(supports_path.read_text()),strict=True):
        assert (episode['image_id'],episode['case']) == (saved['image_id'],saved['case'])
        selected = {k:torch.tensor(saved[k],device=device,dtype=torch.long if k=='labels' else torch.float32)
                    for k in ('boxes','labels','scores')}
        assert all(selected[k].cpu().tolist() == saved[k] for k in selected)
        episode['native'] = DetectorNativeLoss(source,{'base':selected},'det_pseudo')
    torch.cuda.reset_peak_memory_stats()
    cycles = []
    for spec in schedule():
        cycle = {'cycle':spec['cycle'],'pair_order':spec['pairs'],'role_order':spec['roles'],'pairs':[]}
        for position,name in enumerate(spec['pairs']):
            pair = {'pair':name,'block_position':position,'evaluations':[]}
            for eval_position,role in enumerate(spec['roles']):
                checkpoint = name if role == 'probe' and name != 'null' else 'original'
                predictor = copy.deepcopy(templates[checkpoint])
                torch.manual_seed(20260913)
                rows,_,_ = evaluate(predictor,isp,source,clip,episodes)
                current_hashes = [state_hash(model) for model in models]
                assert current_hashes == frozen_hashes and frozen_unchanged(models,states)
                assert state_hash(predictor) == freeze[checkpoint]['state_sha256']
                assert all(p.grad is None for p in predictor.parameters())
                assert torch.count_nonzero(isp.phi) == 0 and isp.phi.grad is None
                for row in rows:
                    row.update(cycle=spec['cycle'],pair=name,block_position=position,role=role,
                               evaluation_position=eval_position,empty_support=row['support_count']==0)
                result = {'cycle':spec['cycle'],'pair':name,'block_position':position,'role':role,
                          'evaluation_position':eval_position,'checkpoint':checkpoint,'episodes':rows,
                          'predictor_unchanged':True,'predictor_grad_none':True,'isp_unchanged':True,
                          'frozen_models_unchanged':True,'frozen_grad_none':True,
                          'source_state_sha256':current_hashes[0],'clip_state_sha256':current_hashes[1]}
                write_json(output/f"cycle_{spec['cycle']:02d}_block_{position}_{name}_{role}.json",result)
                pair['evaluations'].append(result)
            cycle['pairs'].append(pair)
        write_json(output/f"cycle_{spec['cycle']:02d}.json",cycle)
        cycles.append(cycle)
        print(json.dumps({'cycle':spec['cycle'],'raw_evaluations_saved':8,'pair_order':spec['pairs']}),flush=True)
    # All raw evaluations have been saved before any effect summary.
    summary = summarize(cycles)
    write_json(output/'summary.json',summary)
    assert all(state_hash(t) == freeze[n]['state_sha256'] for n,t in templates.items())
    assert all(hashlib.sha256((prior_root/f).read_bytes()).hexdigest()==h for f,h in CHECKPOINTS.values())
    write_json(output/'completion.json',{'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),'status':'completed',
               'cycles':8,'pairs_per_cycle':4,'evaluations':64,'episode_rows':512,'optimizer_steps':0,
               'all_checkpoints_unchanged':True,'frozen_models_unchanged':True,'decision':summary['decision'],
               'elapsed_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','supports','prior-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(args.manifest,args.supports,args.prior_root,args.output)
