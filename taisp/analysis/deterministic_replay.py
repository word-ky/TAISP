"""T013-E deterministic measurement gate and the three frozen T013-C probes."""
import argparse
import copy
import hashlib
import json
import os
import struct
import time
import traceback
from pathlib import Path

ENV_BEFORE_TORCH = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.parameter_predictor import ParameterPredictor
from .gradient_conflict import evaluate, geometry, directions, one_step, loss_means, delta_comparison, write_json
from .source_meta_smoke import load_episodes, frozen_unchanged, group_statistics, parameter_vector
from .repeatability import PINS, head_vector

CONTROL_MANIFEST_SHA = '6e689f25b061f94018499d25d889f29143e79f64bfa593339f9c907093a40afb'


def first_mismatch(reference, actual, path='outputs'):
    """Exact retained-output comparison, including float signed zero; no tolerance."""
    if isinstance(reference, dict):
        for key in reference:
            mismatch = first_mismatch(reference[key], actual[key], f'{path}.{key}')
            if mismatch is not None:
                return mismatch
    elif isinstance(reference, list):
        for index, (a, b) in enumerate(zip(reference, actual, strict=True)):
            mismatch = first_mismatch(a, b, f'{path}[{index}]')
            if mismatch is not None:
                return mismatch
    elif isinstance(reference, float):
        a, b = struct.pack('>d', reference), struct.pack('>d', actual)
        if a != b:
            return {'path': path, 'reference': reference, 'actual': actual,
                    'reference_float64_hex': a.hex(), 'actual_float64_hex': b.hex()}
    elif reference != actual:
        return {'path': path, 'reference': reference, 'actual': actual}
    return None


def determinism_gate(repeats):
    for index in range(1, 3):
        for field in ('episodes', 'geometry'):
            mismatch = first_mismatch(repeats[0][field], repeats[index][field], field)
            if mismatch is not None:
                return {'passed': False, 'first_mismatch_repeat': index, 'first_mismatch': mismatch}
    return {'passed': True, 'first_mismatch': None, 'comparison': 'All retained episode and geometry scalar/vector bits, no tolerance.'}


def spearman(predicted, actual):
    def ranks(x):
        x = torch.as_tensor(x, dtype=torch.double)
        return (x[None, :] < x[:, None]).sum(1)+(1+(x[None, :] == x[:, None]).sum(1))/2
    return delta_comparison(ranks(predicted), ranks(actual))['pearson_r']


def decision(probes):
    delta = {name: p['delta_comparison']['actual_group_means'] for name,p in probes.items()}
    clean_harms = delta['clean']['clean'] < 0 and delta['clean']['corrupted'] > 0
    corrupt_harms = delta['corrupted']['corrupted'] < 0 and delta['corrupted']['clean'] > 0
    if clean_harms or corrupt_harms:
        result = 'functionally_active_source_objective_conflict'
    elif all(d['clean'] < 0 and d['corrupted'] < 0 for d in delta.values()):
        result = 'all_three_improve_both_not_immediate_finite_step_bottleneck'
    elif delta['joint']['clean']*delta['joint']['corrupted'] < 0:
        result = 'asymmetric_joint_finite_step_interaction'
    else:
        result = 'unresolved_pattern'
    return {'rule': result, 'clean_only_cross_harm': clean_harms, 'corrupt_only_cross_harm': corrupt_harms,
            'thresholds': 'strict signed deltas relative to zero, as predeclared; no tolerance',
            'next_action': 'Stop for research review. No T013-F or method change.'}


def prior_ranges(repeat, controls):
    def placement(value, prior):
        lo, hi = min(prior), max(prior)
        return {'value': value, 'prior_min': lo, 'prior_max': hi, 'inside_inclusive': lo <= value <= hi}
    groups = {g: placement(v, [loss_means(r['episodes'])[g] for r in controls])
              for g,v in loss_means(repeat['episodes']).items()}
    episodes = []
    for i,row in enumerate(repeat['episodes']):
        old = [r['episodes'][i] for r in controls]
        episodes.append({'image_id': row['image_id'], 'case': row['case'],
            **{f: placement(row[f], [r[f] for r in old]) for f in
               ('outer_loss','phi3_norm','phi0_gradient_norm','head_weight_gradient_norm','head_bias_gradient_norm')},
            'flattened_head_gradient_norm_float64': placement(head_vector(row).norm().item(), [head_vector(r).norm().item() for r in old]),
            'phi3_coordinates': [placement(x, [r['phi3'][j] for r in old]) for j,x in enumerate(row['phi3'])]})
    aggregate = {s: {f: placement(repeat['geometry'][s][f], [r['geometry'][s][f] for r in controls])
                     for f in ('cosine','clean_norm','corrupted_norm')} for s in ('phi','head')}
    return {'group_losses': groups, 'episodes': episodes, 'aggregate_geometry': aggregate,
            'interpretation': 'Descriptive comparison against all 12 T013-D repeats; outside-range values do not trigger tuning.'}


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{list(tensor.shape)}\n'.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run(paths, control_root, control_manifest, output):
    assert ENV_BEFORE_TORCH == ':4096:8'
    for key,path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[key], key
    assert hashlib.sha256(control_manifest.read_bytes()).hexdigest() == CONTROL_MANIFEST_SHA
    files = {r['path']: r['sha256'] for r in json.loads(control_manifest.read_text())}
    controls = []
    for i in range(12):
        relative = f'artifacts/audit/repeat_{i:02d}.json'
        data = (control_root/relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == files[relative]
        controls.append(json.loads(data))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    source, clip = load_detector(device), load_clip_guidance(device, local_files_only=True)
    # The shared inner loop does this before evaluation; initial isolation is earlier.
    clip.eval()
    isp, original = DifferentiableISP().to(device), ParameterPredictor().to(device)
    saved_state = torch.load(paths['original'], map_location=device, weights_only=True)
    assert all(torch.equal(v, saved_state[k]) for k,v in original.state_dict().items())
    models = (source, clip)
    states = [{k:v.clone() for k,v in m.state_dict().items()} for m in models]
    original_hashes = [state_hash(m) for m in models]

    def isolation():
        hashes = [state_hash(m) for m in models]
        assert hashes == original_hashes and frozen_unchanged(models, states)
        assert all(torch.equal(v, saved_state[k]) for k,v in original.state_dict().items())
        assert torch.count_nonzero(isp.phi) == 0 and isp.phi.grad is None
        return {'frozen_models_unchanged': True, 'frozen_grad_none': True,
                'source_state_sha256': hashes[0], 'clip_state_sha256': hashes[1],
                'original_predictor_unchanged': True, 'isp_unchanged': True}

    from .run_t002 import environment_metadata
    environment = environment_metadata({'seed':20260913,'inner_steps':3,'inner_lr':.1,'outer_lr':1e-3})
    assert environment['detector_sha256'] == '258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    assert environment['clip_sha256'] == 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    environment.update(interpretation='T013-E fixed train2017 deterministic replay; no validation or AP.',
                       cublas_workspace_config_before_torch=ENV_BEFORE_TORCH,
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                       cudnn_benchmark=torch.backends.cudnn.benchmark, pins=PINS,
                       control_manifest_sha256=CONTROL_MANIFEST_SHA, initial_isolation=isolation())
    write_json(output/'environment.json', environment)
    episodes = load_episodes(json.loads(paths['manifest'].read_text()), device)
    supports = json.loads(paths['supports'].read_text())
    for episode,saved in zip(episodes,supports,strict=True):
        assert (episode['image_id'],episode['case']) == (saved['image_id'],saved['case'])
        selected = {k:torch.tensor(saved[k],device=device,dtype=torch.long if k=='labels' else torch.float32)
                    for k in ('boxes','labels','scores')}
        assert all(selected[k].cpu().tolist() == saved[k] for k in selected)
        episode['native'] = DetectorNativeLoss(source, {'base':selected}, 'det_pseudo')
    torch.cuda.reset_peak_memory_stats()
    repeats = []
    for index in range(3):
        torch.manual_seed(20260913)
        predictor = copy.deepcopy(original)
        rows, pg, hg = evaluate(predictor,isp,source,clip,episodes,gradients=True)
        assert all(torch.equal(v,saved_state[k]) for k,v in predictor.state_dict().items())
        repeat = {'repeat':index,'episodes':rows,'geometry':geometry(pg,hg),'isolation':isolation()}
        write_json(output/f'repeat_{index:02d}.json',repeat)
        repeats.append(repeat)
        print(json.dumps({'repeat':index,'loss':loss_means(rows),'phi_cosine':repeat['geometry']['phi']['cosine']}),flush=True)
    gate = determinism_gate(repeats)
    write_json(output/'determinism_gate.json',gate)
    write_json(output/'prior_range_comparison.json',prior_ranges(repeats[0],controls))
    print(json.dumps({'determinism_gate':gate}),flush=True)
    if not gate['passed']:
        write_json(output/'completion.json',{'status':'stopped_exact_mismatch','gate':gate,'optimizer_steps':0,
                   'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),'isolation':isolation()})
        return
    baseline = repeats[0]['episodes']
    probes = {}
    for name,gradient in directions(hg).items():
        predictor = one_step(original,gradient.to(device))
        changed = [k for k,v in predictor.state_dict().items() if not torch.equal(v,saved_state[k])]
        assert changed and all(k.startswith('head.') for k in changed)
        assert all(torch.equal(v,original.features.state_dict()[k]) for k,v in predictor.features.state_dict().items())
        after,_,_ = evaluate(predictor,isp,source,clip,episodes)
        actual = torch.tensor([a['outer_loss']-b['outer_loss'] for a,b in zip(after,baseline)],dtype=torch.double)
        predicted = -1e-3*(hg.double()@gradient.double())
        comparison = delta_comparison(predicted,actual)
        comparison['spearman_rho_average_tie_ranks'] = spearman(predicted,actual)
        result = {'direction':gradient.tolist(),'coefficient':1e-3,'optimizer_steps':1,
                  'before_means':loss_means(baseline),'after_means':loss_means(after),
                  'delta_comparison':comparison,'episodes':after,
                  'clean':group_statistics(after,True),'corrupted':group_statistics(after,False),
                  'parameter_delta_norm':(parameter_vector(predictor)-parameter_vector(original)).norm().item(),
                  'changed_state_names':changed,'feature_trunk_unchanged':True,'isolation':isolation()}
        torch.save(predictor.state_dict(),output/f'{name}_one_step_predictor.pt')
        write_json(output/f'{name}.json',result)
        probes[name] = result
        print(json.dumps({'probe':name,'delta':comparison['actual_group_means']}),flush=True)
    write_json(output/'receipt.json',{'gradient_audit':repeats[0],'probes':probes,'decision':decision(probes)})
    write_json(output/'completion.json',{'status':'completed','source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
               'primary_repeats':3,'gate':gate,'independent_one_step_probes':3,'optimizer_steps':3,
               'isolation':isolation(),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
               'elapsed_seconds':time.perf_counter()-started,'decision':decision(probes)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PINS:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--control-root',type=Path,required=True)
    parser.add_argument('--control-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        run({key:getattr(args,key) for key in PINS},args.control_root,args.control_manifest,args.output)
    except RuntimeError:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'runtime_failure.json',{'status':'stopped_runtime_error','traceback':traceback.format_exc(),
                   'cublas_workspace_config_before_torch':ENV_BEFORE_TORCH,'no_retry':True})
        raise
