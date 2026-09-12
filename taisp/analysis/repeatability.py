"""T013-D no-update repeats and saved-data algebra; never trains a model."""
import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.parameter_predictor import ParameterPredictor
from .gradient_conflict import evaluate, geometry, loss_means, write_json
from .source_meta_smoke import load_episodes, frozen_unchanged

PINS = {
    'manifest': '164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493',
    'supports': '6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35',
    'prior': '7c35fb9e5f7b1c923c06f625ddf806d443e763b56aa52d8027594f0b95cce943',
    'original': 'a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120',
}


def stats(values):
    x = torch.as_tensor(values, dtype=torch.double)
    return {'mean': x.mean().item(), 'std_population': x.std(unbiased=False).item(),
            'min': x.min().item(), 'max': x.max().item(), 'range': (x.max()-x.min()).item(),
            'median': x.quantile(.5).item(), 'max_abs_deviation_from_r0': (x-x[0]).abs().max().item()}


def effect_scale(effect, values):
    x = torch.as_tensor(values, dtype=torch.double)
    control = stats(x)
    return {'signed_effect': float(effect), 'control': control,
            'abs_effect_over_std': abs(float(effect))/control['std_population'] if control['std_population'] else None,
            'abs_effect_over_range': abs(float(effect))/control['range'] if control['range'] else None,
            'count_equal_or_larger_abs_change': int(((x-x[0]).abs() >= abs(float(effect))).sum()),
            'repeat_denominator_including_r0': len(x)}


def cosine(a, b):
    a, b = torch.as_tensor(a, dtype=torch.double), torch.as_tensor(b, dtype=torch.double)
    return (a@b/(a.norm()*b.norm()).clamp_min(1e-12)).item()


def head_vector(row):
    return torch.cat((torch.tensor(row['head_weight_gradient'], dtype=torch.double).flatten(),
                       torch.tensor(row['head_bias_gradient'], dtype=torch.double)))


def summarize(repeats, prior):
    episode_summary = []
    for i in range(8):
        rows = [r['episodes'][i] for r in repeats]
        phi = torch.tensor([r['phi3'] for r in rows], dtype=torch.double)
        episode_summary.append({
            'image_id': rows[0]['image_id'], 'case': rows[0]['case'],
            'loss': stats([r['outer_loss'] for r in rows]),
            'phi3_radius': stats(phi.norm(dim=1)), 'phi3_coordinates': [stats(phi[:,j]) for j in range(8)],
            'phi3_max_coordinate_deviation': (phi-phi[0]).abs().max().item(),
            'phi_gradient_cosines_to_r0': [cosine(r['phi0_gradient'], rows[0]['phi0_gradient']) for r in rows],
            'head_gradient_cosines_to_r0': [cosine(head_vector(r), head_vector(rows[0])) for r in rows]})
    aggregate = {}
    for space in ('phi', 'head'):
        rows = [r['geometry'][space] for r in repeats]
        cs = [r['cosine'] for r in rows]
        aggregate[space] = {'cosines': cs, 'cosine_stats': stats(cs),
                            'negative_count': sum(c < 0 for c in cs),
                            'all_same_sign': len({(c > 0)-(c < 0) for c in cs}) == 1,
                            'clean_norm_stats': stats([r['clean_norm'] for r in rows]),
                            'corrupted_norm_stats': stats([r['corrupted_norm'] for r in rows])}
    group_losses = {name: [loss_means(r['episodes'])[name] for r in repeats]
                    for name in ('joint', 'clean', 'corrupted')}
    comparisons = {}
    baseline = prior['gradient_audit']['episodes']
    for name, probe in prior['probes'].items():
        group = {g: effect_scale(probe['delta_comparison']['actual_group_means'][g], group_losses[g])
                 for g in group_losses}
        episodes = []
        for i, (before, after) in enumerate(zip(baseline, probe['episodes'])):
            rows = [r['episodes'][i] for r in repeats]
            coordinate = [effect_scale(after['phi3'][j]-before['phi3'][j], [r['phi3'][j] for r in rows]) for j in range(8)]
            episodes.append({'image_id': before['image_id'], 'case': before['case'],
                             'loss': effect_scale(after['outer_loss']-before['outer_loss'], [r['outer_loss'] for r in rows]),
                             'phi3_radius': effect_scale(after['phi3_norm']-before['phi3_norm'], [r['phi3_norm'] for r in rows]),
                             'phi3_coordinates': coordinate})
        radius_groups = {}
        for g, indices in [('joint', range(8)), ('clean', range(0,8,2)), ('corrupted', range(1,8,2))]:
            effect = sum(probe['episodes'][i]['phi3_norm']-baseline[i]['phi3_norm'] for i in indices)/len(indices)
            values = [sum(r['episodes'][i]['phi3_norm'] for i in indices)/len(indices) for r in repeats]
            radius_groups[g] = effect_scale(effect, values)
        comparisons[name] = {'group_loss_effects': group, 'group_radius_effects': radius_groups, 'episodes': episodes}
    return {'episode_variability': episode_summary, 'aggregate_geometry': aggregate,
            'group_loss_distributions': {g: {'values': values, **stats(values)} for g, values in group_losses.items()},
            't013c_fixed_effect_comparison': comparisons,
            'interpretation': 'Descriptive no-update repeatability controls; not p-values, confidence intervals or optimizer ranking.'}


def output_summary(outputs):
    centered = outputs-outputs.mean(0)
    energy = outputs.square().sum()
    distances = torch.cdist(outputs, outputs)
    return {'outputs': outputs.tolist(), 'total_energy': energy.item(),
            'centered_energy': centered.square().sum().item(),
            'centered_energy_fraction': (centered.square().sum()/energy).item(),
            'pairwise_distances': distances.tolist(),
            'same_image_distances': [distances[i,i+1].item() for i in range(0,8,2)],
            'different_image_distances': [{'episodes':[i,j], 'distance':distances[i,j].item()}
                                          for i in range(8) for j in range(i+1,8) if i//2 != j//2],
            'centered_singular_values': torch.linalg.svdvals(centered).tolist(),
            'centered_numerical_rank': torch.linalg.matrix_rank(centered).item()}


def common_mode(prior):
    rows = prior['gradient_audit']['episodes']
    # Reconstruct literal float32 SGD head update, then analyze in float64.
    head = torch.stack([torch.cat((torch.tensor(r['head_weight_gradient']).flatten(),
                                   torch.tensor(r['head_bias_gradient']))) for r in rows]).mean(0)
    learned = torch.zeros_like(head).add_(head, alpha=-1e-3).double()
    weight, bias = learned[:-8].reshape(8,16), learned[-8:]
    saved = prior['probes']['joint']['conditioning']
    features = torch.tensor(saved['features'], dtype=torch.double)
    wh = features@weight.T
    outputs = wh+bias
    mean = wh.mean(0)
    centered = wh-mean
    return {'reconstructed_weight': weight.tolist(), 'reconstructed_bias': bias.tolist(),
            'max_saved_wh_difference': (wh-torch.tensor(saved['weight_term'], dtype=torch.double)).abs().max().item(),
            'max_saved_bias_difference': (bias-torch.tensor(saved['bias'], dtype=torch.double)).abs().max().item(),
            'max_saved_output_difference': (outputs-torch.tensor(saved['outputs'], dtype=torch.double)).abs().max().item(),
            'per_episode_norms': [{'wh': w.norm().item(), 'common': mean.norm().item(), 'centered': c.norm().item()}
                                  for w,c in zip(wh,centered)],
            'wh_only': output_summary(wh), 'full_original_output': output_summary(outputs),
            'bias_removed': output_summary(wh),
            'common_mode_removed_upper_bound': output_summary(outputs-outputs.mean(0)),
            'interpretation': 'Exactly two algebraic counterfactuals; whole-microset centering is not deployable. No model calls or training.'}


def run(paths, output):
    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[key], key
    prior = json.loads(paths['prior'].read_text(encoding='utf-8'))
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    supports = json.loads(paths['supports'].read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    source, clip = load_detector(device), load_clip_guidance(device, local_files_only=True)
    isp, original = DifferentiableISP().to(device), ParameterPredictor().to(device)
    saved_state = torch.load(paths['original'], map_location=device, weights_only=True)
    assert all(torch.equal(v, saved_state[k]) for k,v in original.state_dict().items())
    from .run_t002 import environment_metadata
    environment = environment_metadata({'seed':20260913, 'repeats':12, 'inner_steps':3, 'inner_lr':.1})
    assert environment['detector_sha256'] == '258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    assert environment['clip_sha256'] == 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    environment['interpretation'] = 'T013-D fixed train microset no-update repeatability; no validation or AP.'
    write_json(output/'environment.json', environment)
    models = (source, clip)
    states = [{k:v.clone() for k,v in model.state_dict().items()} for model in models]
    episodes = load_episodes(manifest, device)
    for episode, saved in zip(episodes,supports):
        assert (episode['image_id'],episode['case']) == (saved['image_id'],saved['case'])
        selected = {k:torch.tensor(saved[k], device=device, dtype=torch.long if k=='labels' else torch.float32)
                    for k in ('boxes','labels','scores')}
        assert all(selected[k].cpu().tolist() == saved[k] for k in selected)
        episode['native'] = DetectorNativeLoss(source, {'base':selected}, 'det_pseudo')
    torch.cuda.reset_peak_memory_stats()
    repeats = []
    for index in range(12):
        torch.manual_seed(20260913)
        predictor = copy.deepcopy(original)
        rows, pg, hg = evaluate(predictor, isp, source, clip, episodes, gradients=True)
        assert frozen_unchanged(models,states)
        assert all(torch.equal(v,saved_state[k]) for k,v in predictor.state_dict().items())
        result = {'repeat':index, 'episodes':rows, 'geometry':geometry(pg,hg),
                  'frozen_models_unchanged':True, 'predictor_unchanged':True}
        write_json(output/f'repeat_{index:02d}.json', result)
        repeats.append(result)
        print(json.dumps({'repeat':index, 'loss':loss_means(rows),
                          'phi_cosine':result['geometry']['phi']['cosine'],
                          'head_cosine':result['geometry']['head']['cosine']}), flush=True)
    summary = summarize(repeats,prior)
    algebra = common_mode(prior)
    write_json(output/'repeatability_summary.json',summary)
    write_json(output/'common_mode.json',algebra)
    # Optional branch only after primary outcomes are durably saved.
    deterministic = {'attempted':True, 'settings_change':'torch.use_deterministic_algorithms(True) only', 'repeats':[]}
    try:
        torch.use_deterministic_algorithms(True)
        for index in range(3):
            torch.manual_seed(20260913)
            rows, _, _ = evaluate(copy.deepcopy(original),isp,source,clip,episodes[:2],gradients=True)
            deterministic['repeats'].append({'repeat':index,'episodes':rows})
        deterministic['status'] = 'completed_three_pair_repeats'
    except RuntimeError:
        deterministic.update(status='raised_stopped_branch', traceback=traceback.format_exc())
    finally:
        torch.use_deterministic_algorithms(False)
    write_json(output/'deterministic_diagnostic.json',deterministic)
    assert frozen_unchanged(models,states)
    assert all(torch.equal(v,saved_state[k]) for k,v in original.state_dict().items())
    assert torch.count_nonzero(isp.phi) == 0 and isp.phi.grad is None
    write_json(output/'completion.json',{'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
              'pins':PINS,'primary_repeats':12,'episodes_per_repeat':8,'optimizer_steps':0,
              'frozen_models_unchanged':True,'original_state_verified':True,
              'all_primary_results_finite':True,'optional_status':deterministic['status'],
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
              'elapsed_seconds':time.perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PINS:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run({key:getattr(args,key) for key in PINS},args.output)
