"""T013-C analysis-only gradient geometry, independent probes and conditioning."""
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.parameter_predictor import ParameterPredictor
from .source_meta_smoke import load_episodes, source_outer_episode, frozen_unchanged, group_statistics


def relation(a, b):
    dot = torch.dot(a, b)
    return {'dot': dot.item(), 'cosine': (dot/(a.norm()*b.norm()).clamp_min(1e-12)).item(),
            'clean_norm': a.norm().item(), 'corrupted_norm': b.norm().item(),
            'clean_to_corrupted_norm_ratio': (a.norm()/(b.norm()+1e-12)).item()}


def geometry(phi_gradients, head_gradients):
    pg, hg = phi_gradients.double(), head_gradients.double()
    pairwise = pg@pg.T/(pg.norm(dim=1)[:, None]*pg.norm(dim=1)[None, :]).clamp_min(1e-12)
    result = {'phi_pairwise_cosines': pairwise.tolist(),
              'same_image_clean_corrupted_cosines': [pairwise[i, i+1].item() for i in range(0, 8, 2)]}
    for name, gradients in [('phi', pg), ('head', hg)]:
        clean, corrupted = gradients[::2].mean(0), gradients[1::2].mean(0)
        result[name] = {**relation(clean, corrupted), 'clean_mean': clean.tolist(),
                        'corrupted_mean': corrupted.tolist(),
                        'coordinate_sign_agreement': (clean.sign() == corrupted.sign()).tolist(),
                        'sign_agreement_fraction': (clean.sign() == corrupted.sign()).double().mean().item()}
    return result


def directions(head_gradients):
    return {'joint': head_gradients.mean(0), 'clean': head_gradients[::2].mean(0),
            'corrupted': head_gradients[1::2].mean(0)}


def one_step(original, gradient):
    """Independent original copy, one literal SGD step; zero trunk gradient."""
    predictor = copy.deepcopy(original)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-3)
    for p in predictor.parameters():
        p.grad = torch.zeros_like(p)
    size = predictor.head.weight.numel()
    predictor.head.weight.grad.copy_(gradient[:size].reshape_as(predictor.head.weight))
    predictor.head.bias.grad.copy_(gradient[size:].reshape_as(predictor.head.bias))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return predictor


def loss_means(rows):
    return {'joint': sum(r['outer_loss'] for r in rows)/8,
            'clean': sum(r['outer_loss'] for r in rows[::2])/4,
            'corrupted': sum(r['outer_loss'] for r in rows[1::2])/4}


def delta_comparison(predicted, actual):
    predicted, actual = predicted.double(), actual.double()
    a, b = predicted-predicted.mean(), actual-actual.mean()
    denominator = a.norm()*b.norm()
    return {'predicted': predicted.tolist(), 'actual': actual.tolist(),
            'sign_agreement': (predicted.sign() == actual.sign()).tolist(),
            'sign_agreement_fraction': (predicted.sign() == actual.sign()).double().mean().item(),
            'pearson_r': (a@b/denominator).item() if denominator > 0 else None,
            'predicted_group_means': {'joint': predicted.mean().item(), 'clean': predicted[::2].mean().item(),
                                      'corrupted': predicted[1::2].mean().item()},
            'actual_group_means': {'joint': actual.mean().item(), 'clean': actual[::2].mean().item(),
                                   'corrupted': actual[1::2].mean().item()}}


@torch.no_grad()
def conditioning(predictor, images):
    h = torch.cat([predictor.features(x) for x in images]).double().cpu()
    weight, bias = predictor.head.weight.double().cpu(), predictor.head.bias.double().cpu()
    wh = h@weight.T
    outputs = wh+bias
    centered = outputs-outputs.mean(0)
    ew, eb = wh.square().sum(), 8*bias.square().sum()
    cross, total = 2*(wh*bias).sum(), outputs.square().sum()
    distance = torch.cdist(outputs, outputs)
    return {'features': h.tolist(), 'weight_term': wh.tolist(), 'bias': bias.tolist(),
            'outputs': outputs.tolist(), 'weight_term_norms': wh.norm(dim=1).tolist(),
            'bias_norm': bias.norm().item(), 'output_norms': outputs.norm(dim=1).tolist(),
            'population_std_per_coordinate': centered.square().mean(0).sqrt().tolist(),
            'population_covariance': (centered.T@centered/8).tolist(),
            'centered_singular_values': torch.linalg.svdvals(centered).tolist(),
            'pairwise_distances': distance.tolist(),
            'same_image_clean_corrupted_distances': [distance[i, i+1].item() for i in range(0, 8, 2)],
            'different_image_distances': [{'episodes': [i, j], 'distance': distance[i, j].item()}
                                          for i in range(8) for j in range(i+1, 8) if i//2 != j//2],
            'weight_term_energy': ew.item(), 'bias_energy': eb.item(), 'cross_energy': cross.item(),
            'total_output_energy': total.item(), 'component_weight_share': (ew/(ew+eb)).item(),
            'component_bias_share': (eb/(ew+eb)).item(), 'weight_over_total': (ew/total).item(),
            'bias_over_total': (eb/total).item(), 'cross_over_total': (cross/total).item(),
            'centered_energy_over_total': (centered.square().sum()/total).item(),
            'interpretation': 'Descriptive nonorthogonal terms; Wh may itself contain a shared component.'}


def evaluate(predictor, isp, source, clip, episodes, *, gradients=False):
    rows, phi_gradients, head_gradients = [], [], []
    for episode in episodes:
        result, loss, components = source_outer_episode(episode['image'], episode['targets'],
                                                        predictor, isp, episode['native'], clip, source)
        row = {'image_id': episode['image_id'], 'case': episode['case'], 'outer_loss': loss.item(),
               'components': {k: v.item() for k, v in components.items()},
               'phi0': result.phi0.detach().cpu().tolist(), 'phi3': result.phi.detach().cpu().tolist(),
               'phi0_norm': result.phi0.norm().item(), 'phi3_norm': result.phi.norm().item(),
               'saturation_rate': result.diagnostics[-1]['saturation_rate'],
               'support_count': len(episode['native'].boxes)}
        if gradients:
            gp, gw, gb, *trunk = torch.autograd.grad(loss, [result.phi0, predictor.head.weight,
                                                          predictor.head.bias, *predictor.features.parameters()])
            assert all(torch.isfinite(g).all() for g in (gp, gw, gb, *trunk))
            assert all(torch.count_nonzero(g) == 0 for g in trunk)
            row.update(phi0_gradient=gp.detach().cpu().tolist(), phi0_gradient_norm=gp.norm().item(),
                       head_weight_gradient=gw.detach().cpu().tolist(), head_bias_gradient=gb.detach().cpu().tolist(),
                       head_weight_gradient_norm=gw.norm().item(), head_bias_gradient_norm=gb.norm().item(),
                       trunk_gradient_norm=0.)
            phi_gradients.append(gp.detach().cpu())
            head_gradients.append(torch.cat((gw.detach().flatten(), gb.detach().flatten())).cpu())
        assert torch.isfinite(loss) and torch.isfinite(result.phi).all()
        rows.append(row)
    return rows, (torch.stack(phi_gradients) if gradients else None), (torch.stack(head_gradients) if gradients else None)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(manifest_path, supports_path, output):
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == '164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493'
    assert hashlib.sha256(supports_path.read_bytes()).hexdigest() == '6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    supports = json.loads(supports_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    # Same seed and constructor order as T013-B, before any other RNG operation.
    source, clip = load_detector(device), load_clip_guidance(device, local_files_only=True)
    isp, original = DifferentiableISP().to(device), ParameterPredictor().to(device)
    original_state = {k: v.clone() for k, v in original.state_dict().items()}
    torch.save(original_state, output/'original_predictor.pt')
    assert torch.count_nonzero(original.head.weight) == torch.count_nonzero(original.head.bias) == 0
    models = (source, clip)
    states = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    episodes = load_episodes(manifest, device)
    for episode, saved in zip(episodes, supports):
        assert (episode['image_id'], episode['case']) == (saved['image_id'], saved['case'])
        selected = {k: torch.tensor(saved[k], device=device, dtype=torch.long if k == 'labels' else torch.float32)
                    for k in ('boxes', 'labels', 'scores')}
        assert all(selected[k].cpu().tolist() == saved[k] for k in selected)
        episode['native'] = DetectorNativeLoss(source, {'base': selected}, 'det_pseudo')
    torch.cuda.reset_peak_memory_stats()
    baseline, pg, hg = evaluate(original, isp, source, clip, episodes, gradients=True)
    assert frozen_unchanged(models, states)
    audit = {'episodes': baseline, 'geometry': geometry(pg, hg), 'frozen_models_unchanged': True}
    write_json(output/'gradients.json', audit)
    print(json.dumps({'stage': 'gradient_geometry', 'phi': audit['geometry']['phi']['cosine'],
                      'head': audit['geometry']['head']['cosine']}), flush=True)
    probes = {}
    for name, gradient in directions(hg).items():
        predictor = one_step(original, gradient.to(device))
        assert all(torch.equal(original.state_dict()[k], v) for k, v in original_state.items())
        assert all(torch.equal(predictor.features.state_dict()[k], v) for k, v in original.features.state_dict().items())
        after, _, _ = evaluate(predictor, isp, source, clip, episodes)
        actual = torch.tensor([a['outer_loss']-b['outer_loss'] for a, b in zip(after, baseline)], dtype=torch.double)
        predicted = -1e-3*(hg.double()@gradient.double())
        assert frozen_unchanged(models, states)
        probe = {'direction': gradient.tolist(), 'coefficient': 1e-3, 'optimizer_steps': 1,
                 'before_means': loss_means(baseline), 'after_means': loss_means(after),
                 'delta_comparison': delta_comparison(predicted, actual), 'episodes': after,
                 'clean': group_statistics(after, True), 'corrupted': group_statistics(after, False),
                 'frozen_models_unchanged': True, 'original_unchanged': True, 'trunk_unchanged': True}
        if name == 'joint':
            probe['conditioning'] = conditioning(predictor, [e['image'] for e in episodes])
        torch.save(predictor.state_dict(), output/f'{name}_one_step_predictor.pt')
        write_json(output/f'{name}.json', probe)
        probes[name] = probe
        print(json.dumps({'probe': name, 'after': probe['after_means'],
                          'delta': probe['delta_comparison']['actual_group_means']}), flush=True)
    repeated, _, _ = evaluate(original, isp, source, clip, episodes)
    assert all(torch.equal(original.state_dict()[k], v) for k, v in original_state.items())
    assert torch.count_nonzero(isp.phi) == 0 and isp.phi.grad is None and frozen_unchanged(models, states)
    repeat = {'episodes': repeated, 'means': loss_means(repeated),
              'loss_deltas': [a['outer_loss']-b['outer_loss'] for a, b in zip(repeated, baseline)],
              'max_phi0_difference': max(abs(x-y) for a, b in zip(repeated, baseline) for x, y in zip(a['phi0'], b['phi0'])),
              'max_phi3_difference': max(abs(x-y) for a, b in zip(repeated, baseline) for x, y in zip(a['phi3'], b['phi3']))}
    receipt = {'source_revision': os.environ.get('TAISP_SOURCE_REVISION'), 'seed': 20260913,
               'python': platform.python_version(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
               'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'supports_sha256': hashlib.sha256(supports_path.read_bytes()).hexdigest(),
               'inner_steps': 3, 'inner_lr': .1, 'norm_epsilon': 1e-12, 'outer_coefficient': 1e-3,
               'independent_one_step_probes': 3, 'gradient_audit': audit, 'probes': probes,
               'original_no_update_repeat': repeat, 'frozen_models_unchanged': True,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'elapsed_seconds': time.perf_counter()-started}
    write_json(output/'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--supports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.supports, args.output)
