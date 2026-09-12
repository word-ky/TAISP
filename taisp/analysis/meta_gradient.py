"""T013-B tiny mock-only exact/FO/finite-difference audit; not deployment."""
import argparse
import csv
import json
import platform
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from taisp import DifferentiableISP
from taisp.training.initialization import adapt_from_initialization
from taisp.tta.trust_radius import transfer_norm


class MockQuadratic(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', torch.tensor(value, dtype=torch.float64))
        self.boxes = torch.ones(1, 4, dtype=torch.float64)

    def forward(self, original, enhanced):
        return (enhanced-self.value).square().mean()


def fixture(index):
    j, group = index % 3, index // 3
    x = torch.linspace(.08+.015*j, .70+.025*j, 3*9*11,
                       dtype=torch.float64).reshape(1, 3, 9, 11)
    phi0 = [0., .08, .20, .35][group]*torch.linspace(-1, 1, 8, dtype=torch.float64)
    return x, phi0, DifferentiableISP().double(), MockQuadratic(.9), MockQuadratic(.4)


def outer_loss(enhanced):
    return (enhanced-.55).square().mean()


def reference_unroll(image, isp, detector, semantic, phi0, *, create_graph=True):
    """Mock-only exact reference; identical K=3/lr=.1/norm-transfer forward."""
    phi = phi0.clone().requires_grad_(True)
    trajectory = [phi.detach().clone()]
    for _ in range(3):
        enhanced = isp(image, phi)
        gd = torch.autograd.grad(detector(image, enhanced), phi,
                                 create_graph=create_graph, retain_graph=True)[0]
        gc = torch.autograd.grad(semantic(image, enhanced), phi,
                                 create_graph=create_graph)[0]
        update, _ = transfer_norm(gd, gc, 1e-12)
        phi = phi-.1*update
        if not create_graph:
            phi = phi.detach().requires_grad_(True)
        trajectory.append(phi.detach().clone())
    return phi, isp(image, phi), torch.stack(trajectory)


def finite_difference(image, isp, detector, semantic, phi0, epsilon=1e-5):
    gradients = []
    for offset in torch.eye(8, dtype=phi0.dtype)*epsilon:
        values = []
        for initial in (phi0.detach()+offset, phi0.detach()-offset):
            _, image_k, _ = reference_unroll(image, isp, detector, semantic,
                                             initial, create_graph=False)
            values.append(outer_loss(image_k).item())
        gradients.append((values[0]-values[1])/(2*epsilon))
    return torch.tensor(gradients, dtype=phi0.dtype)


def audit_episode(index):
    x, phi0, isp, detector, semantic = fixture(index)
    phi0.requires_grad_(True)
    fo_result = adapt_from_initialization(x, isp, detector, semantic, phi0)
    fo = torch.autograd.grad(outer_loss(fo_result.enhanced), phi0)[0]
    exact_phi, exact_image, trajectory = reference_unroll(x, isp, detector, semantic, phi0)
    exact = torch.autograd.grad(outer_loss(exact_image), phi0)[0]
    fd = finite_difference(x, isp, detector, semantic, phi0)
    fo_trajectory = torch.tensor([d['phi'] for d in fo_result.diagnostics], dtype=phi0.dtype)
    cosine = lambda a, b: F.cosine_similarity(a, b, dim=0, eps=1e-12).item()
    return {
        'episode': index, 'phi0': phi0.detach().tolist(),
        'input_min': x.min().item(), 'input_max': x.max().item(),
        'fo': fo.tolist(), 'exact': exact.tolist(), 'fd': fd.tolist(),
        'cos_fo_exact': cosine(fo, exact), 'cos_exact_fd': cosine(exact, fd),
        'sign_agreement': (fo.sign() == exact.sign()).double().mean().item(),
        'norm_ratio': (fo.norm()/(exact.norm()+1e-12)).item(),
        'trajectory_max_abs_difference': (fo_trajectory-trajectory).abs().max().item(),
        'image_max_abs_difference': (fo_result.enhanced-exact_image).abs().max().item(),
        'exact_fd_max_abs_difference': (exact-fd).abs().max().item(),
        'saturation_rate': ((exact_image <= 1e-4) | (exact_image >= 1-1e-4)).double().mean().item(),
        'fo_trajectory': fo_trajectory.tolist(), 'exact_trajectory': trajectory.tolist(),
        'outer_loss': outer_loss(exact_image).item(),
        'all_gradients_finite': bool(torch.isfinite(torch.stack((fo, exact, fd))).all()),
    }


def summarize(rows):
    median = statistics.median(r['cos_fo_exact'] for r in rows)
    positive = sum(r['cos_fo_exact'] > 0 for r in rows)
    finite = all(r['all_gradients_finite'] for r in rows)
    reference_valid = all(r['cos_exact_fd'] >= .999 for r in rows) and finite
    fidelity_gate = median >= .5 and positive >= 9 and finite
    return {'episodes': len(rows), 'median_cos_fo_exact': median,
            'positive_fo_exact': positive, 'all_gradients_finite': finite,
            'minimum_cos_exact_fd': min(r['cos_exact_fd'] for r in rows),
            'reference_valid': reference_valid, 'fidelity_gate': fidelity_gate,
            'part_b_allowed': reference_valid and fidelity_gate}


def run_audit(output):
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    rows = [audit_episode(i) for i in range(12)]
    result = {'seed': 20260913, 'dtype': 'float64', 'device': 'cpu', 'threads': 1,
              'python': platform.python_version(), 'torch': torch.__version__,
              'finite_difference_epsilon': 1e-5, 'inner_steps': 3, 'inner_lr': .1,
              'norm_epsilon': 1e-12, 'summary': summarize(rows), 'episodes': rows}
    output.mkdir(parents=True, exist_ok=True)
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    fields = ['episode', 'cos_fo_exact', 'cos_exact_fd', 'sign_agreement', 'norm_ratio',
              'trajectory_max_abs_difference', 'image_max_abs_difference',
              'exact_fd_max_abs_difference', 'saturation_rate', 'all_gradients_finite']
    with (output/'audit.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.output)['summary'], indent=2))
