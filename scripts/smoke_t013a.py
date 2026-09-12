"""One synthetic-image, frozen real-model smoke; no dataset or training."""
import argparse
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
from taisp.training.initialization import adapt_from_initialization
from taisp.tta.trust_radius import adapt_clip_radius


def run_smoke():
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    device = 'cuda:0'
    source = load_detector(device)
    clip = load_clip_guidance(device, local_files_only=True)
    support = {'base': {
        'boxes': torch.tensor([[20., 30., 110., 120.]], device=device, requires_grad=True),
        'labels': torch.tensor([1], device=device),
        'scores': torch.tensor([.8], device=device, requires_grad=True)}}
    native = DetectorNativeLoss(source, support, 'det_pseudo')
    assert not native.boxes.requires_grad and not native.weights.requires_grad
    states = [{k: v.clone() for k, v in m.state_dict().items()} for m in (source, clip)]
    x = .2+.5*torch.rand(1, 3, 129, 161, device=device)
    isp = DifferentiableISP().to(device)
    predictor = ParameterPredictor().to(device)
    initial = predictor(x)
    initial.retain_grad()
    full = adapt_clip_radius(x, isp, native, clip)
    connected = adapt_from_initialization(x, isp, native, clip, initial)
    phi_error = (connected.phi.flatten()-full.phi).abs().max().item()
    image_error = (connected.enhanced-full.enhanced).abs().max().item()
    torch.testing.assert_close(connected.phi.flatten(), full.phi, rtol=0, atol=0)
    torch.testing.assert_close(connected.enhanced, full.enhanced, rtol=0, atol=0)
    ((connected.enhanced-.6)**2).mean().backward()
    assert torch.isfinite(initial.grad).all() and initial.grad.norm() > 0
    assert torch.isfinite(predictor.head.weight.grad).all() and predictor.head.weight.grad.norm() > 0
    for model, state in zip((source, clip), states):
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        assert all(not m.training for m in model.modules())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert torch.equal(native.boxes, support['base']['boxes'])
    assert support['base']['boxes'].grad is None and support['base']['scores'].grad is None
    assert isp.phi.grad is None and torch.count_nonzero(isp.phi) == 0
    return {
        'purpose': 'Real frozen-model autograd smoke on one synthetic image; no detection evidence.',
        'source_revision': os.environ.get('TAISP_SOURCE_REVISION'), 'seed': 20260913,
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': device, 'gpu': torch.cuda.get_device_name(0),
        'image_shape': list(x.shape), 'support': 'one fixed synthetic box/label/score; detached',
        'steps': 3, 'lr': .1, 'eps': 1e-12, 'approximation': 'FOMAML-style',
        'phi_max_abs_error': phi_error, 'image_max_abs_error': image_error,
        'phi0_gradient_norm': initial.grad.norm().item(),
        'predictor_head_gradient_norm': predictor.head.weight.grad.norm().item(),
        'frozen_parameters_and_buffers_unchanged': True, 'support_unchanged': True,
        'phi0': initial.detach().cpu().tolist(), 'phi3': connected.phi.detach().cpu().tolist(),
        'elapsed_seconds': time.perf_counter()-started,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
