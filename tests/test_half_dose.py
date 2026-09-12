import ast
import inspect
import os

import pytest
import torch

from taisp import DifferentiableISP
from taisp.tta import trust_radius
from taisp.tta.trust_radius import adapt_clip_radius, adapt_half_dose, transfer_norm
from test_trust_radius import ToyLoss


def test_half_algebra_at_same_phi_and_complete_episode_diagnostics():
    isp = DifferentiableISP()
    x = torch.linspace(.1, .8, 3*9*11).reshape(1, 3, 9, 11)
    pseudo, clip = ToyLoss(.9), ToyLoss(.4)
    full = adapt_clip_radius(x, isp, pseudo, clip)
    half = adapt_half_dose(x, isp, pseudo, clip)
    torch.testing.assert_close(torch.tensor(half.diagnostics[1]['phi']), .5*torch.tensor(full.diagnostics[1]['phi']))
    assert len(half.diagnostics) == 4
    for k, d in enumerate(half.diagnostics):
        phi = torch.tensor(d['phi'], requires_grad=True)
        enhanced = isp(x, phi)
        gd = torch.autograd.grad(pseudo(x, enhanced), phi, retain_graph=True)[0]
        gc = torch.autograd.grad(clip(x, enhanced), phi)[0]
        h, _ = transfer_norm(gd, gc)
        torch.testing.assert_close(torch.tensor(d['gradient_per_coordinate']), .5*h)
        assert d['pre_attenuation_hybrid_norm'] == h.norm().item()
        assert d['applied_half_dose_norm'] == (.5*h).norm().item()
        assert d['detector_gradient_norm'] == gd.norm().item()
        assert d['clip_gradient_norm'] == gc.norm().item()
        assert d['dose_coefficient'] == .5 and d['step_seconds'] > 0
        assert 0 <= d['saturation_rate'] <= 1
        if k < 3:
            torch.testing.assert_close(torch.tensor(half.diagnostics[k+1]['phi']), phi-.1*(.5*h))


def test_half_freeze_reset_fixed_support_and_empty():
    isp = DifferentiableISP()
    x = torch.full((1, 3, 9, 11), .3)
    pseudo, clip = ToyLoss(.9), ToyLoss(.4)
    boxes = pseudo.boxes.clone()
    first = adapt_half_dose(x, isp, pseudo, clip)
    adapt_half_dose(x*.5, isp, pseudo, clip)
    again = adapt_half_dose(x, isp, pseudo, clip)
    torch.testing.assert_close(first.phi, again.phi, rtol=0, atol=0)
    assert torch.equal(isp.phi, torch.zeros(8)) and torch.equal(boxes, pseudo.boxes)
    for model in (pseudo, clip):
        assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    zero = adapt_half_dose(x, isp, ToyLoss(.9, empty=True), clip)
    assert torch.equal(zero.phi, torch.zeros(8))
    assert all(d['gradient_norm'] == d['pre_attenuation_hybrid_norm'] == 0 for d in zero.diagnostics)


def test_half_deployable_call_graph_has_no_targets_or_annotations():
    assert list(inspect.signature(adapt_half_dose).parameters) == ['image', 'isp', 'detector_loss', 'clip_loss', 'steps', 'lr', 'eps']
    tree = ast.parse(inspect.getsource(trust_radius))
    assert not any(isinstance(n, ast.Name) and n.id in {'annotations', 'targets', 'fcos', 'ssd', 'oracle'} for n in ast.walk(tree))
    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == ['import time', 'import torch', 'from .adapt import AdaptResult']


def test_reference_inputs_match_accepted_t009():
    pytest.importorskip('pycocotools')
    from taisp.analysis.run_t009 import verify_half_reference
    keys = ('seed', 'count', 'device', 'threads', 'semantic_lr', 'semantic_steps',
            'support_threshold', 'support_topk', 'radius_eps')
    cfg = dict(zip(keys, (20260912, 1000, 'cuda:0', 1, .1, 3, .5, 20, 1e-12)))
    reference = {k: k for k in ('clip_model', 'clip_revision', 'clip_sha256', 'detector_sha256',
        'positive_prompts', 'negative_prompts', 'target', 'ssd', 'annotation_sha256', 'subset_sha256', 'cases')}
    reference.update(config=cfg.copy(), evaluated_image_ids=[1, 2])
    cfg['variants'] = ['det_pseudo_half_dose']
    verify_half_reference(cfg, reference, reference, [1, 2])
    changed = dict(reference, clip_sha256='different')
    with pytest.raises(AssertionError, match='clip_sha256'):
        verify_half_reference(cfg, changed, reference, [1, 2])


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit real-model integration')
def test_real_half_source_clip_frozen_detached_support_and_reset():
    from taisp.models.detector import load_detector
    from taisp.losses.clip_semantic import load_clip_guidance
    from taisp.losses.detector_native import DetectorNativeLoss
    torch.set_num_threads(1)
    torch.manual_seed(7)
    source, clip = load_detector('cpu'), load_clip_guidance('cpu', local_files_only=True)
    support = {'base': {'boxes': torch.tensor([[20., 30., 170., 230.]], requires_grad=True),
                        'labels': torch.tensor([1]), 'scores': torch.tensor([.8], requires_grad=True)}}
    native = DetectorNativeLoss(source, support, 'det_pseudo')
    states = [{k: v.clone() for k, v in m.state_dict().items()} for m in (source, clip)]
    before = native.boxes.clone()
    assert not native.boxes.requires_grad and not native.weights.requires_grad
    x = .2+.5*torch.rand(1, 3, 257, 319)
    isp = DifferentiableISP()
    full = adapt_clip_radius(x, isp, native, clip, steps=1)
    half = adapt_half_dose(x, isp, native, clip, steps=1)
    again = adapt_half_dose(x, isp, native, clip, steps=1)
    torch.testing.assert_close(half.phi, .5*full.phi)
    torch.testing.assert_close(half.phi, again.phi, rtol=0, atol=0)
    assert half.phi.norm() > 0 and torch.equal(native.boxes, before)
    for state, model in zip(states, (source, clip)):
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        assert all(not m.training for m in model.modules())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
