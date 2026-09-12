import inspect
import os

import pytest
import torch
from torch import nn

from taisp import DifferentiableISP
from taisp.tta.trust_radius import adapt_clip_radius, transfer_norm


def test_norm_transfer_algebra_and_zero():
    gd = torch.tensor([1., -2., 3.], dtype=torch.double)
    gc = torch.tensor([-4., 5., 6.], dtype=torch.double)
    update, scale = transfer_norm(gd, gc)
    torch.testing.assert_close(update.norm(), gc.norm()*gd.norm()/(gd.norm()+1e-12))
    torch.testing.assert_close(update/update.norm(), gd/gd.norm())
    assert scale > 0
    update, scale = transfer_norm(torch.zeros_like(gd), gc)
    assert torch.equal(update, torch.zeros_like(gd)) and scale == 0
    update, scale = transfer_norm(gd, torch.zeros_like(gc))
    assert torch.equal(update, torch.zeros_like(gd)) and scale == 0
    assert list(inspect.signature(adapt_clip_radius).parameters) == ['image', 'isp', 'detector_loss', 'clip_loss', 'steps', 'lr', 'eps']


class ToyLoss(nn.Module):
    def __init__(self, value, empty=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))
        self.boxes = torch.empty(0, 4) if empty else torch.ones(1, 4)
        self.empty = empty

    def forward(self, original, enhanced):
        return enhanced.sum()*0 if self.empty else (enhanced-self.weight).square().mean()


def test_episode_reset_freeze_fixed_support_current_phi_and_empty():
    isp = DifferentiableISP()
    x = torch.linspace(.1, .8, 3*9*11).reshape(1, 3, 9, 11)
    pseudo, clip = ToyLoss(.9), ToyLoss(.4)
    boxes = pseudo.boxes.clone()
    first = adapt_clip_radius(x, isp, pseudo, clip)
    adapt_clip_radius(x*.5, isp, pseudo, clip)
    again = adapt_clip_radius(x, isp, pseudo, clip)
    torch.testing.assert_close(first.phi, again.phi, rtol=0, atol=0)
    assert torch.equal(isp.phi, torch.zeros(8)) and torch.equal(boxes, pseudo.boxes)
    for model in [pseudo, clip]:
        assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    for i, d in enumerate(first.diagnostics):
        p = torch.tensor(d['phi'], requires_grad=True)
        y = isp(x, p)
        gd = torch.autograd.grad(pseudo(x, y), p, retain_graph=True)[0]
        gc = torch.autograd.grad(clip(x, y), p)[0]
        update, scale = transfer_norm(gd, gc)
        torch.testing.assert_close(torch.tensor(d['gradient_per_coordinate']), update)
        assert d['support_count'] == 1 and d['step_seconds'] > 0
        if i < 3:
            torch.testing.assert_close(torch.tensor(first.diagnostics[i+1]['phi']), p-.1*update)
    zero = adapt_clip_radius(x, isp, ToyLoss(.9, empty=True), clip)
    assert torch.equal(zero.phi, torch.zeros(8))
    assert all(d['gradient_norm'] == 0 and d['scale_factor'] == 0 for d in zero.diagnostics)


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit real-model integration')
def test_real_source_clip_frozen_and_support_unchanged():
    from taisp.models.detector import load_detector
    from taisp.losses.clip_semantic import load_clip_guidance
    from taisp.losses.detector_native import DetectorNativeLoss
    torch.set_num_threads(1)
    torch.manual_seed(7)
    source, clip = load_detector('cpu'), load_clip_guidance('cpu', local_files_only=True)
    support = {'base': {'boxes': torch.tensor([[20., 30., 170., 230.]]),
                        'labels': torch.tensor([1]), 'scores': torch.tensor([.8])}}
    native = DetectorNativeLoss(source, support, 'det_pseudo')
    states = [{k: v.clone() for k, v in m.state_dict().items()} for m in [source, clip]]
    before = {k: v.clone() for k, v in support['base'].items()}
    x = .2+.5*torch.rand(1, 3, 257, 319)
    isp = DifferentiableISP()
    a = adapt_clip_radius(x, isp, native, clip, steps=1)
    b = adapt_clip_radius(x, isp, native, clip, steps=1)
    torch.testing.assert_close(a.phi, b.phi, rtol=0, atol=0)
    assert a.phi.norm() > 0 and torch.isfinite(a.phi).all()
    for state, model in zip(states, [source, clip]):
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        assert all(not m.training for m in model.modules())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in support['base'].items())
    assert torch.equal(native.boxes, before['boxes'])
