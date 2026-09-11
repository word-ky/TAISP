import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from taisp import DifferentiableISP
from taisp.losses.conditioned_clip import CLIPConditioner, load_conditioner
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.tta import AdaptConfig, adapt


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        return F.normalize(x.mean((-2, -1)) * self.scale, dim=-1)


def test_condition_equations_detach_and_episode_reset():
    encoder = TinyEncoder()
    condition = CLIPConditioner(encoder, torch.tensor([1., 1., 1.]), torch.eye(3), .05)
    x = torch.tensor([.2, .3, .4]).reshape(1, 3, 1, 1).expand(1, 3, 6, 7).requires_grad_()
    episode = condition.prepare(x)
    expected = torch.softmax(F.normalize(torch.tensor([[.2, .3, .4]]), dim=-1) / .05, -1)
    torch.testing.assert_close(episode.weights, expected)
    torch.testing.assert_close(episode.gate, expected @ condition.masks)
    torch.testing.assert_close(episode.guidance.direction,
                               F.normalize(condition.positive - expected @ condition.negatives, dim=-1))
    assert all(not v.requires_grad for v in (episode.weights, episode.gate, episode.guidance.direction))
    isp = DifferentiableISP()
    cfg = AdaptConfig(steps=2, regularization_weight=0)
    def run(image):
        e = condition.prepare(image)
        return adapt(image, isp, e.guidance, config=cfg, coordinate_gate=e.gate)
    first = run(x.detach())
    run(x.detach().flip(1))
    repeat = run(x.detach())
    torch.testing.assert_close(first.phi, repeat.phi, atol=0, rtol=0)
    assert first.phi.norm() > 0 and first.phi[7] == 0
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    torch.testing.assert_close(isp.phi, torch.zeros(8), atol=0, rtol=0)


def test_gate_is_exact_masked_sgd_and_has_no_gradient():
    x = torch.linspace(.2, .6, 3*6*7).reshape(1, 3, 6, 7)
    isp = DifferentiableISP()
    gate = torch.tensor([.2, 0, .5, 0, 1., 0, .2, 0], requires_grad=True)
    objective = lambda original, enhanced: (enhanced - .7).square().mean()
    cfg = AdaptConfig(steps=1, lr=.1, regularization_weight=0)
    ungated = adapt(x, isp, objective, config=cfg)
    gated = adapt(x, isp, objective, config=cfg, coordinate_gate=gate)
    torch.testing.assert_close(gated.phi, ungated.phi * gate.detach())
    assert torch.equal(gated.phi[gate == 0], torch.zeros(4))
    assert gate.grad is None
    torch.testing.assert_close(torch.tensor(gated.diagnostics[0]['gradient_per_coordinate']),
                               gate.detach() * torch.tensor(gated.diagnostics[0]['raw_gradient_per_coordinate']))


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit pretrained-model integration run')
def test_real_conditioner_frozen_gradients_and_reset():
    torch.manual_seed(17)
    generic = load_clip_guidance('cuda:0', local_files_only=True)
    condition = load_conditioner(generic, local_files_only=True)
    before = {k: v.clone() for k, v in condition.state_dict().items()}
    x = .2 + .5 * torch.rand(1, 3, 240, 320, device='cuda:0')
    isp = DifferentiableISP().cuda()
    cfg = AdaptConfig(steps=1, regularization_weight=0)
    def run(image):
        e = condition.prepare(image)
        assert not e.weights.requires_grad and not e.gate.requires_grad
        return adapt(image, isp, e.guidance, config=cfg, coordinate_gate=e.gate)
    a = run(x)
    run(x * .8)
    b = run(x)
    torch.testing.assert_close(a.phi0, torch.zeros_like(a.phi0), atol=0, rtol=0)
    torch.testing.assert_close(a.phi, b.phi, atol=1e-8, rtol=1e-5)
    assert torch.isfinite(a.phi).all() and a.phi.norm() > 0 and a.phi[7] == 0
    for k, v in condition.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in condition.parameters())
