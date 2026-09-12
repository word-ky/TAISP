import inspect

import pytest
import torch

from taisp import DifferentiableISP
from taisp.models.parameter_predictor import ParameterPredictor
from taisp.training.initialization import adapt_from_initialization
from taisp.tta.trust_radius import adapt_clip_radius
from test_trust_radius import ToyLoss


def fixture():
    torch.manual_seed(20260913)
    x = torch.linspace(.1, .8, 3*9*11).reshape(1, 3, 9, 11)
    return x, DifferentiableISP(), ToyLoss(.9), ToyLoss(.4)


@pytest.mark.parametrize('offset', [0., .04])
def test_forward_parity_entire_trajectory_and_deployment_detachment(offset):
    x, isp, det, clip = fixture()
    with torch.no_grad():
        isp.phi.copy_(torch.linspace(-offset, offset, 8))
    initial = isp.phi.detach().clone().requires_grad_(True)
    deployed = adapt_clip_radius(x, isp, det, clip)
    trained = adapt_from_initialization(x, isp, det, clip, initial)
    torch.testing.assert_close(trained.phi, deployed.phi, rtol=0, atol=0)
    torch.testing.assert_close(trained.enhanced, deployed.enhanced, rtol=0, atol=0)
    for a, b in zip(trained.diagnostics, deployed.diagnostics):
        assert {k: v for k, v in a.items() if k != 'step_seconds'} == {
            k: v for k, v in b.items() if k != 'step_seconds'}
    assert all(not t.requires_grad for t in (deployed.phi0, deployed.phi, deployed.enhanced))
    assert all(t.requires_grad for t in (trained.phi0, trained.phi, trained.enhanced))
    assert list(inspect.signature(adapt_clip_radius).parameters) == [
        'image', 'isp', 'detector_loss', 'clip_loss', 'steps', 'lr', 'eps']
    assert torch.equal(initial, isp.phi)


def test_initialization_sensitivity_reset_and_empty_support():
    x, isp, det, clip = fixture()
    a = torch.zeros(8, requires_grad=True)
    b = torch.linspace(-.03, .05, 8, requires_grad=True)
    first = adapt_from_initialization(x, isp, det, clip, a)
    second = adapt_from_initialization(x*.8, isp, det, clip, b)
    again = adapt_from_initialization(x, isp, det, clip, a)
    assert first.diagnostics[0]['phi'] == a.tolist()
    assert second.diagnostics[0]['phi'] == b.tolist()
    assert not torch.equal(first.phi, second.phi)
    torch.testing.assert_close(first.phi, again.phi, rtol=0, atol=0)
    empty = adapt_from_initialization(x, isp, ToyLoss(.9, empty=True), clip, b)
    assert torch.equal(empty.phi, b)
    torch.testing.assert_close(empty.enhanced, isp(x, b), rtol=0, atol=0)
    assert all(d['gradient_norm'] == 0 for d in empty.diagnostics)
    assert torch.equal(torch.autograd.grad(empty.phi.sum(), b)[0], torch.ones_like(b))
    assert torch.equal(isp.phi, torch.zeros(8)) and torch.equal(a, torch.zeros(8))
    assert torch.equal(b, torch.linspace(-.03, .05, 8))


def test_outer_gradient_and_identity_state_jacobian_without_model_hessians(monkeypatch):
    x, isp, det, clip = fixture()
    initial = torch.zeros(8, requires_grad=True)
    original_grad = torch.autograd.grad
    calls = []

    def first_order_only(*args, **kwargs):
        assert not kwargs.get('create_graph', False)
        calls.append(args[1])
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, 'grad', first_order_only)
    result = adapt_from_initialization(x, isp, det, clip, initial)
    assert len(calls) == 8  # two gradients at each diagnostic state 0..3
    assert all(t.is_leaf and t is not initial for t in calls)
    jacobian = torch.stack([original_grad(result.phi[i], initial, retain_graph=True)[0]
                            for i in range(8)])
    torch.testing.assert_close(jacobian, torch.eye(8), rtol=0, atol=0)
    outer = (result.enhanced-.6).square().mean()
    gradient = original_grad(outer, initial)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    for model in (det, clip):
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize('nonzero_head', [False, True])
def test_predictor_head_and_feature_trunk_outer_gradient(nonzero_head):
    x, isp, det, clip = fixture()
    predictor = ParameterPredictor()
    if nonzero_head:
        with torch.no_grad():
            predictor.head.weight.fill_(.01)
    initial = predictor(x)
    result = adapt_from_initialization(x, isp, det, clip, initial)
    (result.enhanced-.6).square().mean().backward()
    for p in predictor.head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    trunk = predictor.features[0].weight.grad
    assert torch.isfinite(trunk).all()
    assert trunk.norm() > 0 if nonzero_head else trunk.norm() == 0
    assert isp.phi.grad is None
    assert all(p.grad is None for model in (det, clip) for p in model.parameters())
