import inspect

import torch
from torch import nn

from taisp import DifferentiableISP
from taisp.models import ParameterPredictor
from taisp.tta import AdaptConfig, adapt


def synthetic_loss(original, enhanced):
    # Fixed scalar toy objective; no labels/data annotations involved.
    return (enhanced - 0.55).square().mean()


def image():
    return torch.linspace(0.15, 0.35, 3 * 6 * 7).reshape(1, 3, 6, 7)


def test_loss_decreases_and_episodes_reset():
    isp = DifferentiableISP()
    x = image()
    config = AdaptConfig(steps=15, lr=0.5)
    first = adapt(x, isp, synthetic_loss, config=config)
    adapt(x * 0.8, isp, synthetic_loss, config=config)
    again = adapt(x, isp, synthetic_loss, config=config)
    assert first.diagnostics[-1]["total"] < first.diagnostics[0]["total"] * 0.25
    torch.testing.assert_close(first.phi, again.phi)
    torch.testing.assert_close(isp.phi, torch.zeros(8))
    assert not first.phi.requires_grad and not first.enhanced.requires_grad
    assert isp.phi.grad is None
    assert len(first.diagnostics) == 16


def test_downstream_and_predictor_state_unchanged_and_final_prediction():
    torch.manual_seed(3)
    downstream = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout(0.5))
    predictor = ParameterPredictor()
    before = {k: v.clone() for k, v in downstream.state_dict().items()}
    predictor_before = {k: v.clone() for k, v in predictor.state_dict().items()}
    result = adapt(image(), DifferentiableISP(), synthetic_loss, downstream=downstream,
                   predictor=predictor, config=AdaptConfig(steps=3, consistency_weight=0.1))
    for key, value in downstream.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    for key, value in predictor.state_dict().items():
        torch.testing.assert_close(value, predictor_before[key], atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in downstream.parameters())
    assert all(p.grad is None for p in predictor.parameters())
    assert not downstream.training
    torch.testing.assert_close(result.prediction, downstream(result.enhanced))
    assert result.diagnostics[-1]["consistency"] > 0


def test_consistency_gradient_reaches_phi_through_frozen_model():
    model = nn.Conv2d(3, 3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
    phi0 = torch.full((8,), 0.05)
    result = adapt(image(), DifferentiableISP(), synthetic_loss, downstream=model,
                   phi0=phi0, config=AdaptConfig(steps=1, lr=0.1,
                       semantic_weight=0, consistency_weight=1, regularization_weight=0))
    assert result.diagnostics[-1]["consistency"] < result.diagnostics[0]["consistency"]
    assert not torch.equal(result.phi, phi0)


def test_unrolled_gradient_matches_finite_difference():
    isp = DifferentiableISP().double()
    x = image().double()
    config = AdaptConfig(steps=2, lr=0.2, regularization_weight=0.1)
    phi0 = torch.full((8,), 0.01, dtype=torch.double, requires_grad=True)

    def unroll(p):
        return adapt(x, isp, synthetic_loss, phi0=p, config=config,
                     differentiable=True).enhanced

    assert torch.autograd.gradcheck(unroll, (phi0,))
    grad = torch.autograd.grad(unroll(phi0).square().mean(), phi0)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0


def test_zero_steps_and_outer_no_grad():
    with torch.no_grad():
        result = adapt(image(), DifferentiableISP(), synthetic_loss,
                       config=AdaptConfig(steps=0))
    torch.testing.assert_close(result.enhanced, image())
    assert len(result.diagnostics) == 1
    assert all("label" not in name for name in inspect.signature(adapt).parameters)
