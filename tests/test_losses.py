import torch
from torch import nn

from taisp.losses import SemanticDirectionLoss, consistency_loss, state_regularization
from taisp.models import ParameterPredictor


def test_semantic_direction_and_frozen_encoder():
    encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        encoder[-1].weight.copy_(torch.eye(3))
    loss = SemanticDirectionLoss(encoder, torch.tensor([1.0, 0.0, 0.0]))
    x = torch.full((1, 3, 4, 4), 0.2)
    y = torch.full_like(x, 0.3, requires_grad=True)
    torch.testing.assert_close(loss(x, y), torch.tensor(-0.1))
    loss(x, y).backward()
    assert y.grad[:, 0].abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())


def test_consistency_reference_is_detached():
    features = torch.tensor([1.0, 3.0], requires_grad=True)
    reference = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = consistency_loss(features, reference)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    assert reference.grad is None
    torch.testing.assert_close(features.grad, torch.tensor([1.0, 2.0]))


def test_regularization_exact_squared_norm():
    phi = torch.tensor([1.0, 2.0], requires_grad=True)
    phi0 = torch.tensor([1.0, 1.0])
    loss = state_regularization(phi, phi0, identity_weight=0.1)
    torch.testing.assert_close(loss, torch.tensor(1.5))
    loss.backward()
    torch.testing.assert_close(phi.grad, torch.tensor([0.2, 2.4]))


def test_predictor_starts_at_identity_and_can_train():
    predictor = ParameterPredictor()
    phi = predictor(torch.rand(2, 3, 8, 8))
    torch.testing.assert_close(phi, torch.zeros(2, 8))
    phi.sum().backward()
    assert predictor.head.weight.grad.abs().sum() > 0
