"""Deployment-only episodic phi optimization; no label input or model optimizer."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from taisp.isp.module import DifferentiableISP
from taisp.losses import consistency_loss, state_regularization


@dataclass
class AdaptConfig:
    steps: int = 10
    lr: float = 0.1
    semantic_weight: float = 1.0
    consistency_weight: float = 0.0
    regularization_weight: float = 0.01
    identity_weight: float = 0.0


@dataclass
class AdaptResult:
    phi0: torch.Tensor
    phi: torch.Tensor
    enhanced: torch.Tensor
    prediction: torch.Tensor | None
    diagnostics: list[dict[str, float | int]]


@torch.enable_grad()
def adapt(
    image: torch.Tensor,
    isp: DifferentiableISP,
    semantic_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    downstream: nn.Module | None = None,
    predictor: nn.Module | None = None,
    phi0: torch.Tensor | None = None,
    config: AdaptConfig | None = None,
    differentiable: bool = False,
) -> AdaptResult:
    """Optimize a fresh phi for one image, using functional SGD.

    downstream must return aligned differentiable Bx... features/predictions;
    it is permanently frozen and set to eval, including its buffers. The final
    prediction uses the final enhanced image. Modules must share image's device.
    Pass either an explicit phi0 or a predictor, otherwise use isp.phi.

    Deployment detaches initialization and outputs. differentiable=True retains
    the update graph to phi0 (including an optional predictor) for future outer
    training; it performs no outer update. Only phi changes in the inner loop.
    """
    if image.shape[0] != 1:
        raise ValueError("adapt expects one image per episode (batch size 1)")
    config = config or AdaptConfig()
    if downstream is None and config.consistency_weight != 0:
        raise ValueError("nonzero consistency weight requires a downstream model")
    if downstream is not None:
        downstream.eval().requires_grad_(False)
    if predictor is not None:
        predictor.eval()
    if phi0 is None:
        phi0 = predictor(image) if predictor is not None else isp.phi
    initial = phi0.clone() if differentiable else phi0.detach().clone()
    initial.requires_grad_(True)
    phi = initial
    with torch.no_grad():
        reference = downstream(image) if downstream is not None else None

    history = []
    for step in range(config.steps + 1):
        enhanced = isp(image, phi)
        prediction = downstream(enhanced) if downstream is not None else None
        semantic = semantic_loss(image, enhanced)
        consistency = (
            consistency_loss(prediction, reference)
            if prediction is not None else phi.new_zeros(())
        )
        regularization = state_regularization(phi, initial, config.identity_weight)
        total = (
            config.semantic_weight * semantic
            + config.consistency_weight * consistency
            + config.regularization_weight * regularization
        )
        history.append({
            "step": step, "total": total.detach().item(),
            "semantic": semantic.detach().item(),
            "consistency": consistency.detach().item(),
            "regularization": regularization.detach().item(),
        })
        if step == config.steps:
            break
        gradient = torch.autograd.grad(total, phi, create_graph=differentiable)[0]
        history[-1]["gradient_norm"] = gradient.detach().norm().item()
        phi = phi - config.lr * gradient
        if not differentiable:
            phi = phi.detach().requires_grad_(True)

    if not differentiable:
        initial, phi, enhanced = initial.detach(), phi.detach(), enhanced.detach()
        prediction = prediction.detach() if prediction is not None else None
    return AdaptResult(initial, phi, enhanced, prediction, history)
