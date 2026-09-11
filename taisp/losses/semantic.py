"""Replaceable semantic guidance; no pretrained CLIP model in T001."""

import torch
from torch import nn
from torch.nn import functional as F


class MockImageEncoder(nn.Module):
    """Deterministic RGB means, solely for testing the interface."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(-2, -1))


class SemanticDirectionLoss(nn.Module):
    """Negative feature displacement projected onto a desired direction.

    L = -mean_b <E(enhanced)_b - stopgrad(E(original)_b), normalize(d)>.
    Projection has finite gradients at identity (zero displacement). This is
    a smoke-test surrogate, not a validated CLIP restoration objective.
    Encoder must return BxD differentiable features with its preprocessing.
    TODO: integrate CLIP text directions/prompts under a later research task.
    """

    def __init__(self, encoder: nn.Module, direction: torch.Tensor):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.register_buffer("direction", F.normalize(direction.detach().clone(), dim=-1))

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            reference = self.encoder(original)
        displacement = self.encoder(enhanced) - reference
        return -(displacement * self.direction).sum(dim=-1).mean()
