"""Fixed spatial weights and frozen patch-token restoration direction."""

import torch
from torch import nn
from torch.nn import functional as F


class PatchDirectionLoss(nn.Module):
    def __init__(self, encoder, direction, weights=None):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.register_buffer('direction', F.normalize(direction.detach().clone(), dim=-1))
        self.register_buffer('weights', (direction.new_full((49,), 1/49) if weights is None
                                         else weights.detach().clone() / weights.detach().sum()))

    def patch_scores(self, original, enhanced):
        """Unweighted per-patch negative displacement projections, B x 49."""
        self.encoder.eval()
        with torch.no_grad():
            reference = self.encoder.patch_features(original)
        return -((self.encoder.patch_features(enhanced)-reference)*self.direction).sum(-1)

    def forward(self, original, enhanced):
        return (self.patch_scores(original, enhanced)*self.weights).sum(-1).mean()
