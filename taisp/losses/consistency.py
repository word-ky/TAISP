import torch
from torch.nn import functional as F


def consistency_loss(features: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """MSE of aligned frozen-model features or differentiable predictions.

    The reference is fixed. A future detector adapter chooses its pre-NMS
    tensor representation; variable-length post-NMS boxes are not aligned.
    """
    return F.mse_loss(features, reference.detach())
