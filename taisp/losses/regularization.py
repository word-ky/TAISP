import torch


def state_regularization(
    phi: torch.Tensor, phi0: torch.Tensor, identity_weight: float = 0.0
) -> torch.Tensor:
    """Squared L2 per state, averaged over episodes; zero is raw identity."""
    proximity = (phi - phi0).square().sum(dim=-1).mean()
    identity = phi.square().sum(dim=-1).mean()
    return proximity + identity_weight * identity
