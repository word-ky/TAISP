"""Label-free runtime: row pseudo gradient @ fixed source-trained orthogonal Q."""
import torch

from .trust_radius import _adapt_radius


@torch.enable_grad()
def adapt_gradient_transport(image, isp, detector_loss, clip_loss, q, *, steps=3, lr=.1, eps=1e-12):
    q = q.detach().to(image)
    torch.testing.assert_close(q.T @ q, torch.eye(8, device=q.device, dtype=q.dtype), rtol=0, atol=1e-5)
    return _adapt_radius(image, isp, detector_loss, clip_loss, steps=steps, lr=lr,
                         eps=eps, half_dose=False, gradient_transport=q)
