"""T007: source pseudo direction with current-phi CLIP gradient magnitude."""
import time

import torch

from .adapt import AdaptResult


def transfer_norm(detector_gradient, clip_gradient, eps=1e-12):
    detector_norm, clip_norm = detector_gradient.norm(), clip_gradient.norm()
    scale = clip_norm/(detector_norm+eps) if detector_norm.item() else detector_norm.new_zeros(())
    return detector_gradient*scale, scale


@torch.enable_grad()
def adapt_clip_radius(image, isp, detector_loss, clip_loss, *, steps=3, lr=.1, eps=1e-12):
    """Fresh episode; frozen losses; fixed original support owned by detector_loss."""
    detector_loss.eval().requires_grad_(False)
    clip_loss.eval().requires_grad_(False)
    initial = isp.phi.detach().clone()
    phi = initial.clone().requires_grad_(True)
    history = []
    for step in range(steps+1):
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
        started = time.perf_counter()
        enhanced = isp(image, phi)
        pseudo, clip = detector_loss(image, enhanced), clip_loss(image, enhanced)
        gd = torch.autograd.grad(pseudo, phi, retain_graph=True)[0].detach()
        gc = torch.autograd.grad(clip, phi)[0].detach()
        update, scale = transfer_norm(gd, gc, eps)
        nd, nc = gd.norm().item(), gc.norm().item()
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
        history.append({
            'step': step, 'total': pseudo.item(), 'semantic': pseudo.item(),
            'clip_loss': clip.item(), 'consistency': 0., 'regularization': 0.,
            'phi': phi.detach().tolist(),
            'physical': {k: v.detach().tolist() for k, v in isp.decode(phi).items()},
            'saturation_rate': ((enhanced <= 1e-4) | (enhanced >= 1-1e-4)).float().mean().item(),
            'detector_gradient': gd.tolist(), 'clip_gradient': gc.tolist(),
            'detector_gradient_norm': nd, 'clip_gradient_norm': nc,
            'detector_clip_ratio': nd/nc if nc else None,
            'scale_factor': scale.item(), 'gradient_norm': update.norm().item(),
            'gradient_per_coordinate': update.tolist(),
            'gradient_abs_per_coordinate': update.abs().tolist(),
            'support_count': len(detector_loss.boxes),
            'step_seconds': time.perf_counter()-started,
            'update_applied': step < steps,
        })
        if step < steps:
            phi = (phi-lr*update).detach().requires_grad_(True)
    return AdaptResult(initial, phi.detach(), enhanced.detach(), None, history)
