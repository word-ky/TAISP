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
    return _adapt_radius(image, isp, detector_loss, clip_loss, steps=steps, lr=lr, eps=eps, half_dose=False)


@torch.enable_grad()
def adapt_half_dose(image, isp, detector_loss, clip_loss, *, steps=3, lr=.1, eps=1e-12):
    """T012: the fixed 0.5-dose variant, with no target or annotation inputs."""
    return _adapt_radius(image, isp, detector_loss, clip_loss, steps=steps, lr=lr, eps=eps, half_dose=True)


def _adapt_radius(image, isp, detector_loss, clip_loss, *, steps, lr, eps, half_dose, phi0=None, gradient_transport=None):
    """Fresh episode; frozen losses; fixed original support owned by detector_loss."""
    detector_loss.eval().requires_grad_(False)
    clip_loss.eval().requires_grad_(False)
    initial = isp.phi.detach().clone() if phi0 is None else phi0.clone()
    phi = initial.clone().requires_grad_(True)
    history = []
    for step in range(steps+1):
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
        started = time.perf_counter()
        # Training-only FOMAML approximation: inner graphs end at a fresh probe.
        probe = phi if phi0 is None else phi.detach().requires_grad_(True)
        enhanced = isp(image, probe)
        pseudo, clip = detector_loss(image, enhanced), clip_loss(image, enhanced)
        gd = torch.autograd.grad(pseudo, probe, retain_graph=True)[0].detach()
        gc = torch.autograd.grad(clip, probe)[0].detach()
        direction = gd if gradient_transport is None else gd @ gradient_transport
        if gradient_transport is not None:
            torch.testing.assert_close(direction.norm(), gd.norm(), rtol=1e-5, atol=1e-8)
        update, scale = transfer_norm(direction, gc, eps)
        hybrid_norm = update.norm().item()
        if half_dose:
            update = .5*update
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
        if half_dose:
            history[-1].update(dose_coefficient=.5, pre_attenuation_hybrid_norm=hybrid_norm,
                               applied_half_dose_norm=update.norm().item())
        if gradient_transport is not None:
            history[-1].update(transported_gradient=direction.tolist(),
                               transported_gradient_norm=direction.norm().item(),
                               norm_preservation_passed=True)
        if step < steps:
            phi = phi-lr*update
            if phi0 is None:
                phi = phi.detach().requires_grad_(True)
    if phi0 is not None:
        # Only this small ISP graph is needed by an outer objective.
        return AdaptResult(initial, phi, isp(image, phi), None, history)
    return AdaptResult(initial, phi.detach(), enhanced.detach(), None, history)
