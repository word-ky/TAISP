"""T013-A: first-order initialization plumbing; no outer training policy."""

import torch

from taisp.tta.trust_radius import _adapt_radius


@torch.enable_grad()
def adapt_from_initialization(image, isp, detector_loss, clip_loss, phi0, *,
                              steps=3, lr=.1, eps=1e-12):
    """Accepted full-hybrid forward with a FOMAML-style outer gradient.

    Supply explicit phi0 (8,) or a single-image ParameterPredictor output (1,8).
    The state graph reaches phi0, but the current-phi detector/CLIP update vector
    is stop-gradient. Thus d(phi_K)/d(phi0) is identity, not the exact unrolled
    Jacobian. No second-order model derivative or outer optimizer is constructed.
    Losses own the same detached original-image support as deployment. Their
    models are frozen/eval; isp.phi and phi0 are never mutated. Empty support
    preserves phi0 exactly. Outputs retain the outer graph, unlike deployment.
    """
    return _adapt_radius(image, isp, detector_loss, clip_loss, steps=steps, lr=lr,
                         eps=eps, half_dose=False, phi0=phi0)
