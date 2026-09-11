"""Analysis-only annotated loss of a frozen torchvision detector."""

import torch


def detector_task_loss(detector, image, targets, *, seed=20260912):
    """Native RPN+ROI classification/regression loss with fixed sampling RNG.

    Only branches needed to return native training losses use training flags.
    Backbone/transform/submodules remain eval and all weights remain frozen.
    This routine is never reachable by an import/call inside adapt.py.
    """
    model = detector.model
    model.eval()
    devices = [image.device.index or 0] if image.is_cuda else []
    try:
        model.training = model.rpn.training = model.roi_heads.training = True
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            components = model(list(image), targets)
        return sum(components.values()), components
    finally:
        model.eval()
