"""T018-A experimental full native loss with fixed teacher pseudo targets."""
import torch
from torch import nn

from .oracle import detector_task_loss

KEYS = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')


class NativePseudoTargetLoss(nn.Module):
    def __init__(self, detector, selected, *, seed=20260912):
        super().__init__()
        self.detector = detector
        self.boxes = selected['boxes'].detach().clone()
        self.labels = selected['labels'].detach().clone().long()
        self.seed = seed
        self.loss_history = []

    def forward(self, original, enhanced):
        if len(self.boxes):
            total, components = detector_task_loss(
                self.detector, enhanced, [{'boxes': self.boxes, 'labels': self.labels}], seed=self.seed)
            assert set(components) == set(KEYS)
            assert all(torch.isfinite(v).all() for v in components.values())
        else:
            total = enhanced.sum()*0
            components = {k: total for k in KEYS}
        self.loss_history.append({k: v.detach().item() for k, v in components.items()})
        return total
