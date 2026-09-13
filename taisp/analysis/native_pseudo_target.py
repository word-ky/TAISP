"""T018-A experimental full native loss with fixed teacher pseudo targets."""
import torch
from torch import nn

from .oracle import detector_task_loss

KEYS = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
COMPONENT_SETS = {
    'full': KEYS,
    'native_cls': ('loss_classifier',),
    'native_conf': ('loss_classifier', 'loss_objectness'),
    'native_roi': ('loss_classifier', 'loss_box_reg'),
    'native_conf_roi': ('loss_classifier', 'loss_objectness', 'loss_box_reg'),
}


class NativePseudoTargetLoss(nn.Module):
    def __init__(self, detector, selected, *, seed=20260912, component_set='full'):
        super().__init__()
        self.detector = detector
        self.boxes = selected['boxes'].detach().clone()
        self.labels = selected['labels'].detach().clone().long()
        self.seed = seed
        self.active_keys = COMPONENT_SETS[component_set]
        self.component_set = component_set
        self.loss_history = []

    def forward(self, original, enhanced):
        if len(self.boxes):
            total, components = detector_task_loss(
                self.detector, enhanced, [{'boxes': self.boxes, 'labels': self.labels}], seed=self.seed)
            assert set(components) == set(KEYS)
            assert all(torch.isfinite(v).all() for v in components.values())
            if self.component_set != 'full':
                total = sum(components[k] for k in self.active_keys)
        else:
            total = enhanced.sum()*0
            components = {k: total for k in KEYS}
        self.loss_history.append({k: components[k].detach().item() for k in self.active_keys})
        return total
