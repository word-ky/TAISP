"""R044 analysis-only fixed power-2 teacher target. No runtime replacement."""
import torch
from torch import nn
from torch.nn import functional as F
from taisp.models.detector_signal import fixed_roi_logits


def power2_target(logits):
    return F.softmax(2*F.log_softmax(logits.detach(),dim=-1),dim=-1).detach()


class SoftPseudoLoss(nn.Module):
    def __init__(self, hard, original):
        super().__init__()
        self.detector=hard.detector
        self.boxes=hard.boxes
        self.weights=hard.weights
        with torch.no_grad():
            self.original_logits=(fixed_roi_logits(self.detector,original,self.boxes)
                                  if len(self.boxes) else original.new_empty(0,91))
            assert self.original_logits.shape==(len(self.boxes),91)
            self.target=power2_target(self.original_logits)

    def forward(self, original, enhanced):
        if not len(self.boxes):return enhanced.sum()*0
        logits=fixed_roi_logits(self.detector,enhanced,self.boxes)
        return (self.weights*(-self.target*F.log_softmax(logits,dim=-1)).sum(-1)).sum()
