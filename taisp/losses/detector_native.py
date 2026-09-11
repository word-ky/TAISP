"""Label-free detector confidence and fixed-support view consistency."""
import torch
from torch import nn
from torch.nn import functional as F

from taisp.models.detector_signal import fixed_roi_logits


def confidence_consistency(base_logits, labels, weights, flip_logits=None, js_coefficient=0.):
    ce = F.cross_entropy(base_logits, labels, reduction='none')
    js = torch.zeros_like(ce)
    if flip_logits is not None:
        ce = (ce+F.cross_entropy(flip_logits, labels, reduction='none'))/2
        log_p, log_q = F.log_softmax(base_logits, -1), F.log_softmax(flip_logits, -1)
        log_m = torch.logaddexp(log_p, log_q)-torch.log(base_logits.new_tensor(2.))
        js = (log_p.exp()*(log_p-log_m)+log_q.exp()*(log_q-log_m)).sum(-1)/2
    return (weights*ce).sum(), (weights*js).sum()


class DetectorNativeLoss(nn.Module):
    def __init__(self, detector, support, variant, js_coefficient=1.):
        super().__init__()
        self.detector = detector
        self.variant = variant
        self.js_coefficient = js_coefficient if variant == 'det_stable_js' else 0.
        chosen = support['base'] if variant == 'det_pseudo' else support['stable']
        self.boxes = chosen['boxes'].detach().clone()
        self.labels = chosen['labels'].detach().clone()
        scores = chosen['scores'].detach().clone()
        self.weights = scores/scores.sum() if len(scores) else scores
        self.flip_boxes = chosen['flip_boxes'].detach().clone() if variant != 'det_pseudo' else None

    def components(self, enhanced):
        if not len(self.boxes):
            zero = enhanced.sum()*0
            return zero, zero
        base = fixed_roi_logits(self.detector, enhanced, self.boxes)
        flipped = fixed_roi_logits(self.detector, enhanced.flip(-1), self.flip_boxes) if self.flip_boxes is not None else None
        return confidence_consistency(base, self.labels, self.weights, flipped, self.js_coefficient)

    def forward(self, original, enhanced):
        ce, js = self.components(enhanced)
        return ce+self.js_coefficient*js
