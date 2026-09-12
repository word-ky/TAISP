"""Pinned independent target; never imported by deployment objectives."""
import hashlib
from pathlib import Path

import torch

from taisp.models.detector import FrozenDetector


def load_fcos_target(device='cpu'):
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
    model = fcos_resnet50_fpn(
        weights=FCOS_ResNet50_FPN_Weights.COCO_V1, weights_backbone=None,
        min_size=800, max_size=1333, image_mean=[.485, .456, .406], image_std=[.229, .224, .225],
        score_thresh=.2, nms_thresh=.6, detections_per_img=100, topk_candidates=1000,
        center_sampling_radius=1.5,
    )
    return FrozenDetector(model).to(device).eval()


def target_metadata(target):
    import torchvision
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
    model, weights = target.model, FCOS_ResNet50_FPN_Weights.COCO_V1
    path = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1]
    return {'torchvision': torchvision.__version__, 'weight_enum': 'FCOS_ResNet50_FPN_Weights.COCO_V1',
            'url': weights.url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'score_thresh': model.score_thresh, 'nms_thresh': model.nms_thresh,
            'detections_per_img': model.detections_per_img, 'topk_candidates': model.topk_candidates,
            'center_sampling_radius': model.center_sampling_radius,
            'transform': {'min_size': list(model.transform.min_size), 'max_size': model.transform.max_size,
                          'image_mean': model.transform.image_mean, 'image_std': model.transform.image_std,
                          'size_divisible': model.transform.size_divisible}}


def target_task_loss(target, image, annotations):
    """Native FCOS annotated losses; only top-level loss-return branch is enabled."""
    model = target.model
    model.eval()
    try:
        model.training = True
        components = model(list(image), annotations)
        return sum(components.values()), components
    finally:
        model.eval()
