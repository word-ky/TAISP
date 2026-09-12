"""T009 evaluation-only SSD300; no annotated or differentiable loss adapter."""
import hashlib
from pathlib import Path

import torch

from taisp.models.detector import FrozenDetector


def load_ssd_target(device='cpu'):
    from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
    model = ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1, weights_backbone=None,
        score_thresh=.01, nms_thresh=.45, detections_per_img=200, topk_candidates=400,
        image_mean=[.48235, .45882, .40784], image_std=[1/255, 1/255, 1/255],
    )
    return FrozenDetector(model).to(device).eval()


def ssd_metadata(target):
    import torchvision
    from torchvision.models.detection import SSD300_VGG16_Weights
    weights, model = SSD300_VGG16_Weights.COCO_V1, target.model
    path = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1]
    return {'torchvision': torchvision.__version__, 'weight_enum': 'SSD300_VGG16_Weights.COCO_V1',
            'url': weights.url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'categories': weights.meta['categories'], 'class_mapping': 'COCO category id equals prediction label; 91 slots including background/N/A',
            'score_thresh': model.score_thresh, 'nms_thresh': model.nms_thresh,
            'detections_per_img': model.detections_per_img, 'topk_candidates': model.topk_candidates,
            'transform': {'min_size': list(model.transform.min_size), 'max_size': model.transform.max_size,
                          'fixed_size': list(model.transform.fixed_size), 'size_divisible': model.transform.size_divisible,
                          'image_mean': model.transform.image_mean, 'image_std': model.transform.image_std}}
