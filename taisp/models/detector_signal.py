"""Frozen fixed-ROI signal and original-only stable object support."""
from collections import OrderedDict

import torch


def flip_boxes(boxes, width):
    return torch.stack((width-boxes[:, 2], boxes[:, 1],
                        width-boxes[:, 0], boxes[:, 3]), dim=-1)


def select_predictions(prediction, threshold=0.5, topk=20):
    indices = torch.where((prediction['scores'] >= threshold) & (prediction['labels'] > 0))[0]
    indices = indices[torch.argsort(prediction['scores'][indices], descending=True, stable=True)[:topk]]
    return {k: prediction[k][indices].detach() for k in ('boxes', 'scores', 'labels')}


def stable_pairs(base, flipped_in_base, iou_threshold=0.5):
    """Greedy descending IoU, then base/flip index ties; one-to-one same-class."""
    a, b = base['boxes'], flipped_in_base['boxes']
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (rb-lt).clamp(min=0).prod(-1)
    area_a, area_b = (a[:, 2:]-a[:, :2]).prod(-1), (b[:, 2:]-b[:, :2]).prod(-1)
    ious = intersection / (area_a[:, None]+area_b[None, :]-intersection)
    candidates = [(float(ious[i, j]), i, j) for i in range(len(a)) for j in range(len(b))
                  if base['labels'][i] == flipped_in_base['labels'][j] and ious[i, j] >= iou_threshold]
    candidates.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_a, used_b, pairs = set(), set(), []
    for iou, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, iou))
    return pairs


def combine_support(base, flipped, width, iou_threshold=0.5):
    mapped = {**flipped, 'boxes': flip_boxes(flipped['boxes'], width)}
    pairs = stable_pairs(base, mapped, iou_threshold)
    ii = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=base['boxes'].device)
    jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=base['boxes'].device)
    return {'base': base, 'flipped': flipped, 'pairs': pairs,
            'stable': {'boxes': base['boxes'][ii], 'flip_boxes': flipped['boxes'][jj],
                       'labels': base['labels'][ii],
                       'scores': (base['scores'][ii]+flipped['scores'][jj])/2}}


@torch.no_grad()
def original_support(detector, image, threshold=0.5, topk=20, iou_threshold=0.5):
    base = select_predictions(detector(image)[0], threshold, topk)
    flipped = select_predictions(detector(image.flip(-1))[0], threshold, topk)
    return combine_support(base, flipped, image.shape[-1], iou_threshold)


def fixed_roi_logits(detector, image, boxes):
    """Single enhanced BCHW image; detached original-view xyxy boxes, no targets."""
    model = detector.model
    model.eval()
    height, width = image.shape[-2:]
    transformed, _ = model.transform(list(image), None)
    rh, rw = transformed.image_sizes[0]
    scaled = boxes.detach() * boxes.new_tensor([rw/width, rh/height, rw/width, rh/height])
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    pooled = model.roi_heads.box_roi_pool(features, [scaled], transformed.image_sizes)
    encoded = model.roi_heads.box_head(pooled)
    logits, _ = model.roi_heads.box_predictor(encoded)
    return logits
