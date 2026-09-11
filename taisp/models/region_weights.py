"""Detached original-detector regions mapped to the parity-correct CLIP grid."""

import torch

from taisp.losses.clip_semantic import resize_crop_geometry


@torch.no_grad()
def region_patch_weights(boxes, scores, image_size, *, threshold=.5, topk=20):
    h, w = image_size
    rh, rw, top, left = resize_crop_geometry(h, w)
    indices = torch.nonzero(scores >= threshold).flatten()
    indices = indices[torch.argsort(scores[indices], descending=True, stable=True)[:topk]]
    selected_boxes, selected_scores = boxes[indices], scores[indices]
    mapped = selected_boxes * boxes.new_tensor([rw/w, rh/h, rw/w, rh/h])
    mapped -= boxes.new_tensor([left, top, left, top])
    mapped = mapped.clamp(0, 224)
    visible = (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
    mapped = mapped[visible]
    yy, xx = torch.meshgrid(torch.arange(7, device=boxes.device), torch.arange(7, device=boxes.device), indexing='ij')
    low = torch.stack([xx.flatten(), yy.flatten()], -1).to(boxes.dtype) * 32
    intersection = (torch.minimum(mapped[:, None, 2:], low[None]+32)
                    - torch.maximum(mapped[:, None, :2], low[None])).clamp_min(0).prod(-1)
    raw = (intersection/1024 * selected_scores[visible, None]).sum(0)
    support = intersection.sum(0) > 0
    fallback = raw.sum().item() == 0
    weights = torch.full_like(raw, 1/49) if fallback else raw/raw.sum()
    return {'weights': weights, 'support': support, 'raw_weights': raw,
            'original_boxes': selected_boxes[visible], 'scores': selected_scores[visible],
            'mapped_boxes': mapped, 'selected_count': len(indices), 'visible_count': int(visible.sum()),
            'uniform_fallback': fallback, 'support_fraction': support.float().mean().item(),
            'effective_patches': (1/weights.square().sum()).item()}


@torch.no_grad()
def original_region_weights(detector, original, *, threshold=.5, topk=20):
    """Call once on the original corrupted image, with no labels or adaptation."""
    detector.eval().requires_grad_(False)
    prediction = detector(original)[0]
    return region_patch_weights(prediction['boxes'], prediction['scores'], original.shape[-2:],
                                threshold=threshold, topk=topk)
