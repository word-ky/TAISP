"""T025-A GT-free exposure-pair geometry candidate; analysis only."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from .roi_equivariance import (sha, write, setup, images, common_jvp, isolated,
                               state_hash, support_mask, mask_receipt, select_predictions)


NAME = 'roi_bbox_exposure_stability'
EXPOSURES = (1.20, 1/1.20)


def expose(y, a):
    return a*y/(1+(a-1)*y)


def class_deltas(regression, labels):
    return regression.reshape(len(labels), -1, 4)[torch.arange(len(labels), device=labels.device), labels]


def fixed_roi_regression(detector, image, boxes):
    """Unchanged fixed-ROI transform/backbone/head path, raw predictor deltas."""
    model = detector.model
    height, width = image.shape[-2:]
    transformed, _ = model.transform(list(image), None)
    rh, rw = transformed.image_sizes[0]
    scaled = boxes.detach()*boxes.new_tensor([rw/width, rh/height, rw/width, rh/height])
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    pooled = model.roi_heads.box_roi_pool(features, [scaled], transformed.image_sizes)
    encoded = model.roi_heads.box_head(pooled)
    _, regression = model.roi_heads.box_predictor(encoded)
    return regression


def candidate_cotangent(detector, image, boxes, labels):
    if not len(boxes):
        return torch.zeros_like(image), {'loss': 0., 'per_object': [], 'delta_hi': [], 'delta_lo': [],
                                        'empty_support': True, 'class_indices': []}
    y = image.detach().requires_grad_(True)
    hi, lo = [class_deltas(fixed_roi_regression(detector, expose(y, a), boxes), labels) for a in EXPOSURES]
    per_object = F.smooth_l1_loss(hi, lo, beta=1., reduction='none').sum(-1)
    loss = per_object.mean()
    c = torch.autograd.grad(loss, y)[0].detach()
    assert torch.isfinite(loss) and torch.isfinite(c).all()
    return c, {'loss': loss.item(), 'per_object': per_object.detach().cpu().tolist(),
        'delta_hi': hi.detach().cpu().tolist(), 'delta_lo': lo.detach().cpu().tolist(),
        'class_indices': labels.cpu().tolist(), 'empty_support': False}


def run(manifest_path, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, isp, env = setup()
    env.update({'image_manifest_sha256': sha(manifest_path), 'exposures': EXPOSURES,
                'candidate': NAME, 'smooth_l1_beta': 1.})
    write(output/'environment.json', env)
    rows = []
    for index, (info, case, image) in enumerate(images(json.loads(manifest_path.read_text()))):
        with torch.no_grad():
            selected = select_predictions(source(image)[0], .5, 20)
        mask, rectangles = support_mask(image, selected['boxes'])
        identity = isp(image, image.new_zeros(8)).detach()
        c, objective = candidate_cotangent(source, identity, selected['boxes'], selected['labels'])
        refs, jacobian = common_jvp(image, isp, mask, {NAME: c})
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case, 'block': info['block'],
            'support_count': len(selected['boxes']), 'supports': {k: v.cpu().tolist() for k,v in selected.items()},
            'roi_order': list(range(len(selected['boxes']))), 'mask': mask_receipt(mask, rectangles),
            'objectives': {NAME: objective}, 'gradients': refs, 'jacobian': jacobian,
            'cotangent': {'shape': list(c.shape), 'dtype': str(c.dtype), 'finite': bool(torch.isfinite(c).all()),
                'l2': c.double().norm().item(), 'max_abs': c.abs().max().item(),
                'sha256': hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest()},
            'isolation': isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())}
        assert row['isolation']
        write(output/f'record_{index:02d}.json', row)
        rows.append(row)
        print(json.dumps({'stage': 'candidate', 'episode': index, 'supports': len(selected['boxes'])}), flush=True)
    assert len(rows) == 48 and state_hash(source) == env['source_state_sha256']
    forbidden = [k for k in sys.modules if k.startswith('taisp.') and
                 any(s in k for s in ('oracle', 'source_meta', 'common_jacobian', '_reference'))]
    assert not forbidden, forbidden
    write(output/'records.json', rows)
    write(output/'completion.json', {'status': 'candidate_complete', 'episodes': len(rows),
        'source_hash_after': state_hash(source), 'frozen_eval_grad_none': isolated(source),
        'forbidden_modules_loaded': forbidden, 'seconds': time.perf_counter()-started})
    write(output/'sha256.json', {p.name: sha(p) for p in sorted(output.glob('*.json'))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.output)
