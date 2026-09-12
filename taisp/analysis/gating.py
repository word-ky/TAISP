"""T010 pure offline scalar extraction, ranking and frozen prediction selection."""
from collections import defaultdict
import math

import numpy as np


SCORES = ('clip_norm', 'det_norm', 'log_norm_ratio', 'gradient_cosine',
          'pseudo_loss', 'support_count', 'support_confidence')
HYBRID = 'det_pseudo_clip_radius'


def extract_scores(row):
    d = row['diagnostics'][0]
    gd, gc = np.asarray(d['detector_gradient']), np.asarray(d['clip_gradient'])
    denominator = float(np.linalg.norm(gd)*np.linalg.norm(gc))
    nd, nc = d['detector_gradient_norm'], d['clip_gradient_norm']
    confidence = row['support']['scores']
    return {'clip_norm': nc, 'det_norm': nd,
            'log_norm_ratio': math.log((nd+1e-12)/(nc+1e-12)),
            'gradient_cosine': float(gd @ gc)/denominator if denominator else 0.,
            'pseudo_loss': d['total'], 'support_count': row['support_count'],
            'support_confidence': sum(confidence)/len(confidence) if confidence else 0.}


def rank_mask(values, image_ids, coverage, orientation):
    """Pooled score rank; ties use image ID then immutable input-row ordinal."""
    values = np.asarray(values, dtype=float)
    key = -values if orientation == 'high' else values
    order = np.lexsort((np.arange(len(values)), image_ids, key))
    count = int(len(values)*coverage)
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:count]] = True
    boundary = int(order[count-1]) if count else None
    return mask, {'selected': count, 'total': len(values),
                  'cutoff_score': float(values[boundary]) if boundary is not None else None,
                  'boundary_image_id': int(image_ids[boundary]) if boundary is not None else None,
                  'boundary_row_ordinal': boundary}


def compose_predictions(raw, hybrid, selected_ids, image_ids):
    """Keep native per-image prediction order and protect anchors from COCOeval mutation."""
    by_raw, by_hybrid = defaultdict(list), defaultdict(list)
    for source, indexed in ((raw, by_raw), (hybrid, by_hybrid)):
        for prediction in source:
            indexed[prediction['image_id']].append(prediction)
    return [dict(p) for i in image_ids
            for p in (by_hybrid if i in selected_ids else by_raw)[i]]


def latency_estimate(row, score, selected):
    setup, adaptation = row['source_setup_seconds'], row['adapt_seconds_3']
    if score == 'no_adapt':
        return setup
    if score in ('full_hybrid', 'support_count', 'support_confidence'):
        return setup+selected*adaptation
    identity_joint = row['identity_step_seconds']
    return setup+identity_joint+selected*(adaptation-identity_joint)
