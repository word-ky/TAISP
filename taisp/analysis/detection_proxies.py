"""Offline proxies on retained detections, not COCO AP or deployment losses."""
import numpy as np


PROXY_DIRECTION = {'recall50': 1, 'recall75': 1, 'best_iou_mean': 1, 'best_iou_median': 1,
                   'agnostic_iou_mean': 1, 'class_score_mean': 1, 'tp_score50_mean': 1,
                   'tp_score75_mean': 1, 'fp05': -1, 'fp50': -1, 'duplicates': -1}


def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    aa, bb = a.copy(), b.copy()
    aa[:, 2:] += aa[:, :2]
    bb[:, 2:] += bb[:, :2]
    wh = np.maximum(0, np.minimum(aa[:, None, 2:], bb[None, :, 2:])-np.maximum(aa[:, None, :2], bb[None, :, :2]))
    inter = wh.prod(-1)
    union = a[:, 2:].prod(-1)[:, None]+b[:, 2:].prod(-1)[None, :]-inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match(overlaps, same_class, threshold):
    """Rows already score-descending; columns ordered by GT annotation id."""
    n, g = overlaps.shape
    gt_match = np.full(g, -1, dtype=int)
    tp, duplicate = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in range(n):
        eligible = same_class[i] & (overlaps[i] >= threshold)
        available = np.flatnonzero(eligible & (gt_match < 0))
        if len(available):
            j = available[np.argmax(overlaps[i, available])]
            gt_match[j] = i
            tp[i] = True
        elif np.any(eligible & (gt_match >= 0)):
            duplicate[i] = True
    return tp, duplicate, gt_match


def image_proxies(predictions, annotations):
    ordered = sorted(enumerate(predictions), key=lambda z: -z[1]['score'])
    preds = [p for _, p in ordered]
    gt = sorted([a for a in annotations if not a.get('iscrowd', 0) and a['bbox'][2] > 0 and a['bbox'][3] > 0], key=lambda a: a['id'])
    scores = np.array([p['score'] for p in preds], dtype=float)
    same = np.array([p['category_id'] for p in preds])[:, None] == np.array([a['category_id'] for a in gt])[None, :]
    overlaps = iou_matrix([p['bbox'] for p in preds], [a['bbox'] for a in gt])
    n, g = len(preds), len(gt)
    best = np.where(same, overlaps, 0).max(0) if n else np.zeros(g)
    agnostic = overlaps.max(0) if n else np.zeros(g)
    class_score = np.where(same, scores[:, None], 0).max(0) if n else np.zeros(g)
    mean = lambda values: float(np.mean(values)) if len(values) else None
    out = {'n_gt': g, 'n_predictions': n, 'n_crowd_gt': sum(a.get('iscrowd', 0) for a in annotations),
           'gt_ids': [a['id'] for a in gt], 'best_iou': best.tolist(), 'agnostic_best_iou': agnostic.tolist(),
           'class_scores': class_score.tolist(), 'best_iou_mean': mean(best),
           'best_iou_median': float(np.median(best)) if g else None,
           'agnostic_iou_mean': mean(agnostic), 'class_score_mean': mean(class_score)}
    for name, threshold in (('50', .5), ('75', .75)):
        tp, duplicate, matches = greedy_match(overlaps, same, threshold)
        tp_scores = scores[tp]
        out.update({f'recall{name}': float(tp.sum()/g) if g else None,
                    f'tp_scores{name}': tp_scores.tolist(), f'tp_score{name}_mean': mean(tp_scores),
                    f'gt_match_prediction{name}': [ordered[j][0] if j >= 0 else None for j in matches]})
        if name == '50':
            out.update(fp05=int(((~tp) & (scores >= .05)).sum()), fp50=int(((~tp) & (scores >= .5)).sum()),
                       duplicates=int(duplicate.sum()))
    return out
