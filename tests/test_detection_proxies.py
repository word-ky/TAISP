import numpy as np
import pytest

from taisp.analysis.detection_proxies import image_proxies, greedy_match, iou_matrix


def pred(box, score, label=1):
    return {'bbox': box, 'score': score, 'category_id': label}


def gt(box, ident=1, label=1, crowd=0):
    return {'bbox': box, 'id': ident, 'category_id': label, 'iscrowd': crowd}


def test_same_class_score_order_duplicates_and_thresholds():
    q = image_proxies([pred([0, 0, 10, 10], .5), pred([0, 0, 10, 10], .9),
                       pred([0, 0, 10, 10], .95, 2), pred([30, 30, 2, 2], .05)], [gt([0, 0, 10, 10])])
    assert q['recall50'] == q['recall75'] == 1
    assert q['gt_match_prediction50'] == [1]
    assert q['tp_scores50'] == [.9] and q['class_scores'] == [.9]
    assert q['fp05'] == 3 and q['fp50'] == 2 and q['duplicates'] == 1


def test_geometry_class_absence_and_independent_iou_thresholds():
    q = image_proxies([pred([0, 0, 5, 10], .8)], [gt([0, 0, 10, 10]), gt([0, 0, 10, 10], 2, 2)])
    assert q['recall50'] == .5 and q['recall75'] == 0
    assert q['best_iou'] == [.5, 0] and q['agnostic_best_iou'] == [.5, .5]
    assert q['class_scores'] == [.8, 0] and q['tp_score75_mean'] is None
    np.testing.assert_array_equal(iou_matrix([], [[0, 0, 1, 1]]), np.empty((0, 1)))


def test_greedy_unmatched_gt_and_tie_order():
    tp, duplicate, matched = greedy_match(np.array([[.8, .8], [.9, .7], [.6, .6]]), np.ones((3, 2), dtype=bool), .5)
    assert matched.tolist() == [0, 1]
    assert tp.tolist() == [True, True, False] and duplicate.tolist() == [False, False, True]


def test_empty_predictions_no_gt_and_crowd_policy():
    q = image_proxies([], [gt([0, 0, 10, 10])])
    assert q['recall50'] == 0 and q['best_iou'] == q['class_scores'] == [0]
    assert q['tp_score50_mean'] is None and q['fp05'] == 0
    q = image_proxies([pred([0, 0, 10, 10], .9)], [gt([0, 0, 10, 10], crowd=1)])
    assert q['n_gt'] == 0 and q['n_crowd_gt'] == 1 and q['recall50'] is None
    assert q['fp50'] == 1  # Proxy policy intentionally differs from official crowd-ignore AP.
