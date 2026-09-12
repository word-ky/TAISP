import pytest

from scripts.analyze_t008 import ClusterStats, failure_flags, summarize
from taisp.analysis.detection_proxies import PROXY_DIRECTION


def test_unequal_cluster_counts_and_undefined_conditional_denominator():
    stats = ClusterStats([{'image_id': 1}, {'image_id': 1}, {'image_id': 2}])
    q = stats.estimate([1, 1, 0])
    assert q['estimate'] == pytest.approx(2/3) and q['ci95'] == [0., 1.]
    assert stats.estimate([1, 0, 1], [False, False, False])['estimate'] is None
    assert stats.estimate([1, float('nan'), 3])['estimate'] == 2


def test_failure_orientation_and_overlapping_categories():
    delta = {k: 0 for k in PROXY_DIRECTION}
    assert not any(failure_flags(delta).values())
    delta.update(best_iou_mean=-.1, class_score_mean=-.2, fp05=1, duplicates=1)
    assert all(failure_flags(delta).values())


def test_no_gt_not_counted_as_geometry_success():
    valid = {k: 0 for k in PROXY_DIRECTION}
    valid['best_iou_mean'] = -.1
    missing = {k: None if PROXY_DIRECTION[k] == 1 else 0 for k in PROXY_DIRECTION}
    rows = [{'image_id': i, 'loss_delta': -1, 'delta': d, 'failure': failure_flags(d)}
            for i, d in enumerate([valid, missing])]
    result = summarize(rows)
    geometry = result['failures']['geometry']
    assert geometry['given_loss_good']['estimate'] == 1
    assert geometry['given_loss_good']['valid_observations'] == 1
    assert geometry['undefined_observations'] == 1
    assert result['failures']['false_positives']['given_loss_good']['valid_observations'] == 2
