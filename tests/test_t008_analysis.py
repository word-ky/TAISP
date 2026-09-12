import pytest

from scripts.analyze_t008 import ClusterStats, failure_flags
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
