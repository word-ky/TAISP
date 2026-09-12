import pytest

from scripts.report_t007 import flip_values, ratio_stratum, cluster_fraction


def test_flip_direction_zeros_strata_and_unequal_cluster_sizes():
    row = lambda v: {'target': {'det_loss_delta_sem1': v}}
    assert flip_values(row(-.1), row(.1))['harmful_to_beneficial'] == 1
    assert flip_values(row(.1), row(-.1))['beneficial_to_harmful'] == 1
    assert flip_values(row(0), row(.1))['either_zero'] == 1
    for ratio, expected in [(0, '[0,1)'), (1, '[1,2)'), (2, '[2,4)'), (4, '[4,inf)'), (None, 'clip_zero')]:
        assert ratio_stratum({'diagnostics': [{'detector_clip_ratio': ratio}]}) == expected
    q = cluster_fraction([{'image_id': 1}, {'image_id': 1}, {'image_id': 2}], [1, 1, 0])
    assert q['estimate'] == pytest.approx(2/3)
    assert q['ci95'] == [0., 1.]
