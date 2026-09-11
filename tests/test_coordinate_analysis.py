import numpy as np

from scripts.analyze_t003_coordinates import bucket, decomposition, saturation_groups


def test_known_coordinate_contributions_energy_and_saturation():
    rows = [dict(family='gamma', g_sem=[3., 4., 0, 0, 0, 0, 0, 0],
                 g_det=[2., -1., 0, 0, 0, 0, 0, 0], det_loss_delta_sem1=delta,
                 saturation_before=before, saturation_1=after)
            for delta, before, after in [(-1., 0., .01), (2., .05, .11)]]
    d = decomposition(rows)
    assert d['coordinates']['gamma']['signed_contribution_mean'] == 6
    assert d['coordinates']['red_gain']['signed_contribution_mean'] == -4
    assert d['coordinates']['red_gain']['sign_agreement'] == 0
    np.testing.assert_allclose(d['coordinates']['gamma']['relative_semantic_energy_mean'], .36)
    np.testing.assert_allclose(d['outside_norm_fraction_mean'], .8)
    s = saturation_groups(rows)
    assert s['[0,1%]->[0,1%]']['benefit_fraction'] == 1
    assert s['(1%,5%]->(10%,100%]']['harm_fraction'] == 1
    assert [bucket(v) for v in [0, .01, .05, .1, 1]] == [0, 0, 1, 2, 3]
