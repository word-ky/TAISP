import math

from scripts.report_t005 import basics, paired, finite_json


def test_fallbacks_stay_in_denominators_and_paired_effect():
    rows = [{'image_id': i, 'family': 'clean', 'severity': 0,
             'g_sem': [1., 0.] if i else [0., 0.], 'gradient_cosine': .5 if i else None,
             'det_loss_delta_sem1': -1. if i else 0.} for i in range(2)]
    ref = [{**r, 'gradient_cosine': .2, 'det_loss_delta_sem1': 1.} for r in rows]
    s = basics(rows)
    assert s['count'] == 2 and s['cosine_valid']['count'] == 1
    assert s['cosine_zero_coded']['mean'] == .25
    assert s['positive_fraction_all'] == .5 and s['benefit_fraction_all'] == .5
    assert s['coordinate_energy_mean_zero_for_empty'] == [.5, 0.]
    assert math.isclose(paired(rows, ref)['cosine_zero_coded']['estimate'], .05)
    assert paired(rows, ref)['benefit_rate']['estimate'] == .5
    assert finite_json({'undefined': float('nan'), 'nested': [float('inf'), 1.]}) == {'undefined': None, 'nested': [None, 1.]}
