import numpy as np

from scripts.report_t003 import paired_delta


def test_paired_bootstrap_matches_image_condition_keys_not_row_order():
    ref = [dict(image_id=i, family='gamma', severity=s, value=float(i+s)) for i in range(4) for s in (1, 2)]
    rows = [{**r, 'value': r['value']+2} for r in reversed(ref)]
    result = paired_delta(rows, ref, lambda r: r['value'], replicates=50)
    assert result['estimate'] == 2
    np.testing.assert_array_equal(result['ci95'], [2, 2])
