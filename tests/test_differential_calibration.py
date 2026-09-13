import copy

import numpy as np

from taisp.analysis.differential_calibration import crossfit, evaluate, fit, folds, gate, null_comparison
from taisp.analysis.linear_capacity import pair_rows
from taisp.analysis.spatial_action import EPS


def test_known_diagonal_coefficients_and_no_intercept():
    x = np.array([[1., 2., 0.], [2., -1., 0.], [3., 4., 0.]])
    y = x * np.array([-2., .5, 7.])
    got = np.array(fit(x, y)['a'])
    np.testing.assert_allclose(got[:2], [-2., .5], atol=1e-12)
    assert got[2] == 0
    shifted = np.asarray(fit(x, y + 4.)['a'])
    np.testing.assert_allclose(shifted, (x * (y + 4.)).sum(0) / ((x*x).sum(0) + EPS))
    np.testing.assert_array_equal(np.zeros(3) * shifted, np.zeros(3))


def test_crossfit_heldout_targets_cannot_change_own_predictions():
    rng = np.random.default_rng(8)
    dp, dt = rng.normal(size=(32, 8)), rng.normal(size=(32, 8))
    prediction, receipts = crossfit(dp, dt)
    for f, (train, test) in enumerate(folds()):
        changed = dt.copy()
        changed[test] += 1000
        next_prediction, next_receipts = crossfit(dp, changed)
        np.testing.assert_array_equal(prediction[test], next_prediction[test])
        assert receipts[f]['a'] == next_receipts[f]['a']
        assert not set(train // 2) & set(test // 2)
        assert len(train) == 24 and len(test) == 8


def test_pair_permutation_moves_target_pairs_only():
    rng = np.random.default_rng(20260913)
    schedule = [[rng.permutation(12).tolist() for _ in range(4)] for _ in range(256)]
    again = np.random.default_rng(20260913)
    assert schedule == [[again.permutation(12).tolist() for _ in range(4)] for _ in range(256)]
    x = np.arange(256, dtype=float).reshape(32, 8)
    for perms in schedule:
        for f, (train, test) in enumerate(folds()):
            moved = train[pair_rows(perms[f])]
            assert sorted(moved.tolist()) == train.tolist()
            np.testing.assert_array_equal(moved[1::2], moved[::2] + 1)
            assert not set(moved) & set(test)
    _, receipts = crossfit(x, x**2, schedule[0])
    for f, (train, _) in enumerate(folds()):
        np.testing.assert_array_equal(receipts[f]['target_indices'], train[pair_rows(schedule[0][f])])
        np.testing.assert_allclose(receipts[f]['a'], fit(x[train], x[receipts[f]['target_indices']]**2)['a'])


def test_norm_matching_shared_preservation_and_known_productivity():
    st, dt, sp, dp = map(np.array, ([1., 2.], [3., 0.], [2., 1.], [-1., 0.]))
    raw = (np.r_[st+dt, st-dt] @ np.r_[sp+dp, sp-dp]) / (np.linalg.norm(np.r_[sp+dp, sp-dp]) + EPS)
    r = evaluate(st, dt, sp, dp, np.array([7., 0.]), raw)
    assert r['norm_check']['passed']
    qnm, q = np.array(r['q_diff_nm']), np.array(r['q'])
    np.testing.assert_allclose(np.linalg.norm(qnm), np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose((q[:2]+q[2:])/2, sp, atol=1e-15)
    np.testing.assert_allclose(r['metrics']['D_cal'], 14 / np.sqrt(12), atol=2e-12)
    np.testing.assert_allclose(r['metrics']['C_diff_cal'], 6 / np.sqrt(12), atol=2e-12)
    assert r['raw_diff_dot_sign'] == -1 and r['diff_dot_sign'] == 1
    for sp, dp in [(np.zeros(2), np.zeros(2)), (np.ones(2), np.zeros(2)), (np.ones(2), np.ones(2))]:
        zero = evaluate(np.ones(2), np.ones(2), sp, dp, dp, 0.)
        assert zero['norm_check']['passed']
    collapsed = evaluate(st, dt, sp, np.ones(2), np.zeros(2), raw)
    assert not collapsed['norm_check']['passed']  # No fallback direction or second scaling.


def test_exact_gate_boundaries_and_corrected_null_tail():
    scopes = {'overall': {'metrics': {m: {'positive_count': 20, 'median': .01}
                                    for m in ('C_diff_cal', 'Delta_D_cal')}},
              'corrupted': {'metrics': {m: {'positive_count': 10, 'median': .01}
                                      for m in ('C_diff_cal', 'Delta_D_cal')}}}
    scopes.update({f'block{i}': {'metrics': {m: {'median': .01 if i < 3 else 0.}
                                          for m in ('C_diff_cal', 'Delta_D_cal')}} for i in range(4)})
    comp = {k: {'observed': 2., 'null_95th_percentile_linear': 1.} for k in ('median_Delta_D_cal', 'positive_Delta_D_cal')}
    assert gate(scopes, comp)['passed']
    for metric in ('C_diff_cal', 'Delta_D_cal'):
        for scope, key, value in [('overall', 'positive_count', 19), ('corrupted', 'positive_count', 9),
                                  ('overall', 'median', 0.), ('block2', 'median', 0.)]:
            changed = copy.deepcopy(scopes)
            changed[scope]['metrics'][metric][key] = value
            assert not gate(changed, comp)['passed']
    for key in comp:
        changed = copy.deepcopy(comp)
        changed[key]['observed'] = 1.
        assert not gate(scopes, changed)['passed']
    comparison = null_comparison(1., [0.] * 255 + [1.])
    assert comparison['ties'] == 1 and comparison['corrected_one_sided_tail'] == 2/257
    assert comparison['null_95th_percentile_linear'] == 0.
