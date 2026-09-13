import copy

import numpy as np

from taisp.analysis.linear_capacity import (
    cross_validate, fit, gate, metrics, null_comparison, pair_rows, permutation_schedule,
)
from taisp.analysis.source_replication import fold_indices


def test_svd_affine_reconstruction_and_minimum_norm_rank_deficient_solution():
    rng = np.random.default_rng(27)
    z = rng.normal(size=(48,3))
    h = np.column_stack((z,z[:,0],np.zeros(48)))
    g = z@rng.normal(size=(3,8))+np.arange(8)
    result = fit(h,g)
    b = np.array(result['B'])
    prediction = result['g_bar']+(h-result['mu'])@b.T
    np.testing.assert_allclose(prediction,g,rtol=1e-12,atol=1e-12)
    assert result['numerical_rank']==3 and result['rank_deficient']
    np.testing.assert_allclose(b[:,0],b[:,3],atol=1e-12)
    np.testing.assert_array_equal(b[:,4],0.)
    # The null-space perturbation preserves predictions but increases coefficient norm.
    changed = b.copy()
    changed[:,0] += 1
    changed[:,3] -= 1
    np.testing.assert_allclose((h-result['mu'])@changed.T,(h-result['mu'])@b.T,atol=1e-12)
    assert np.linalg.norm(changed)>np.linalg.norm(b)


def test_fixed_numerical_rank_tolerance_discards_only_subthreshold_singular_value():
    diagonal = np.diag([1.,1e-12,1e-17])
    h = np.vstack((diagonal,-diagonal))
    result = fit(h,h)
    assert result['numerical_rank']==2
    expected = np.finfo(np.float64).eps*6*np.sqrt(2)
    np.testing.assert_allclose(result['rank_tolerance'],expected,rtol=1e-14)
    np.testing.assert_allclose(result['B'],np.diag([1.,1.,0.]),atol=1e-14)


def test_four_folds_keep_pairs_and_exclude_heldout_gradients_from_fit():
    rng = np.random.default_rng(7)
    h = rng.normal(size=(64,16))
    b = rng.normal(size=(8,16))
    g = h@b.T+np.arange(8)
    result = cross_validate(h,g)
    np.testing.assert_allclose(result['predicted_gradient'],g,atol=1e-12)
    all_test = []
    for f,(train,test) in enumerate(fold_indices()):
        assert not set(train//2)&set(test//2)
        assert len(train)==48 and len(test)==16
        all_test.extend(test)
        changed = g.copy()
        changed[test] += 100
        other = cross_validate(h,changed)
        assert result['folds'][f]==other['folds'][f]
        np.testing.assert_array_equal(np.array(result['predicted_gradient'])[test],np.array(other['predicted_gradient'])[test])
    assert sorted(all_test)==list(range(64))


def test_128_deterministic_permutations_preserve_pairs_positions_and_marginals():
    schedule = permutation_schedule()
    assert schedule==permutation_schedule() and len(schedule)==128
    assert len({tuple(order) for replicate in schedule for order in replicate})>1
    for replicate in schedule:
        assert len(replicate)==4
        for f,order in enumerate(replicate):
            assert sorted(order)==list(range(24))
            rows = pair_rows(order)
            assert sorted(rows)==list(range(48))
            np.testing.assert_array_equal(rows[1::2],rows[::2]+1)
            np.testing.assert_array_equal(rows%2,np.arange(48)%2)
            train,test = fold_indices()[f]
            assert not set(train[rows]//2)&set(test//2)
    rng = np.random.default_rng(6)
    h,g = rng.normal(size=(48,16)),rng.normal(size=(48,8))
    order = schedule[0][0]
    actual = fit(h,g,order)
    reference = fit(h,g[pair_rows(order)])
    np.testing.assert_allclose(actual['B'],reference['B'],atol=1e-12)
    np.testing.assert_allclose(actual['g_bar'],g.mean(0),atol=1e-14)


def test_metrics_pool_sums_zero_cosines_and_first_order_sign():
    residual = np.array([[1.,0.],[3.,0.],[0.,0.]])
    predicted = np.array([[0.,0.],[2.,0.],[1.,0.]])
    g = np.array([[2.,0.],[4.,0.],[1.,0.]])
    result = metrics(g,residual,predicted,predicted+1)
    assert result['residual_sse']==3 and result['residual_energy']==10
    assert result['R2_residual']==.7
    assert result['positive_residual_dot_count']==1 and result['median_residual_cosine']==0
    assert result['negative_first_order_count']==3
    np.testing.assert_allclose(result['first_order_sum'],-.001*(2+12+2))
    common = metrics(g,residual,np.zeros_like(g),g)
    assert common['R2_residual']==0 and common['median_residual_cosine']==0


def test_strict_gate_and_permutation_tail_boundaries():
    observed = {'R2_residual':.10,'median_residual_cosine':.20,'positive_residual_dot_count':44}
    summary = {'overall':{'linear':observed},'clean':{'linear':{'positive_residual_dot_count':20}},
               'corrupted':{'linear':{'positive_residual_dot_count':24}}}
    summary.update({f'fold{i}':{'linear':{'R2_residual':.01 if i<3 else 0.}} for i in range(4)})
    comparisons = {'R2_residual':{'null_95th_percentile_linear':.099},
                   'median_residual_cosine':{'null_95th_percentile_linear':.199}}
    assert gate(summary,comparisons)['passed']
    equal = copy.deepcopy(comparisons)
    equal['R2_residual']['null_95th_percentile_linear']=.10
    assert not gate(summary,equal)['passed']
    for key,value in [('R2_residual',.099),('median_residual_cosine',.199),('positive_residual_dot_count',43)]:
        modified = copy.deepcopy(summary)
        modified['overall']['linear'][key]=value
        assert not gate(modified,comparisons)['passed']
    for group in ('clean','corrupted'):
        modified = copy.deepcopy(summary)
        modified[group]['linear']['positive_residual_dot_count']=19
        assert not gate(modified,comparisons)['passed']
    modified = copy.deepcopy(summary)
    modified['fold2']['linear']['R2_residual']=0.
    assert not gate(modified,comparisons)['passed']
    result = null_comparison(120,np.arange(128))
    assert result['null_at_least_observed']==8
    assert result['corrected_one_sided_tail']==9/129
    assert result['empirical_percentile_strict']==100*120/128
    np.testing.assert_allclose(result['null_95th_percentile_linear'],120.65)
    ties = null_comparison(1,np.ones(128))
    assert ties['ties']==128 and ties['corrected_one_sided_tail']==1
    assert ties['empirical_percentile_strict']==0
