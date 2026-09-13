import copy
import numpy as np

from taisp.analysis.source_replication import statistics, crossfit, fold_indices, utility, decision


def test_variable_size_moments_and_energy_reconstruction():
    rng = np.random.default_rng(21)
    for n in (16,64):
        h,g = rng.normal(size=(n,16))+1,rng.normal(size=(n,8))+.2
        result = statistics(h,g)
        a,c = np.array(result['A']),np.array(result['C'])
        np.testing.assert_allclose(a+c,np.einsum('ni,nj->ij',g,h)/n,atol=1e-14)
        assert len(result['features']['pairs']) == n//2
        assert len(result['full_output']['same_image_distances']) == n//2
        for energy,cross in [('total_energy','raw_cross_energy'),('centered_energy','centered_cross_energy')]:
            calculated = sum(v[energy] for v in result['components'].values())+sum(v[cross] for v in result['cross_terms'])
            np.testing.assert_allclose(calculated,result['full_output'][energy],rtol=1e-12)


def test_image_grouped_crossfit_excludes_heldout_gradients_and_uses_train_mean():
    rng = np.random.default_rng(18)
    h,g = rng.normal(size=(64,16)),rng.normal(size=(64,8))
    result = crossfit(h,g)
    seen = []
    for fold,(train,test) in enumerate(fold_indices()):
        assert len(train) == 48 and len(test) == 16
        assert not (set(train//2) & set(test//2))
        seen.extend(test.tolist())
        changed = g.copy()
        changed[test] += 1000
        other = crossfit(h,changed)
        assert other['folds'][fold] == result['folds'][fold]
        c = np.array(result['folds'][fold]['C'])
        expected = -.001*(h[test]-h[train].mean(0))@c.T
        np.testing.assert_allclose(result['folds'][fold]['perturbations']['cov']['outputs'],expected,atol=1e-14)
    assert sorted(seen) == list(range(64))


def test_exact_predeclared_thresholds():
    records = [{'clean':i%2==0,'directions':{'cov':{'g_dot_delta':-1. if i<40 else 0.,
                                                        'cosine_to_negative_g':.1}}} for i in range(64)]
    assert utility(records)['passed']
    records[0]['directions']['cov']['g_dot_delta'] = 0.
    assert not utility(records)['passed']
    for r in records:
        r['directions']['cov']['g_dot_delta'] = -1.
        r['directions']['cov']['cosine_to_negative_g'] = 0.
    assert not utility(records)['passed']
    for i,r in enumerate(records):
        r['directions']['cov']['cosine_to_negative_g'] = .1
        r['directions']['cov']['g_dot_delta'] = -1. if not r['clean'] or i<34 else 1.
    assert utility(records)['negative_count'] == 49 and not utility(records)['passed']
    overall = {'features':{'median_rho':.10},'r_C':.10,'r_C_out':.01,'full_output':{'centered_fraction':.049}}
    blocks = [{'full_output':{'centered_fraction':v}} for v in (.099,.099,.099,.10)]
    assert decision(overall,blocks,{'passed':True})['both_pass']
    boundary = copy.deepcopy(overall)
    boundary['full_output']['centered_fraction'] = .05
    assert not decision(boundary,blocks,{'passed':True})['common_mode_replication']
    blocks[0]['full_output']['centered_fraction'] = .10
    assert not decision(overall,blocks,{'passed':True})['structural_replication']
