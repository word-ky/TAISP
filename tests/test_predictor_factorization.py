import numpy as np

from taisp.analysis.predictor_factorization import analyze, factorize, feature_statistics, roundoff_check, triage


def test_factorization_component_sum_and_nonorthogonal_energy():
    rng = np.random.default_rng(17)
    h,g = rng.normal(size=(8,16))+.3,rng.normal(size=(8,8))+.2
    mu,gbar,a,c = factorize(h,g)
    gw = np.einsum('ni,nj->nij',g,h)
    np.testing.assert_allclose(a+c,gw.mean(0),rtol=1e-12,atol=1e-14)
    weight,bias = -.001*gw.mean(0),-.001*gbar
    outputs = h@weight.T+bias
    arrays = {'features':h,'phi0_gradients':g,'head_weight_gradients':gw,'head_bias_gradients':g,
              'saved_conditioning_outputs':outputs,'saved_episode_phi0':outputs}
    result = analyze(arrays,weight,bias)
    assert all(c['passed'] for c in result['checks'].values())
    np.testing.assert_allclose(result['full_output']['outputs'],outputs,atol=1e-14,rtol=1e-12)
    for field,crossfield in [('total_energy','raw_cross_energy'),('centered_energy','centered_cross_energy')]:
        energy = sum(v[field] for v in result['components'].values())+sum(v[crossfield] for v in result['cross_terms'])
        np.testing.assert_allclose(energy,result['full_output'][field],atol=1e-18,rtol=1e-12)
    np.testing.assert_allclose(np.array(result['counterfactuals']['covariance_only']['outputs']),-.001*(h-mu)@c.T,atol=1e-14)


def test_feature_sensitivity_effective_rank_and_nearest_own_image():
    clean = np.array([[1.,0.],[2.,0.],[3.,0.],[4.,0.]])
    h = np.stack([x for row in clean for x in (row,row+np.array([.01,0.]))])
    result = feature_statistics(h)
    np.testing.assert_allclose([p['rho'] for p in result['pairs']],[.005,.01,.01,.005])
    np.testing.assert_allclose(result['median_rho'],.0075)
    np.testing.assert_allclose(result['participation_effective_rank'],1.)
    assert all(p['nearest_clean_is_own_image'] for p in result['pairs'])
    np.testing.assert_allclose(np.diag(result['euclidean_distances']),0.)
    np.testing.assert_allclose(np.diag(result['cosine_matrix']),1.)


def test_frozen_float32_roundoff_bound_and_triage_boundaries():
    saved = np.array([.1,.2])
    assert roundoff_check(saved+1e-9,saved)['passed']
    assert not roundoff_check(saved+1e-4,saved)['passed']
    assert roundoff_check(np.zeros(2),np.zeros(2))['passed']
    assert not roundoff_check(np.ones(2)*1e-15,np.zeros(2))['passed']
    assert triage(.099,.9,.9) == 'representation_insensitive_on_this_microset'
    assert triage(.10,.099,.099) == 'zero_head_common_gradient_collapse_despite_input_variation'
    assert triage(.10,.10,.01) == 'mixed_common_term_domination'
