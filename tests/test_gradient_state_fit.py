import numpy as np
import pytest
from taisp.analysis.gradient_state_fit import (design_fit, transform, pair_permutations,
    auc, average_ranks, split_indices, metrics)


def test_train_only_population_standardization_and_zero_column():
    train=np.array([[1.,7.],[3.,7.],[5.,7.]])
    f=design_fit(train);z=transform([[100.,999.]],f)
    assert f['mean'].tolist()==[3.,7.]
    assert f['std'][0]==pytest.approx(np.sqrt(8/3))
    assert z[0].tolist()==pytest.approx([97/np.sqrt(8/3),0,1])
    assert f['mean'].tolist()==[3.,7.]


def test_svd_tolerance_rank_reconstruction_and_affine_fit():
    x=np.array([[i,2*i,1.] for i in range(6)],dtype=float);f=design_fit(x)
    assert f['tol']==np.finfo(np.float64).eps*max(f['z'].shape)*f['singular_values'][0]
    assert f['rank']==2
    np.testing.assert_allclose(f['z']@f['pinv']@f['z'],f['z'],atol=1e-14)
    target=3*x[:,0]-2;w=f['pinv']@target
    np.testing.assert_allclose(transform([[8,16,1]],f)@w,[22],atol=1e-13)


def test_128_nulls_keep_pairs_and_family_deterministic():
    families=['gamma','contrast','gamma','contrast','color','color']
    pairs=np.arange(12).reshape(6,2)
    a=list(pair_permutations(families));b=list(pair_permutations(families))
    assert len(a)==128
    for p,q in zip(a,b):
        assert np.array_equal(p,q) and sorted(p)==list(range(6))
        assert all(families[i]==families[j] for i,j in enumerate(p))
        shuffled=pairs[p].reshape(-1,2)
        assert np.all(shuffled[:,1]-shuffled[:,0]==1)
    assert any(not np.array_equal(p,np.arange(6)) for p in a)


def test_auc_average_ties_and_undefined():
    assert average_ranks([1,2,2,4]).tolist()==[1,2.5,2.5,4]
    assert auc([-1,1,-1,1],[0,1,1,2])==.875
    assert auc([-1,1],[1,1])==.5
    assert auc([1,1],[1,2]) is None


def test_split_keeps_images_disjoint():
    rows=[{'image_id':1,'partition':'train'},{'image_id':1,'partition':'train'},
          {'image_id':2,'partition':'holdout'},{'image_id':2,'partition':'holdout'}]
    assert split_indices(rows)==([0,1],[2,3])
    rows[2]['image_id']=1
    with pytest.raises(AssertionError):split_indices(rows)


def test_zero_threshold_and_precision_gain():
    rows=[{'S_orig':s,'y':1 if s>0 else -1,'score':r,'norm_pseudo':1,'norm_clip':1,'integrity_passed':True}
          for s,r in [(1,2),(-1,0),(2,-1),(-2,-2)]]
    m=metrics(rows)
    assert m['trusted_coverage']==.25 and m['precision_gain']==.5
    assert m['untrusted']['n']==3
