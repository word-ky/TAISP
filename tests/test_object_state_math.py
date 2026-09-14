import numpy as np
from taisp.analysis.object_state_math import fit_representation,transform,fit_affine,tangent_target,corrected,evaluate


def test_pca_sign_and_train_only_standardization_and_rank_collapse():
    rng=np.random.default_rng(1033);d=rng.normal(size=(40,32));g=rng.normal(size=(40,8))
    counts=np.ones(40)*4;scores=np.ones(40)*.75
    rep=fit_representation(d,g,counts,scores);assert rep['pca_rank']>=16 and rep['pca_svd_calls']==1
    basis=np.asarray(rep['pca_basis']);ii=np.argmax(abs(basis),axis=1)
    assert (basis[np.arange(16),ii]>=0).all()
    z=transform(d,g,counts,scores,rep);assert z.shape==(40,27)
    np.testing.assert_allclose(z.mean(0),0,atol=1e-14)
    assert rep['constant_dimensions']==[25,26] and (z[:,25:]==0).all()
    hold=transform(d[:1]+100,g[:1],counts[:1]*2,scores[:1]*.5,rep)
    assert np.linalg.norm(hold)>1 and (hold[:,25:]==0).all()
    assert fit_representation(np.ones((40,32)),g,counts,scores)['status']=='FAIL_REPRESENTATION_COLLAPSE'


def test_affine_minimum_norm_one_svd_and_tangent_targets(monkeypatch):
    real=np.linalg.svd;calls=[]
    def svd(*a,**kw):calls.append(1);return real(*a,**kw)
    monkeypatch.setattr(np.linalg,'svd',svd)
    h=np.zeros((40,27));g=np.zeros((40,8));g[:,0]=1.;t=np.zeros((40,8));t[:,1]=1.
    model=fit_affine(h,g,t);assert calls==[1] and model['rank']==1 and model['condition_full'] is None
    np.testing.assert_allclose(model['B'],0,atol=1e-14)
    np.testing.assert_allclose(model['b'],[0,1/(1+1e-12)]+[0]*6,atol=1e-14)
    out=corrected(h,g,model);u=np.asarray(out['u_obj'])
    np.testing.assert_allclose(u[:,0],u[:,1],atol=1e-14)
    assert np.max(abs(np.asarray(out['projection_dot'])))<1e-11
    e=evaluate(t[0],g[0],u[0]);assert e['S_hard']==0 and e['S_obj']>.7


def test_zero_hard_or_task_kept_in_fit_and_hard_abstains():
    h=np.zeros((40,27));g=np.ones((40,8));t=g.copy();g[0]=0;t[1]=0
    target,zero=tangent_target(g,t);assert zero.sum()==2 and (target[:2]==0).all()
    m=fit_affine(h,g,t);assert m['train_count']==40 and m['zero_target_count']==2
    out=corrected(h,g,m);assert out['abstain'][0] and out['u_obj'][0]==[0.]*8
    e=evaluate(t[0],g[0],out['u_obj'][0]);assert e['S_obj']==e['S_hard']==e['Delta']==0
