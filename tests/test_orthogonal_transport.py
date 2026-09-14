import copy
import subprocess
import sys
import numpy as np
from taisp.analysis.orthogonal_transport import fit_vectors,evaluate,summarize,EPS


def test_procrustes_column_convention_reflection_and_single_svd(monkeypatch):
    real=np.linalg.svd;calls=[]
    def svd(*a,**kw):calls.append(1);return real(*a,**kw)
    monkeypatch.setattr(np.linalg,'svd',svd)
    h=np.eye(8);r=np.eye(8);r[0,0]=-1
    f=fit_vectors(h,h@r.T)
    np.testing.assert_allclose(f['R'],r,atol=1e-14)
    assert calls==[1] and f['determinant']<0 and f['orthogonality_max_abs']<1e-10
    e=evaluate([1.]*8,[2.]*8,f['R']);assert e['g_cal']==[-2.]+[2.]*7
    assert e['norm_hard']==e['norm_cal']


def test_zero_and_epsilon_units_retained():
    h=np.vstack([np.eye(8),np.zeros(8),np.ones(8)*EPS/4]);t=h.copy()
    f=fit_vectors(h,t);assert f['train_episode_count']==10 and f['zero_unit_hard']==2
    np.testing.assert_allclose(f['C'],np.eye(8),atol=0)
    e=evaluate([1.]*8,[0.]*8,f['R']);assert e['g_cal']==[0.]*8 and e['S_cal']==e['Delta']==0


def rows():
    return [{'case':'clean_s0' if i%2==0 else ['gamma_s2','contrast_s2','color_cast_s2'][(i//2)%3],
        'block':i//30,'S_hard':1.,'S_cal':2.,'Delta':1.,'cos_hard_task':.5,'cos_cal_task':.8,
        'norm_hard':1.,'norm_cal':1.,'g_hard':[1.]+[0.]*7,'g_cal':[0.,1.]+[0.]*6,
        'integrity_passed':True} for i in range(120)]


def test_mean_outliers_cannot_be_rescued_by_positive_medians():
    r=rows();assert summarize(r)['passed']
    r[1]['Delta']=-200
    s=summarize(r);assert s['gates']['median_overall'] and s['gates']['median_corrupt']
    assert not s['gates']['mean_overall'] and not s['gates']['mean_corrupt'] and not s['passed']
    assert s['overall']['coordinates']['g_cal']['max_energy_coordinate']==1
    assert not s['runtime_authorized']


def test_frozen_positive_count_thresholds_and_clean_requirement():
    r=rows()
    for x in r[:49]:x['Delta']=0
    assert not summarize(r)['gates']['Delta_overall']
    r=rows()
    for x in r[::2]:x['Delta']=-.1
    assert not summarize(r)['gates']['median_clean']
    r=rows();r[0]['integrity_passed']=False;assert not summarize(r)['passed']


def test_reference_import_does_not_load_oracle():
    code="import sys; import taisp.analysis.orthogonal_reference; assert 'taisp.analysis.oracle' not in sys.modules; assert 'taisp.analysis.common_jacobian' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
