import subprocess
import sys
from taisp.analysis.gradient_regime_reference import summary


def rows():
    return [dict(case='clean_s0' if i%2==0 else ['gamma_s2','contrast_s2','color_cast_s2'][(i//2)%3],
                 block=i//30,route=i%2,S_hard=1.,S_cal=2.,S_route=2.,Delta=1.,
                 cos_hard_task=.5,cos_cal_task=.8,norm_hard=1.,norm_cal=1.,
                 g_hard=[1.]+[0.]*7,g_cal=[0.,1.]+[0.]*6,g_route=[0.,1.]+[0.]*6,
                 integrity_passed=True) for i in range(120)]


def test_coverage_threshold_and_cluster_diagnostics():
    r=rows();s=summary(r);assert s['passed'] and s['route_counts']['overall']==[60,60]
    assert s['clusters']['1']['coordinates']['g_route']['max_energy_coordinate']==1
    for i,x in enumerate(r):x['route']=int(i<12)
    assert summary(r)['passed']
    r[11]['route']=0;assert not summary(r)['gates']['holdout_route_coverage']
    for x in r:x['route']=0
    s=summary(r);assert not s['passed'] and s['clusters']['1']['n']==0


def test_original_means_and_counts_remain_gating():
    r=rows();r[1]['Delta']=-500
    s=summary(r);assert s['gates']['median_corrupt'] and not s['gates']['mean_corrupt'] and not s['passed']
    r=rows()
    for x in r[:41]:x['S_cal']=x['S_route']=0
    assert not summary(r)['gates']['S_route_overall']


def test_reference_import_does_not_reveal_oracle():
    code="import sys; import taisp.analysis.gradient_regime_reference; assert 'taisp.analysis.oracle' not in sys.modules; assert 'taisp.analysis.common_jacobian' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
