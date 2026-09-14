import subprocess
import sys
import numpy as np
from taisp.analysis.object_state_math import summary


def rows():
    return [dict(case='clean_s0' if i%2==0 else ['gamma_s2','contrast_s2','color_cast_s2'][(i//2)%3],
        block=i//30,S_hard=1.,S_obj=2.,Delta=1.,cos_hard_task=.5,cos_obj_task=.8,
        norm_hard=1.,norm_obj=1.,g_hard=[1.]+[0.]*7,u_obj=[0.,1.]+[0.]*6,
        abstain=False,support_count=3,integrity_passed=True) for i in range(120)]


def test_r051_82_threshold_and_empty_abstention_counts():
    r=rows();assert summary(r)['passed']
    for x in r[:38]:x.update(S_obj=0.,S_hard=-1.)
    assert summary(r)['gates']['S_obj_overall']
    r[38].update(S_obj=0.,S_hard=-1.);assert not summary(r)['gates']['S_obj_overall']
    r=rows();r[0].update(g_hard=[0.]*8,u_obj=[0.]*8,norm_hard=0.,norm_obj=0.,S_hard=0.,S_obj=0.,Delta=0.,
        cos_hard_task=None,cos_obj_task=None,abstain=True,support_count=0)
    s=summary(r);assert s['overall']['n']==120 and s['abstentions']==1
    assert s['support_subsets']['empty']['n']==1 and s['gates']['finite_zero_integrity']
    r[1]['Delta']=-500;assert not summary(r)['gates']['mean_corrupt']


def test_stage_and_reference_lazy_oracle_boundary():
    for module in ['object_state_stages','object_state_reference']:
        code=f"import sys; import taisp.analysis.{module}; assert 'taisp.analysis.oracle' not in sys.modules; assert 'taisp.analysis.common_jacobian' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
