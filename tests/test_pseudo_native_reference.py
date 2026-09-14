from taisp.analysis.pseudo_native_reference import summarize


from taisp.analysis.native_task_signal import KEYS

def rows():
    return [{'case':'clean_s0' if i%2==0 else 'gamma_s2','block':i//30,'S_hard':1.,'S_native':2.,'Delta':1.,
             'norm_hard':1.,'norm_native':1.,'norm_ratio':1.,'gradient_cosine':1.,'integrity_passed':True,**{f'{prefix}_{k}':1. for k in KEYS for prefix in ['S','norm','cos_task']}} for i in range(120)]


def test_literal_conjunction_and_clean_nonnegative():
    r=rows();assert summarize(r)['passed']
    for v in r:
        if v['case']=='clean_s0':v['Delta']=-.1
    s=summarize(r)
    assert not s['passed'] and not s['gates']['median_clean']
    assert not s['runtime_authorized']


def test_zero_kept_and_failed_integrity_not_rescued():
    r=rows();r[0].update(norm_hard=0.,norm_native=0.,norm_ratio=None,gradient_cosine=None,S_hard=0.,S_native=0.,Delta=0.)
    s=summarize(r);assert s['overall']['n']==120 and s['overall']['zero_native']==1
    assert s['overall']['distributions']['S_native']['n']==120
    assert s['overall']['undefined_ratio']==1
    r[0]['integrity_passed']=False
    assert not summarize(r)['passed']


def test_diagnostic_failure_is_not_authoritative_integrity():
    from taisp.analysis.pseudo_native_reference import candidate_integrity,digest
    c={'isolation':True,'support_match':True,'target_match':True,'rng':{'restored':True},
       'component_sum':{'passed':False},'supports':{'boxes':[],'labels':[]},'pseudo_targets':{'boxes':[],'labels':[]},
       'objectives':{'native':{'gradient':[0.]*8,'loss':0.,'parity':{'passed':True,'relative_l2':0.,'cosine':1.}}}}
    c['supports_sha256']=digest(c['supports'])
    assert candidate_integrity(c)
    c['objectives']['native']['gradient'][0]=float('nan');assert not candidate_integrity(c)
    c['objectives']['native']['gradient'][0]=0.;c['rng']['restored']=False;assert not candidate_integrity(c)


def test_invalid_lock_stops_before_oracle_import(tmp_path):
    import subprocess,sys
    lock=tmp_path/'lock.json';lock.write_text('{"reviewed_commit":"invalid"}')
    code=f"""from pathlib import Path
import sys
from taisp.analysis.pseudo_native_reference import verify_inputs
try: verify_inputs(Path('absent'),Path('absent'),Path({str(lock)!r}))
except AssertionError: pass
else: raise AssertionError('bad lock accepted')
assert 'taisp.analysis.oracle' not in sys.modules
assert 'taisp.analysis.common_jacobian' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_score_definition_and_eight_original_conditions():
    from taisp.analysis.pseudo_native_reference import score
    assert score([3.,4.],[0.,0.])==0.
    assert score([3.,4.],[0.,2.])==8./(2.+1e-12)
    r=rows()
    for v in r[:41]:v['S_native']=0.
    assert not summarize(r)['gates']['S_native_overall']
    r=rows()
    for v in r[:53]:v['Delta']=0.
    assert not summarize(r)['gates']['Delta_overall']
