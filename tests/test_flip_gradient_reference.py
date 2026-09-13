import math

import pytest

from taisp.analysis.flip_gradient_reference import spearman,ranks,describe,summarize


def test_spearman_average_ties_and_constant_undefined():
    assert ranks([1,2,2,4])==[0.,1.5,1.5,3.]
    assert spearman([1,2,2,4],[1,2,3,4])==pytest.approx(math.sqrt(.9))
    assert spearman([1,2,3],[3,2,1])==pytest.approx(-1)
    assert spearman([1,1],[1,2]) is None
    assert describe([0,1,2,3])['quantiles']['0.25']==.75


def test_reliability_only_does_not_rescue_consensus():
    rows=[{'image_id':i//2,'case':'gamma_s2' if i%2 else 'clean_s0','block':i//12,
        'S_orig':-1. if i<24 else 1.,'S_cons':-2.,'S_flip':0.,'Delta_cons':-3.,'agreement':i/47,
        'both_nonzero':True,'abstain':False,'integrity_passed':True} for i in range(48)]
    s=summarize(rows,.5)
    assert not s['E1_passed'] and s['E2_passed']
    assert s['disposition']=='reliability_signal_only_for_review'
    assert s['reliability']['corrupt']['high']['n']==12
    for row in rows[:5]:row['both_nonzero']=False
    assert not summarize(rows,.5)['E2_passed']


def test_median_ties_do_not_trigger_alternate_split():
    rows=[{'image_id':i//2,'case':'gamma_s2' if i%2 else 'clean_s0','block':i//12,
        'S_orig':1.,'S_cons':1.,'S_flip':1.,'Delta_cons':0.,'agreement':1.,
        'both_nonzero':True,'abstain':False,'integrity_passed':True} for i in range(48)]
    s=summarize(rows,1.)
    assert s['reliability']['overall']['low']['n']==0
    assert not s['E2_passed'] and not s['E1_passed']
