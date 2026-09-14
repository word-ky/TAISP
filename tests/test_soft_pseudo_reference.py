from taisp.analysis.soft_pseudo_reference import summarize


def rows():
    return [{'case':'clean_s0' if i%2==0 else 'gamma_s2','block':i//30,'S_hard':1.,'S_soft':2.,'Delta':1.,
             'norm_hard':1.,'norm_soft':1.,'norm_ratio':1.,'gradient_cosine':1.,'integrity_passed':True} for i in range(120)]


def test_literal_conjunction_and_clean_nonnegative():
    r=rows();assert summarize(r)['passed']
    for v in r:
        if v['case']=='clean_s0':v['Delta']=-.1
    s=summarize(r)
    assert not s['passed'] and not s['gates']['median_clean']
    assert not s['runtime_authorized']


def test_zero_kept_and_failed_integrity_not_rescued():
    r=rows();r[0].update(norm_hard=0.,norm_soft=0.,norm_ratio=None,gradient_cosine=None,S_hard=0.,S_soft=0.,Delta=0.)
    s=summarize(r);assert s['overall']['n']==120 and s['overall']['zero_soft']==1
    assert s['overall']['distributions']['S_soft']['n']==120
    assert s['overall']['undefined_ratio']==1
    r[0]['integrity_passed']=False
    assert not summarize(r)['passed']
