import copy

from taisp.analysis.source_roi_memory_reference import describe,aggregate,NAME


def test_quantiles_and_empty_diagnostics():
    d=describe([0.,1.,2.,3.])
    assert d['quantiles']['0.25']==.75 and d['quantiles']['1.0']==3.
    assert d['positive_fraction']==.75 and describe([])['mean'] is None


def rows():
    m={'S_mem':1.,'Delta_global':1.,'Delta_obj':1.,'norms':{'candidate':1.},
       'zero_gradient':False,'near_zero_gradient':False,'material_near_zero':False}
    retrieval={'entry_ids':[0,1,2,3],'classes':[1],'cosines':[.5]*4,
        'unique_anchor_hashes':1,'anchor_pairwise_mean_cosine':None,'anchor_pairwise_min_cosine':None}
    return [{'image_id':i//2,'case':'gamma_s2' if i%2 else 'clean_s0','block':i//12,
        'integrity_passed':True,'memory_health_passed':True,'metrics':{NAME:copy.deepcopy(m)},
        'retrieval':copy.deepcopy(retrieval)} for i in range(48)]


def test_gate_requires_memory_health_and_fulltask_corruption():
    r=rows();assert aggregate(r)['passed']
    r[0]['memory_health_passed']=False;assert not aggregate(r)['passed']
    r[0]['memory_health_passed']=True
    for i in range(1,48,2):r[i]['metrics'][NAME]['Delta_global']=-.1
    s=aggregate(r)
    assert not s['passed'] and s['overall']['distinct_memory_entries']==4
    assert s['overall']['distributions']['Delta_obj']['positive']==48
