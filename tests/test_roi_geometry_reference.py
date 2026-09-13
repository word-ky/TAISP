import copy

from taisp.analysis.roi_geometry_reference import aggregate, NAME


def rows():
    m={'S_geom':1.,'Delta_global':1.,'Delta_obj':0.,'zero_gradient':False,
       'near_zero_gradient':False,'material_near_zero':False,
       'roi_box_diagnostic':{'score':1.},'localization_diagnostic':{'score':1.}}
    return [{'image_id':i//2,'case':'gamma_s2' if i%2 else 'clean_s0','block':i//12,
             'integrity_passed':True,'metrics':{NAME:copy.deepcopy(m)}} for i in range(48)]


def test_geometry_conjunction_and_diagnostic_does_not_promote():
    r=rows()
    assert aggregate(r)['nomination']==NAME
    for i in range(1,48,2):r[i]['metrics'][NAME]['Delta_global']=-.1
    s=aggregate(r)
    assert not s['passed'] and not s['gate']['Delta_corrupt']
    assert s['overall']['roi_box_positive']==48 and s['nomination'] is None


def test_systematic_near_zero_uses_predeclared_distinct_images():
    r=rows()
    for i in [0,1]:r[i]['metrics'][NAME]['material_near_zero']=True
    assert aggregate(r)['gate']['no_systematic_near_zero']
    r[2]['metrics'][NAME]['material_near_zero']=True
    assert not aggregate(r)['gate']['no_systematic_near_zero']


def test_overall_thirty_and_corrupt_fifteen_are_both_required():
    r=rows()
    for i in range(18):r[i]['metrics'][NAME]['S_geom']=-1.
    assert aggregate(r)['gate']['S_overall'] and aggregate(r)['gate']['S_corrupt']
    r[18]['metrics'][NAME]['S_geom']=-1.
    assert not aggregate(r)['gate']['S_overall']
