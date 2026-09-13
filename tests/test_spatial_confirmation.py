import copy
from taisp.analysis.spatial_confirmation import decision,json_hash


def test_prospective_gate_all_four_conditions_and_fixed_floor():
    rows=[dict(d_cur=1e-4,d_sp=1e-4) for _ in range(8)]
    assert decision(rows,True)['passed'] and not decision(rows,False)['passed']
    small=[dict(d_cur=0.,d_sp=1e-6) for _ in range(8)]
    assert decision(small,True)['passed']
    for mode in ['count','median','outlier']:
        x=copy.deepcopy(rows)
        if mode=='count':x[0]['d_sp']=x[1]['d_sp']=2.1e-4
        if mode=='median':
            for r in x:r['d_sp']=1.3e-4
        if mode=='outlier':x[0]['d_sp']=5.1e-4
        assert not decision(x,True)['passed']
    x=copy.deepcopy(rows);x[0]['d_sp']=4e-4
    assert decision(x,True)['passed']


def test_support_hash_order_and_content():
    assert json_hash({'a':[1],'b':[2]})==json_hash({'b':[2],'a':[1]})
    assert json_hash({'a':[1]})!=json_hash({'a':[2]})
