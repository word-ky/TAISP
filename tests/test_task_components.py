import copy

import numpy as np
import torch

from taisp import DifferentiableISP
from taisp.analysis.task_components import KEYS, attribution, component_reference, numerical_checks, sum_check, triage


def test_single_forward_common_jvp_component_and_total_closure():
    isp=DifferentiableISP().double()
    x=torch.full((1,3,4,5),.4,dtype=torch.double)
    mask=torch.zeros(1,1,4,5,dtype=torch.double)
    mask[:,:,:,:3]=1
    calls=[]
    def loss(y):
        calls.append(1)
        parts=dict(zip(KEYS,[(y*y).sum(),-y.sum(),(y.sin()).sum(),2*y.sum()]))
        return sum(parts.values()),parts
    r=component_reference(x,isp,mask,loss)
    assert len(calls)==1 and r['jacobian']['columns']==8 and isp.phi.grad is None
    checks=numerical_checks(r['references'],r['direct_global'],r['references']['total'])
    assert checks['passed']
    assert not sum_check([[1.,0.],[0.,1.]], [1.+1e-7,1.])['passed']


def test_attribution_negative_cancellation_groups_and_productivity():
    ds=[np.array([3.,0.]),np.array([-1.,0.]),np.array([0.,2.]),np.array([0.,-1.])]
    refs={k:{'object':d.tolist(),'background':(-d).tolist(),'global':[0.,0.]} for k,d in zip(KEYS,ds)}
    dp=np.array([1.,-1.]); pn=np.sqrt(4.)
    saved=2*np.dot(sum(ds),dp)/(pn+1e-12)
    p={'pseudo':{'d':dp.tolist(),'norm':pn},'metrics':{'C_diff':saved}}
    a=attribution(refs,p)
    assert a['passed']
    np.testing.assert_allclose(sum(a['metrics'][k+'/A'] for k in KEYS),1.,atol=1e-12)
    assert a['metrics']['loss_box_reg/A']<0 and a['metrics']['loc/A']<0
    np.testing.assert_allclose(a['metrics']['loc/A']+a['metrics']['conf/A'],1.,atol=1e-12)
    np.testing.assert_allclose(sum(a['metrics'][k+'/C_diff'] for k in KEYS),saved,atol=1e-12)


def test_zero_total_differential_is_explicit_not_favorable():
    z=np.zeros(2)
    refs={k:{'object':z.tolist(),'background':z.tolist(),'global':z.tolist()} for k in KEYS}
    a=attribution(refs,{'pseudo':{'d':[1.,0.],'norm':np.sqrt(2)},'metrics':{'C_diff':0.}})
    assert a['passed'] and a['zero_norm']['total_diff']
    assert a['EPS_attribution_deficit']==1
    assert all(a['metrics'][k+'/A']==0 for k in KEYS)
    assert a['metrics']['loc/pseudo_cosine'] is None


def test_dominance_and_utility_literal_thresholds():
    scopes={name:{'metrics':{g+'/A':{'median':.6 if g=='loc' else .4} for g in ('loc','conf')}}
            for name in ('overall','corrupted','block0','block1','block2','block3')}
    for name,s in scopes.items():
        s['metrics'].update({g+'/C_diff':{'positive_count':10 if name=='corrupted' else 20,'median':.01}
                             for g in ('loc','conf')})
    scopes['block3']['metrics']['loc/A']['median']=.4
    assert triage(scopes)['dominant']=='loc' and triage(scopes)['dominant_pseudo_useful']
    for group in ('overall','corrupted','block2'):
        changed=copy.deepcopy(scopes)
        changed[group]['metrics']['loc/A']['median']=.5
        assert triage(changed)['dominant'] is None
    for group,key,value in [('overall','positive_count',19),('corrupted','positive_count',9),('overall','median',0.)]:
        changed=copy.deepcopy(scopes)
        changed[group]['metrics']['loc/C_diff'][key]=value
        assert not triage(changed)['dominant_pseudo_useful']
