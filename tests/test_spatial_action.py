import copy
import math

import numpy as np
import torch

from taisp import DifferentiableISP
from taisp.analysis.spatial_action import compose, gate, geometry, identity_gradients, support_mask


def test_mask_union_clipping_fractional_rasterization_and_detachment():
    image = torch.zeros(1,3,5,6,dtype=torch.double)
    boxes = torch.tensor([[-1.,1.2,2.2,3.1],[1.,2.,8.,4.]],requires_grad=True)
    mask,rectangles = support_mask(image,boxes)
    expected = torch.zeros_like(mask)
    expected[:,:,1:4,0:3]=1
    expected[:,:,2:4,1:6]=1
    torch.testing.assert_close(mask,expected,rtol=0,atol=0)
    assert rectangles==[[0,1,3,4],[1,2,6,4]] and not mask.requires_grad
    assert boxes.grad is None


def test_identity_shared_state_and_all_regional_gradients():
    torch.manual_seed(8)
    isp = DifferentiableISP().double()
    image = .2+.5*torch.rand(1,3,5,6,dtype=torch.double)
    mask,_ = support_mask(image,torch.tensor([[1.,1.,4.,4.]]))
    zero = torch.zeros(8,dtype=torch.double)
    torch.testing.assert_close(compose(image,isp,mask,zero,zero),image,atol=2e-7,rtol=1e-6)
    phi = torch.full((8,),.03,dtype=torch.double,requires_grad=True)
    weights = torch.randn_like(image)
    shared = compose(image,isp,mask,phi,phi)
    direct = isp(image,phi)
    torch.testing.assert_close(shared,direct,rtol=0,atol=0)
    a = torch.autograd.grad((shared*weights).sum(),phi)[0]
    b = torch.autograd.grad((direct*weights).sum(),phi)[0]
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
    obj,bg = phi.detach().clone().requires_grad_(),(-phi.detach()).requires_grad_()
    assert torch.autograd.gradcheck(lambda p,q:compose(image,isp,mask,p,q),(obj,bg))


def test_common_cotangent_equals_direct_regional_objective_gradient():
    torch.manual_seed(31)
    isp = DifferentiableISP().double()
    image = .2+.5*torch.rand(1,3,5,6,dtype=torch.double)
    mask,_ = support_mask(image,torch.tensor([[1.,0.,3.,5.]]))
    weights = torch.randn_like(image)
    objective = lambda y:((y.sin()*weights).sum(),{})
    result = identity_gradients(image,isp,mask,objective)
    obj,bg = torch.zeros(8,dtype=torch.double,requires_grad=True),torch.zeros(8,dtype=torch.double,requires_grad=True)
    regional = compose(image,isp,mask,obj,bg)
    go,gb = torch.autograd.grad(objective(regional)[0],(obj,bg))
    torch.testing.assert_close(torch.tensor(result['object'],dtype=torch.double),go,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(torch.tensor(result['background'],dtype=torch.double),gb,atol=1e-12,rtol=1e-10)
    assert all(c['passed'] for c in result['checks'].values())
    assert isp.phi.grad is None


def test_empty_full_masks_and_empty_pseudo_keep_exact_zero_region():
    isp = DifferentiableISP().double()
    image = torch.full((1,3,4,5),.4,dtype=torch.double)
    for boxes,zero_region in [(torch.empty(0,4),'object'),(torch.tensor([[0.,0.,5.,4.]]),'background')]:
        mask,_ = support_mask(image,boxes)
        result = identity_gradients(image,isp,mask,lambda y:(y.square().sum(),{}))
        np.testing.assert_array_equal(result[zero_region],np.zeros(8))
        other = 'background' if zero_region=='object' else 'object'
        np.testing.assert_allclose(result[other],result['global'],atol=1e-12,rtol=1e-10)
        empty = identity_gradients(image,isp,mask,lambda y:(y.sum()*0,{}))
        for name in ('object','background','global'):
            np.testing.assert_array_equal(empty[name],np.zeros(8))


def test_equal_budget_geometry_shared_opposing_and_empty_vectors():
    a = np.array([1.,0.])
    shared = geometry(a,a,a,a)
    np.testing.assert_allclose(shared['R_task'],1.,atol=1e-12)
    np.testing.assert_allclose(shared['Delta_D'],0.,atol=1e-12)
    opposing = geometry(a,-a,a,-a)
    assert opposing['task_regional_cosine']==-1 and opposing['task_cancellation']==1
    assert opposing['A_spatial']>0 and opposing['A_global']==0 and opposing['Delta_D']>1.4
    np.testing.assert_allclose(opposing['D_spatial'],math.sqrt(2),atol=1e-12)
    empty = geometry(a,-a,np.zeros(2),np.zeros(2))
    assert empty['zero_pseudo_regions'] and empty['D_global']==empty['D_spatial']==0
    degenerate = geometry(a,np.zeros(2),a,np.zeros(2))
    np.testing.assert_allclose(degenerate['R_task'],math.sqrt(2),atol=2e-12)


def test_exact_spatial_advancement_gate_boundaries():
    scopes = {'overall':{'metrics':{'R_task':{'median':1.20},'Delta_D':{'positive_count':20,'median':.01}}},
              'corrupted':{'metrics':{'R_task':{'median':1.15},'Delta_D':{'positive_count':10,'median':.01}}}}
    scopes.update({f'block{i}':{'metrics':{'Delta_D':{'median':.01 if i<3 else 0.}}} for i in range(4)})
    assert gate(scopes)['passed']
    changes = [('overall','R_task','median',1.199),('corrupted','R_task','median',1.149),
               ('overall','Delta_D','positive_count',19),('corrupted','Delta_D','positive_count',9),
               ('overall','Delta_D','median',0.),('block2','Delta_D','median',0.)]
    for group,metric,key,value in changes:
        modified = copy.deepcopy(scopes)
        modified[group]['metrics'][metric][key]=value
        assert not gate(modified)['passed']
