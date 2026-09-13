import copy

import numpy as np
import torch

from taisp import DifferentiableISP
from taisp.analysis.common_jacobian import common_reference, numerical_pass, reference_checks
from taisp.analysis.spatial_action import identity_gradients, support_mask


def test_common_jvp_reconstructs_reverse_global_with_independent_partition():
    torch.manual_seed(41)
    isp = DifferentiableISP().double()
    image = .2+.5*torch.rand(1,3,5,6,dtype=torch.double)
    weights = torch.randn_like(image)
    mask,_ = support_mask(image,torch.tensor([[1.,1.,4.,5.]]))
    reverse,c = identity_gradients(image,isp,mask,lambda y:((y.sin()*weights).sum(),{}),return_cotangent=True)
    reference,info = common_reference(image,isp,mask,{'task':c})
    result = reference['task']
    np.testing.assert_allclose(result['global'],reverse['global'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result['object'],reverse['object'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result['background'],reverse['background'],rtol=1e-10,atol=1e-12)
    checks = reference_checks(result['global'],result['object'],result['background'],reverse['global'])
    assert checks['partition']['passed'] and checks['parity']['passed']
    assert info['columns']==8 and info['reduction_dtype']=='torch.float64'


def test_float32_jvp_float64_reductions_and_multiple_cotangents():
    torch.manual_seed(44)
    isp = DifferentiableISP()
    image = .2+.5*torch.rand(1,3,16,20)
    mask,_ = support_mask(image,torch.tensor([[0.,0.,10.,16.]]))
    c = torch.randn_like(image)
    phi = torch.zeros(8,requires_grad=True)
    direct = torch.autograd.grad((isp(image,phi)*c).sum(),phi)[0].tolist()
    refs,_ = common_reference(image,isp,mask,{'one':c,'negative':-c})
    for region in ('global','object','background'):
        np.testing.assert_allclose(refs['negative'][region],-np.array(refs['one'][region]),atol=1e-12)
    r = refs['one']
    checks = reference_checks(r['global'],r['object'],r['background'],direct)
    assert checks['partition']['passed'] and checks['parity']['passed']


def test_empty_full_masks_zero_cotangent_and_detached_state():
    isp = DifferentiableISP().double()
    original = {k:v.clone() for k,v in isp.state_dict().items()}
    image = torch.full((1,3,4,5),.4,dtype=torch.double,requires_grad=True)
    c = torch.ones_like(image,requires_grad=True)
    for value,empty in [(0.,'object'),(1.,'background')]:
        mask = torch.full((1,1,4,5),value,dtype=torch.double,requires_grad=True)
        refs,_ = common_reference(image,isp,mask,{'nonzero':c,'zero':torch.zeros_like(c)})
        r = refs['nonzero']
        np.testing.assert_array_equal(r[empty],np.zeros(8))
        other = 'background' if empty=='object' else 'object'
        np.testing.assert_array_equal(r[other],r['global'])
        for region in ('global','object','background'):
            np.testing.assert_array_equal(refs['zero'][region],np.zeros(8))
        assert mask.grad is None
    assert image.grad is None and c.grad is None and isp.phi.grad is None
    assert all(torch.equal(v,isp.state_dict()[k]) for k,v in original.items())


def test_closure_parity_and_zero_vector_conventions():
    a=np.array([1.,0.])
    good=reference_checks(a,.8*a,.2*a,np.array([1.,1e-5]))
    assert good['partition']['passed'] and good['parity']['passed']
    bad=reference_checks(a,.8*a,.2*a,np.array([1.,1.001e-5]))
    assert not bad['parity']['passed']
    assert not reference_checks(a+1e-8,.8*a,.2*a,a)['partition']['passed']
    zeros=np.zeros(2)
    assert reference_checks(zeros,zeros,zeros,zeros)['parity']['passed']
    assert not reference_checks(a,a,zeros,zeros)['parity']['passed']


def test_old_failed_bound_is_diagnostic_but_new_checks_are_required():
    objective={'reverse':{'checks':{'identity_image':{'passed':True},'regional_global_output':{'passed':True},
                                    'regional_gradient_sum':{'passed':False}}},
               'reference_checks':{'partition':{'passed':True},'parity':{'passed':True}}}
    row={'task':copy.deepcopy(objective),'pseudo':copy.deepcopy(objective),'isolation':{'frozen':True},
         'jacobian':{'primal_identity_checks':[{'passed':True}]*8}}
    assert numerical_pass(row)
    row['pseudo']['reference_checks']['partition']['passed']=False
    assert not numerical_pass(row)
