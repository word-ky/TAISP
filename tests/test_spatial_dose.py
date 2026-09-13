import inspect

import pytest
import torch
from torch import nn

from taisp import DifferentiableISP
from taisp.isp.spatial import compose,regional_gradients,support_mask
from taisp.tta.spatial_dose import adapt_spatial_dose,dose_step
from taisp.tta.trust_radius import adapt_clip_radius,transfer_norm


def test_mask_and_composition_equal_existing_donor():
    from taisp.analysis.spatial_action import support_mask as old_mask,compose as old_compose
    image=torch.rand(1,3,7,8);boxes=torch.tensor([[-2.,1.2,3.1,6.5],[2.,2.,10.,9.]])
    mask,rects=support_mask(image,boxes);old,r=old_mask(image,boxes)
    assert torch.equal(mask,old) and rects==r
    isp=DifferentiableISP();a=torch.randn(8)*.01;b=-a
    assert torch.equal(compose(image,isp,mask,a,b),old_compose(image,isp,old,a,b))


def test_common_gradient_chain_rule_at_nonzero_states_and_inactive_regions():
    torch.manual_seed(3);isp=DifferentiableISP().double();x=.2+.5*torch.rand(1,3,5,6,dtype=torch.double)
    mask,_=support_mask(x,torch.tensor([[1.,1.,4.,4.]]));state=torch.full((8,),.01,dtype=torch.double)
    cotangent=torch.randn_like(x)
    regional=regional_gradients(x,isp,mask,torch.stack((state,state)),{'loss':cotangent})['loss']
    p=state.clone().requires_grad_();direct=torch.autograd.grad(isp(x,p),p,grad_outputs=cotangent)[0]
    torch.testing.assert_close(regional.sum(0),direct,rtol=1e-10,atol=1e-12)
    for m,zero_region in [(torch.zeros_like(mask),0),(torch.ones_like(mask),1)]:
        g=regional_gradients(x,isp,m,torch.stack((state,state)),{'loss':cotangent})['loss']
        assert torch.equal(g[zero_region],torch.zeros(8,dtype=torch.double))
        torch.testing.assert_close(g[1-zero_region],direct,rtol=1e-10,atol=1e-12)


def test_zero_contrast_exact_global_update_and_zero_common_no_fallback():
    g=torch.arange(1.,9.,dtype=torch.double);c=g*.7
    delta,d=dose_step(g/2,g/2,c/2,c/2)
    update,_=transfer_norm(g,c)
    torch.testing.assert_close(delta,(-.1*update).repeat(2,1),rtol=1e-12,atol=1e-12)
    assert d['c']==0 and d['multipliers']==[1,1]
    delta,d=dose_step(g,-g,c,c)
    assert torch.equal(delta,torch.zeros_like(delta)) and d['zero_common_pseudo']


def test_shared_direction_and_bounded_mean_one_doses():
    torch.manual_seed(8)
    for _ in range(10):
        go,gb,co,cb=torch.randn(4,8)
        delta,d=dose_step(go,gb,co,cb)
        assert -1<=d['c']<=1 and all(.5<=v<=1.5 for v in d['multipliers'])
        for i in range(2):torch.testing.assert_close(delta[i],torch.tensor(d['base_delta'])*d['multipliers'][i])
        torch.testing.assert_close(delta.mean(0),torch.tensor(d['base_delta']))


def test_frozen_vector_parity_boundaries_and_zero_convention():
    from taisp.analysis.spatial_dose_parity import vector_parity
    z=torch.zeros(8,dtype=torch.double);a=torch.ones(8,dtype=torch.double)
    assert vector_parity(z,z)['passed']
    assert not vector_parity(a,z)['passed']
    assert vector_parity(a*1.000009,a)['passed']
    assert not vector_parity(a*1.00002,a)['passed']


class UniformLoss(nn.Module):
    def __init__(self,empty=False):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.2,dtype=torch.double));self.boxes=torch.ones(0 if empty else 1,4)
    def forward(self,original,enhanced):
        return enhanced.sum()*0 if not len(self.boxes) else (enhanced-self.weight).square().mean()


@pytest.mark.parametrize('empty',[False,True])
def test_equal_state_c0_global_k3_and_reset_frozen_state_no_labels(empty):
    x=torch.full((1,3,4,6),.4,dtype=torch.double);isp=DifferentiableISP().double()
    mask=torch.zeros(1,1,4,6,dtype=torch.double);mask[:,:,:,:3]=1
    pseudo,clip=UniformLoss(empty),UniformLoss();old=pseudo.weight.detach().clone()
    a=adapt_spatial_dose(x,isp,pseudo,clip,mask);b=adapt_clip_radius(x,isp,pseudo,clip)
    torch.testing.assert_close(a.phi[0],b.phi,rtol=1e-8,atol=1e-10)
    torch.testing.assert_close(a.phi[1],b.phi,rtol=1e-8,atol=1e-10)
    torch.testing.assert_close(a.enhanced,b.enhanced,rtol=1e-8,atol=1e-10)
    repeat=adapt_spatial_dose(x,isp,pseudo,clip,mask)
    assert torch.equal(a.phi,repeat.phi) and torch.equal(isp.phi,torch.zeros(8,dtype=torch.double))
    assert torch.equal(pseudo.weight,old) and pseudo.weight.grad is None and not pseudo.weight.requires_grad
    assert clip.weight.grad is None and not clip.weight.requires_grad
    assert a.phi0.shape==(2,8) and torch.count_nonzero(a.phi0)==0
    if empty:assert torch.count_nonzero(a.phi)==0
    assert set(inspect.signature(adapt_spatial_dose).parameters)=={'image','isp','detector_loss','clip_loss','mask','steps','lr','eps'}


def test_parity_entry_initializes_clip_and_reports_separate_isolation(monkeypatch,tmp_path):
    from taisp.analysis import spatial_dose_parity as parity
    wrapper=nn.Sequential(nn.Linear(2,2),nn.Dropout())
    initial=parity.state_hash(wrapper)
    monkeypatch.setattr(parity,'load_detector',lambda device: nn.Identity())
    monkeypatch.setattr(parity,'load_clip_guidance',lambda *a,**kw: wrapper)
    class InitializationObserved(Exception):
        pass
    def after_clip():
        assert parity.clip_isolation(wrapper,initial)==dict(
            clip_parameters_frozen_grad_none=True,clip_all_modules_eval=True,
            clip_state_hash_unchanged=True)
        raise InitializationObserved
    monkeypatch.setattr(parity,'DifferentiableISP',after_clip)
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    with pytest.raises(InitializationObserved):
        parity.collect(dict(seed=1,threads=1,device='cpu'),manifest,tmp_path/'out')
    wrapper.train()
    assert not parity.clip_isolation(wrapper,initial)['clip_all_modules_eval']
    wrapper.eval()
    next(wrapper.parameters()).grad=torch.ones_like(next(wrapper.parameters()))
    assert not parity.clip_isolation(wrapper,initial)['clip_parameters_frozen_grad_none']
    next(wrapper.parameters()).grad=None
    with torch.no_grad():next(wrapper.parameters()).add_(1)
    assert not parity.clip_isolation(wrapper,initial)['clip_state_hash_unchanged']
