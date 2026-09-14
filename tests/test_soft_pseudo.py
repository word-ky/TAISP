import subprocess
import sys
import torch
import pytest
from taisp import DifferentiableISP
from taisp.losses import detector_native as dn
from taisp.analysis import soft_pseudo as sp
from taisp.analysis.soft_pseudo_candidates import gradients


def test_power2_full91_background_and_detach():
    z=torch.linspace(-3,3,182,dtype=torch.double).reshape(2,91).requires_grad_(True)
    q=sp.power2_target(z);p=z.softmax(-1);expected=p.square()/p.square().sum(-1,keepdim=True)
    torch.testing.assert_close(q,expected)
    assert q.shape==(2,91) and not q.requires_grad and bool((q[:,0]>0).all())
    torch.testing.assert_close(q.sum(-1),torch.ones(2,dtype=torch.double))
    extreme=sp.power2_target(torch.tensor([[10000.,-10000.]+[0.]*89]))
    assert torch.isfinite(extreme).all() and extreme.sum()==1


def make_pair(monkeypatch,empty=False):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    image=(.2+.4*torch.rand(1,3,20,24,device=device))
    boxes=image.new_tensor([[0,0,12,14],[3,2,20,19]]).reshape(2,4)[:0 if empty else 2]
    support={'boxes':boxes,'labels':torch.tensor([4,9],device=device)[:len(boxes)],
             'scores':image.new_tensor([.8,.6])[:len(boxes)]}
    def logits(detector,x,b):
        matrix=torch.arange(273,device=x.device,dtype=x.dtype).reshape(3,91).sin()
        return (x.mean((-2,-1))@matrix).expand(len(b),91)
    monkeypatch.setattr(sp,'fixed_roi_logits',logits);monkeypatch.setattr(dn,'fixed_roi_logits',logits)
    hard=dn.DetectorNativeLoss(torch.nn.Identity(),{'base':support},'det_pseudo')
    soft=sp.SoftPseudoLoss(hard,image)
    return image,hard,soft


def test_identical_support_weights_and_natural_soft_ce(monkeypatch):
    image,hard,soft=make_pair(monkeypatch)
    assert soft.boxes is hard.boxes and soft.weights is hard.weights
    scores=image.new_tensor([.8,.6])
    assert torch.equal(soft.weights,scores/scores.sum())
    z=sp.fixed_roi_logits(None,image,soft.boxes)
    expected=(soft.weights*(-soft.target*z.log_softmax(-1)).sum(-1)).sum()
    torch.testing.assert_close(soft(image,image),expected)
    assert not soft.target.requires_grad


def test_empty_support_differentiable_zero(monkeypatch):
    image,hard,soft=make_pair(monkeypatch,True);y=image.detach().requires_grad_(True)
    for loss in [hard(image,y),soft(image,y)]:
        assert loss.item()==0 and torch.count_nonzero(torch.autograd.grad(loss,y)[0])==0
    assert soft.target.shape==(0,91)


def test_hard_soft_direct_reverse_common_jvp(monkeypatch):
    image,hard,soft=make_pair(monkeypatch);isp=DifferentiableISP().to(image.device)
    result,jac=gradients(image,isp,{'hard':hard,'soft':soft})
    assert all(r['parity']['passed'] and r['norm']>0 for r in result.values())
    assert jac['columns']==8 and all(jac['primal_identity_checks'])
    assert isp.phi.grad is None


def test_no_annotation_or_clip_import_transitively():
    code="import sys; import taisp.analysis.soft_pseudo_candidates; assert not [k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','reference','source_meta','clip_semantic','memory'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
