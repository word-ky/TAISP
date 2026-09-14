import ast
import inspect
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from taisp import DifferentiableISP
from taisp.analysis.native_task_signal import native_task_loss,pseudo_native_loss,pseudo_targets,KEYS
from taisp.analysis.pseudo_native_candidates import gradients,component_sum_check
from taisp.analysis.oracle import detector_task_loss
from taisp.analysis.roi_equivariance import state_hash,isolated


class ToyNative(torch.nn.Module):
    def __init__(self):
        super().__init__();self.rpn=torch.nn.Identity();self.roi_heads=torch.nn.Identity()
        self.backbone=torch.nn.BatchNorm2d(3);self.fail=False
    def forward(self,images,targets):
        assert self.training and self.rpn.training and self.roi_heads.training
        assert not self.backbone.training
        if self.fail:raise RuntimeError('observed test exception')
        x=self.backbone(torch.stack(images));v=x.square().mean()
        draw=torch.rand(4,device=x.device)
        return {k:(i+1+draw[i])*v for i,k in enumerate(KEYS)}


def fixtures():
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=ToyNative().to(device).eval().requires_grad_(False)
    source=SimpleNamespace(model=model)
    image=(.2+.4*torch.rand(1,3,16,18,device=device)).requires_grad_(True)
    targets=[{'boxes':image.new_tensor([[1,1,10,11]]),'labels':torch.tensor([1],device=device)}]
    return source,image,targets


def test_neutral_helper_is_mechanical_oracle_body():
    a=ast.parse(inspect.getsource(native_task_loss)).body[0].body
    b=ast.parse(inspect.getsource(detector_task_loss)).body[0].body[1:]
    assert [ast.dump(x) for x in a]==[ast.dump(x) for x in b]


def test_equivalence_rng_restore_freeze_and_component_sum():
    source,x,t=fixtures();h=state_hash(source.model)
    cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None
    loss,parts=native_task_loss(source,x,t,seed=20260930)
    expected,ep=detector_task_loss(source,x,t,seed=20260930)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    for k in KEYS:torch.testing.assert_close(parts[k],ep[k],rtol=0,atol=0)
    assert torch.equal(cpu,torch.random.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(x.device))
    torch.testing.assert_close(loss,sum(parts.values()),rtol=0,atol=0)
    g=torch.autograd.grad(loss,x,retain_graph=True)[0]
    cg=sum(torch.autograd.grad(v,x,retain_graph=True)[0] for v in parts.values())
    torch.testing.assert_close(g,cg,rtol=1e-5,atol=1e-7)
    assert isolated(source.model) and state_hash(source.model)==h


def test_eval_restoration_after_exception():
    source,x,t=fixtures();source.model.fail=True
    with pytest.raises(RuntimeError):native_task_loss(source,x,t)
    assert isolated(source.model)


def test_detached_pseudo_targets_empty_zero():
    _,x,t=fixtures();boxes=t[0]['boxes'].requires_grad_(True)
    hard=SimpleNamespace(boxes=boxes,labels=t[0]['labels'])
    copied=pseudo_targets(hard)
    assert not copied[0]['boxes'].requires_grad and torch.equal(copied[0]['boxes'],boxes)
    assert copied[0]['boxes'].data_ptr()!=boxes.data_ptr()
    copied[0]['boxes']=copied[0]['boxes'][:0];copied[0]['labels']=copied[0]['labels'][:0]
    loss,parts=pseudo_native_loss(None,x,copied)
    assert loss.item()==0 and set(parts)==set(KEYS)
    assert torch.count_nonzero(torch.autograd.grad(loss,x)[0])==0


def test_full_component_gradients_and_jvp_parity():
    source,x,t=fixtures();isp=DifferentiableISP().to(x.device)
    values,jac,rng=gradients(x.detach(),isp,source,lambda original,y:y.square().mean(),t)
    assert rng['restored'] and all(v['parity']['passed'] for v in values.values())
    assert component_sum_check(values)['passed'] and jac['columns']==8
    values['native']['gradient'][0]+=1
    assert not component_sum_check(values)['passed']


def test_candidate_transitive_import_boundary():
    code="import sys; import taisp.analysis.pseudo_native_candidates; assert not [k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','reference','source_meta','clip_semantic','memory'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
