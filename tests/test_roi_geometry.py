import ast
import inspect
import os

import pytest
import torch

from taisp import DifferentiableISP
from taisp.analysis import roi_geometry as geom
from taisp.analysis.common_jacobian import common_reference


def test_exposure_pair_bounded_finite_differentiable():
    y = torch.linspace(0,1,101,dtype=torch.double,requires_grad=True)
    assert geom.EXPOSURES == (1.2,1/1.2)
    for a in geom.EXPOSURES:
        z = geom.expose(y,a)
        assert ((z>=0)&(z<=1)).all() and torch.isfinite(z).all()
        d = torch.autograd.grad(z.sum(),y)[0]
        assert torch.allclose(d,a/(1+(a-1)*y).square()) and (d>0).all()


def test_class_specific_four_coordinate_layout_and_roi_order():
    raw = torch.arange(3*91*4).reshape(3,91*4)
    labels = torch.tensor([18,1,90])
    actual = geom.class_deltas(raw,labels)
    assert torch.equal(actual,torch.stack([raw[i,4*c:4*c+4] for i,c in enumerate(labels)]))
    order = torch.tensor([2,0,1])
    assert torch.equal(geom.class_deltas(raw[order],labels[order]),actual[order])


def test_empty_support_exact_zero_without_detector():
    x = torch.full((1,3,8,9),.4)
    c,receipt = geom.candidate_cotangent(None,x,torch.empty(0,4),torch.empty(0,dtype=torch.long))
    assert not c.any() and receipt['loss']==0 and receipt['empty_support']
    r,_ = geom.common_jvp(x,DifferentiableISP(),torch.zeros(1,1,8,9),{geom.NAME:c})
    assert r[geom.NAME]['object']==[0.]*8


def test_geometry_common_jvp_matches_accepted_reference():
    torch.manual_seed(25)
    x=.2+.6*torch.rand(1,3,8,9)
    mask,_=geom.support_mask(x,torch.tensor([[1.,1.,5.,7.]]))
    isp=DifferentiableISP();cs={geom.NAME:torch.randn_like(x)}
    assert geom.common_jvp(x,isp,mask,cs)[0]==common_reference(x,isp,mask,cs)[0]


def test_geometry_has_no_oracle_import_or_input():
    modules=[n.module or '' for n in ast.walk(ast.parse(inspect.getsource(geom))) if isinstance(n,ast.ImportFrom)]
    assert not any(any(x in m for x in ('oracle','reference','source_meta')) for m in modules)
    assert list(inspect.signature(geom.candidate_cotangent).parameters)==['detector','image','boxes','labels']


@pytest.mark.skipif(os.environ.get('TAISP_T025_REAL_SMOKE')!='1',reason='explicit cached real-model CUDA smoke')
def test_real_predictor_layout_and_candidate_frozen_cuda():
    source,isp,env=geom.setup()
    x=torch.linspace(.1,.9,3*48*64,device='cuda').reshape(1,3,48,64)
    boxes=x.new_tensor([[3.,4.,35.,39.],[20.,8.,60.,44.]])
    labels=torch.tensor([1,18],device='cuda')
    captured=[]
    handle=source.model.roi_heads.box_predictor.register_forward_hook(lambda m,args,out:captured.append(out[1].detach()))
    with torch.no_grad():raw=geom.fixed_roi_regression(source,x,boxes)
    handle.remove()
    assert raw.shape==(2,91*4) and torch.equal(raw,captured[0])
    assert torch.equal(geom.class_deltas(raw,labels),torch.stack([raw[0,4:8],raw[1,72:76]]))
    c,r=geom.candidate_cotangent(source,x,boxes,labels)
    assert torch.isfinite(c).all() and r['class_indices']==[1,18]
    assert geom.isolated(source) and geom.state_hash(source)==env['source_state_sha256']
