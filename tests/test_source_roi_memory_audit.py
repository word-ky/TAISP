import ast
import inspect

import torch

from taisp.analysis import source_roi_memory_audit as audit


def test_empty_candidate_exact_zero_no_fallback():
    x=torch.rand(1,3,6,8)
    c,anchors,r=audit.candidate_cotangent(None,x,torch.empty(0,4),torch.empty(0,dtype=torch.long),None,None)
    assert not c.any() and anchors.shape==(0,1024) and r['empty_support'] and r['retrieval_ids']==[]


def test_candidate_has_no_annotation_or_oracle_input_import():
    modules=[n.module or '' for n in ast.walk(ast.parse(inspect.getsource(audit))) if isinstance(n,ast.ImportFrom)]
    assert not any(any(t in m for t in ('oracle','reference','source_roi_memory.','source_meta')) for m in modules)
    assert list(inspect.signature(audit.candidate_cotangent).parameters)==['detector','image','boxes','labels','features','memory_classes']


def test_candidate_anchor_is_detached_and_class_retrieval_used(monkeypatch):
    features=torch.eye(4).repeat_interleave(16,dim=0)
    classes=torch.arange(1,5).repeat_interleave(16)
    x=torch.rand(1,3,6,8);boxes=torch.tensor([[0.,0.,4.,4.]])
    def fake(detector,image,boxes):
        z=torch.stack([image.mean(),image.square().mean(),image.sin().mean(),image.cos().mean()]).unsqueeze(0)
        return z,None
    monkeypatch.setattr(audit,'fixed_roi_representation',fake)
    c,anchor,r=audit.candidate_cotangent(None,x,boxes,torch.tensor([2]),features,classes)
    assert r['retrieval_ids']==[[16,17,18,19]] and not anchor.requires_grad
    assert torch.isfinite(c).all() and c.abs().sum()>0 and r['predicted_classes']==[2]
