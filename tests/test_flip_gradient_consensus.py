import ast
import inspect
import math

import pytest
import torch

from taisp.analysis import flip_gradient_consensus as fg


def test_isp_horizontal_flip_commutation():
    assert fg.preflight('cuda:0' if torch.cuda.is_available() else 'cpu')['passed']


def test_symmetric_scale_free_consensus():
    a=[1.,0.]+[0.]*6;b=[0.,2.]+[0.]*6
    r=fg.consensus(a,b);s=fg.consensus(b,a)
    assert r['g_cons']==s['g_cons'] and r['agreement']==0
    assert r['g_cons'][:2]==pytest.approx([1/math.sqrt(2)]*2)
    assert fg.consensus([3*v for v in a],[.5*v for v in b])['g_cons']==r['g_cons']


def test_zero_and_antiparallel_abstain_without_fallback():
    a=[1.]+[0.]*7
    for b in ([0.]*8,[-v for v in a]):
        r=fg.consensus(a,b)
        assert r['g_cons']==[0.]*8 and r['abstain'] and r['agreement']==-1
    assert not fg.consensus(a,a)['abstain']


def test_candidate_import_boundary():
    imports=[n.module or '' for n in ast.walk(ast.parse(inspect.getsource(fg))) if isinstance(n,ast.ImportFrom)]
    assert not any(any(x in m for x in ('oracle','reference','memory')) for m in imports)
