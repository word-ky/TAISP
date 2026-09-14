import subprocess
import sys
from types import SimpleNamespace
import torch
from taisp.analysis import orthogonal_candidates as o
from taisp.analysis.flip_gradient_consensus import view_gradient


def test_original_hard_function_reused_and_rng_receipts(monkeypatch):
    assert o.view_gradient is view_gradient
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    x=torch.ones(1,3,2,2,device=device);calls=[]
    monkeypatch.setattr(o,'state_hash',lambda source:'fixed');monkeypatch.setattr(o,'isolated',lambda source:True)
    def hard(source,isp,image):
        assert image is x;calls.append(1)
        return {'gradient':[0.]*8,'loss':0.,'parity':{'passed':True}}
    monkeypatch.setattr(o,'view_gradient',hard)
    r=o.candidate(None,SimpleNamespace(phi=x.new_zeros(8)),x)
    assert calls==[1] and r['zero'] and r['integrity_passed'] and r['rng']['restored']


def test_no_annotation_native_or_extra_objective_imports():
    code="import sys; import taisp.analysis.orthogonal_candidates; assert not [k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('oracle','reference','source_meta','common_jacobian','clip_semantic','memory','native_task_signal','soft_pseudo'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
