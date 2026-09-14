import subprocess
import sys
import torch
from taisp import DifferentiableISP
from taisp.analysis.gradient_basis_candidates import clip_gradient


def test_clip_cotangent_matches_direct_loss_gradient():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    isp = DifferentiableISP().to(device)
    image = (.2+.4*torch.rand(1, 3, 16, 20, device=device))
    def loss(original, enhanced):
        return (enhanced.square()-original.detach().square()).mean()
    got, jac = clip_gradient(image, isp, loss)
    phi = image.new_zeros(8, requires_grad=True)
    expected = torch.autograd.grad(loss(image, isp(image, phi)), phi)[0]
    torch.testing.assert_close(torch.tensor(got['gradient'], device=device, dtype=expected.dtype), expected, rtol=1e-5, atol=1e-7)
    assert got['parity']['passed']


def test_candidate_import_boundary_fails_closed():
    code = """
import sys, types
from taisp.analysis.gradient_basis_candidates import label_free
assert label_free()==[]
sys.modules['taisp.analysis.oracle']=types.ModuleType('taisp.analysis.oracle')
try: label_free()
except AssertionError: pass
else: raise RuntimeError('oracle import was accepted')
"""
    subprocess.run([sys.executable, '-c', code], check=True)
