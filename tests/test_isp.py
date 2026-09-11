import io

import torch

from taisp import DifferentiableISP


def test_identity_and_roundtrip():
    torch.manual_seed(1)
    isp = DifferentiableISP()
    x = torch.rand(2, 3, 8, 9)
    x[0, 0, 0, :2] = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(isp(x), x, atol=2e-7, rtol=1e-6)
    buffer = io.BytesIO()
    torch.save(isp.state_dict(), buffer)
    buffer.seek(0)
    restored = DifferentiableISP()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x), isp(x))


def test_bounds_and_batched_parameters():
    isp = DifferentiableISP()
    phi = torch.stack([torch.full((8,), -100.0), torch.full((8,), 100.0)])
    p = isp.decode(phi)
    for name in ("gamma", "gains", "contrast"):
        assert (p[name] >= 0.5).all() and (p[name] <= 2).all()
    for name, bound in (("brightness", 0.25), ("tone", 0.5), ("sharpening", 0.5)):
        assert (p[name].abs() <= bound).all()
    y = isp(torch.rand(2, 3, 8, 9), phi)
    assert y.shape == (2, 3, 8, 9)
    assert torch.isfinite(y).all() and (y >= 0).all() and (y <= 1).all()


def test_all_eight_coordinates_have_correct_gradients():
    torch.manual_seed(2)
    isp = DifferentiableISP().double()
    x = 0.2 + 0.5 * torch.rand(1, 3, 4, 5, dtype=torch.double)
    phi = torch.full((8,), 0.03, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: isp(x, p), (phi,))
    weights = torch.rand_like(x)
    grad = torch.autograd.grad((isp(x, phi) * weights).sum(), phi)[0]
    assert torch.isfinite(grad).all() and (grad.abs() > 1e-8).all()


def test_black_pixels_have_finite_gradients():
    isp = DifferentiableISP()
    x = torch.zeros(1, 3, 4, 4, requires_grad=True)
    isp(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(isp.phi.grad).all()
