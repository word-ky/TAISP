import os

import pytest
import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import CLIPTensorPreprocess, load_clip_guidance
from taisp.tta import AdaptConfig, adapt


def test_tensor_preprocess_determinism_shape_and_gradients():
    torch.manual_seed(12)
    preprocess = CLIPTensorPreprocess(size=16)
    x = torch.rand(1, 3, 24, 40, requires_grad=True)
    a, b = preprocess(x), preprocess(x)
    assert a.shape == (1, 3, 16, 16)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0


def test_tensor_preprocess_constant_color():
    preprocess = CLIPTensorPreprocess()
    x = torch.full((1, 3, 224, 224), 0.5)
    expected = ((x - preprocess.mean) / preprocess.std)
    torch.testing.assert_close(preprocess(x), expected)


@pytest.mark.skipif(os.environ.get("TAISP_REAL_MODELS") != "1", reason="explicit pretrained-model integration run")
def test_real_clip_frozen_image_phi_gradients_and_episode_reset():
    device = "cuda:0"
    torch.manual_seed(9)
    guidance = load_clip_guidance(device, local_files_only=True)
    isp = DifferentiableISP().to(device)
    x = (0.2 + 0.5 * torch.rand(1, 3, 240, 320, device=device)).requires_grad_(True)
    y = isp(x)
    loss = guidance(x, y)
    image_grad, phi_grad = torch.autograd.grad(loss, (x, isp.phi))
    assert torch.isfinite(image_grad).all() and image_grad.norm() > 0
    assert torch.isfinite(phi_grad).all() and phi_grad.norm() > 0
    assert all(not p.requires_grad and p.grad is None for p in guidance.parameters())
    config = AdaptConfig(steps=1, regularization_weight=0)
    first = adapt(x.detach(), isp, guidance, config=config)
    adapt(x.detach() * 0.9, isp, guidance, config=config)
    repeat = adapt(x.detach(), isp, guidance, config=config)
    torch.testing.assert_close(first.phi, repeat.phi, atol=0, rtol=0)
    torch.testing.assert_close(isp.phi, torch.zeros_like(isp.phi))
