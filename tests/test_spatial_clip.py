import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance, resize_crop_geometry
from taisp.losses.spatial_clip import PatchDirectionLoss
from taisp.tta import AdaptConfig, adapt


class TinyPatches(nn.Module):
    def patch_features(self, x):
        return F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)


def test_spatial_weighted_displacement_and_detached_reference():
    x = torch.zeros(1, 3, 2, 2, requires_grad=True)
    y = torch.zeros_like(x, requires_grad=True)
    with torch.no_grad():
        y[0, 0] = torch.tensor([[1., 2.], [3., 4.]])
    weights = torch.tensor([1., 0, 0, 3.], requires_grad=True)
    loss = PatchDirectionLoss(TinyPatches(), torch.tensor([1., 0, 0]), weights)
    value = loss(x, y)
    torch.testing.assert_close(value, torch.tensor(-3.25))
    value.backward()
    assert x.grad is None and weights.grad is None
    torch.testing.assert_close(y.grad[0, 0], torch.tensor([[-.25, 0], [0, -.75]]))


def test_preprocess_region_geometry_odd_rectangle_and_square():
    assert resize_crop_geometry(333, 517) == (224, 347, 0, 61)
    assert resize_crop_geometry(517, 333) == (347, 224, 61, 0)
    assert resize_crop_geometry(612, 612) == (224, 224, 0, 0)


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit pretrained-model integration run')
def test_real_patch_projection_frozen_gradients_and_fresh_episodes():
    torch.manual_seed(23)
    generic = load_clip_guidance('cuda:0', local_files_only=True)
    encoder = generic.encoder
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    isp = DifferentiableISP().cuda()
    for h, w in ((333, 517), (517, 333), (612, 612)):
        x = .2 + .5*torch.rand(1, 3, h, w, device='cuda:0')
        features = encoder.patch_features(x)
        assert features.shape == (1, 49, 512)
        with torch.no_grad():
            vision = encoder.model.vision_model(pixel_values=encoder.preprocess(x), return_dict=True)
            manual = F.normalize(encoder.model.visual_projection(encoder.model.vision_model.post_layernorm(vision.last_hidden_state[:, 1:])),dim=-1)
        torch.testing.assert_close(features, manual, atol=0, rtol=0)
        loss = PatchDirectionLoss(encoder, generic.direction)
        g = torch.autograd.grad(loss(x, isp(x)), isp.phi)[0]
        assert torch.isfinite(g).all() and g.norm() > 0
    x = .2 + .5*torch.rand(1, 3, 240, 320, device='cuda:0')
    weights = torch.arange(1, 50, device=x.device, dtype=x.dtype)
    loss = PatchDirectionLoss(encoder, generic.direction, weights)
    cfg = AdaptConfig(steps=1, regularization_weight=0)
    a = adapt(x, isp, loss, config=cfg)
    adapt(x*.8, isp, loss, config=cfg)
    b = adapt(x, isp, loss, config=cfg)
    torch.testing.assert_close(a.phi0, torch.zeros_like(a.phi0), atol=0, rtol=0)
    torch.testing.assert_close(a.phi, b.phi, atol=1e-8, rtol=1e-5)
    assert a.phi.norm() > 0
    for k, v in encoder.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
