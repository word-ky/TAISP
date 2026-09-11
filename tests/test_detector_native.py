import os

import pytest
import torch

from taisp import DifferentiableISP
from taisp.losses.detector_native import DetectorNativeLoss, confidence_consistency
from taisp.tta import adapt, AdaptConfig


def support(device='cpu', empty=False):
    n = 0 if empty else 2
    chosen = {'boxes': torch.tensor([[20., 30., 170., 230.], [60., 40., 250., 200.]], device=device)[:n],
              'labels': torch.tensor([1, 3], device=device)[:n],
              'scores': torch.tensor([.8, .6], device=device)[:n]}
    return {'base': chosen, 'stable': {**chosen, 'flip_boxes': chosen['boxes'].clone()}}


def test_ce_js_equation_full_background_and_symmetry():
    p = torch.tensor([[.2, .3, .5], [.5, .4, .1]], dtype=torch.float64)
    q = torch.tensor([[.4, .2, .4], [.2, .6, .2]], dtype=torch.float64)
    labels, weights = torch.tensor([1, 2]), torch.tensor([.7, .3], dtype=torch.float64)
    ce, js = confidence_consistency(p.log(), labels, weights, q.log(), 1.)
    m = (p+q)/2
    expected_ce = -(weights*(p[torch.arange(2), labels].log()+q[torch.arange(2), labels].log())/2).sum()
    expected_js = (weights*((p*(p/m).log()).sum(1)+(q*(q/m).log()).sum(1))/2).sum()
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(js, expected_js)
    _, reverse = confidence_consistency(q.log(), labels, weights, p.log())
    _, identical = confidence_consistency(p.log(), labels, weights, p.log())
    torch.testing.assert_close(js, reverse)
    torch.testing.assert_close(identical, torch.zeros_like(identical), atol=1e-15, rtol=0)


@pytest.mark.parametrize('variant', ['det_pseudo', 'det_stable', 'det_stable_js'])
def test_empty_support_exact_no_update_and_episode_reset(variant):
    image = torch.rand(1, 3, 17, 23)
    isp = DifferentiableISP()
    loss = DetectorNativeLoss(None, support(empty=True), variant)
    cfg = AdaptConfig(steps=3, regularization_weight=0)
    a, b = adapt(image, isp, loss, config=cfg), adapt(image, isp, loss, config=cfg)
    assert torch.equal(a.phi, torch.zeros(8))
    # No update means exactly the unchanged ISP(phi0) output. The existing ISP
    # identity calculation is numerically, not bitwise, equal to the raw image.
    assert torch.equal(a.enhanced, isp(image, torch.zeros(8)))
    assert torch.equal(a.phi, b.phi) and torch.equal(isp.phi, torch.zeros(8))
    assert all(d['semantic'] == 0 for d in a.diagnostics)


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit real-model integration')
def test_real_native_loss_gradients_frozen_state_and_reset():
    from taisp.models.detector import load_detector
    torch.manual_seed(42)
    detector, isp = load_detector('cuda:0'), DifferentiableISP().cuda()
    image = .2+.5*torch.rand(1, 3, 257, 319, device='cuda')
    state = {k: v.cpu().clone() for k, v in detector.state_dict().items()}
    cfg = AdaptConfig(steps=1, regularization_weight=0)
    for variant in ('det_pseudo', 'det_stable', 'det_stable_js'):
        loss = DetectorNativeLoss(detector, support('cuda'), variant)
        a, b = adapt(image, isp, loss, config=cfg), adapt(image, isp, loss, config=cfg)
        assert a.phi.isfinite().all() and a.phi.norm() > 0
        assert torch.equal(a.phi0, torch.zeros_like(a.phi0))
        torch.testing.assert_close(a.phi, b.phi, atol=1e-6, rtol=1e-6)
        assert torch.equal(isp.phi, torch.zeros_like(isp.phi))
        assert not loss.boxes.requires_grad and not loss.weights.requires_grad
    for k, v in detector.state_dict().items():
        assert torch.equal(v.cpu(), state[k])
    assert all(not p.requires_grad and p.grad is None for p in detector.parameters())
