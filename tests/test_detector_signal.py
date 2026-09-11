import inspect
import os

import pytest
import torch

from taisp import DifferentiableISP
from taisp.models.detector_signal import (flip_boxes, select_predictions, stable_pairs,
                                         original_support, fixed_roi_logits)


def test_flip_geometry_and_matching_ties():
    boxes = torch.tensor([[1., 2., 7., 8.], [1., 2., 7., 8.]])
    torch.testing.assert_close(flip_boxes(boxes, 13), torch.tensor([[6., 2., 12., 8.]]).repeat(2, 1))
    assert torch.equal(flip_boxes(flip_boxes(boxes, 13), 13), boxes)
    base = {'boxes': boxes, 'labels': torch.tensor([2, 2])}
    assert stable_pairs(base, base) == [(0, 0, 1.), (1, 1, 1.)]
    other = {**base, 'labels': torch.tensor([3, 2])}
    assert stable_pairs(base, other) == [(0, 1, 1.)]
    shifted = {**base, 'boxes': boxes+20}
    assert stable_pairs(base, shifted) == []


def test_selection_and_original_only_support():
    pred = {'boxes': torch.tensor([[1., 2., 5., 6.]]).repeat(4, 1),
            'scores': torch.tensor([.8, .8, .4, .95]), 'labels': torch.tensor([1, 2, 1, 0])}
    chosen = select_predictions(pred, topk=1)
    assert chosen['labels'].tolist() == [1]
    seen = []
    image = torch.rand(1, 3, 9, 11, requires_grad=True)
    def detector(x):
        assert not torch.is_grad_enabled()
        seen.append(x.detach().clone())
        return [pred]
    original_support(detector, image)
    assert len(seen) == 2 and torch.equal(seen[0].flip(-1), seen[1])
    assert list(inspect.signature(original_support).parameters) == ['detector', 'image', 'threshold', 'topk', 'iou_threshold']
    assert list(inspect.signature(fixed_roi_logits).parameters) == ['detector', 'image', 'boxes']


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit real-model integration')
def test_real_fixed_roi_path_freeze_and_phi_gradient():
    from taisp.models.detector import load_detector
    torch.manual_seed(42)
    detector = load_detector('cuda:0')
    isp = DifferentiableISP().cuda()
    image = .2+.5*torch.rand(1, 3, 257, 319, device='cuda')
    boxes = torch.tensor([[20., 30., 170., 230.], [60., 40., 250., 200.]], device='cuda')
    state = {k: v.cpu().clone() for k, v in detector.state_dict().items()}
    def forbidden(*args):
        raise AssertionError('fixed ROI signal must bypass RPN')
    hook = detector.model.rpn.register_forward_pre_hook(forbidden)
    logits = fixed_roi_logits(detector, isp(image), boxes)
    assert logits.shape == (2, 91)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3], device='cuda'))
    grad = torch.autograd.grad(loss, isp.phi)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0
    hook.remove()
    assert all(not m.training for m in detector.modules())
    assert all(not p.requires_grad and p.grad is None for p in detector.parameters())
    for k, v in detector.state_dict().items():
        assert torch.equal(v.cpu(), state[k])
