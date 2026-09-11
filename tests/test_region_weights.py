import torch
from torch import nn

from taisp.models.region_weights import original_region_weights, region_patch_weights


def test_known_grid_overlap_score_weights_and_filtering():
    boxes = torch.tensor([[0., 0, 32, 32], [32, 0, 64, 32], [0, 0, 224, 224]], requires_grad=True)
    scores = torch.tensor([.8, .6, .4], requires_grad=True)
    r = region_patch_weights(boxes, scores, (224, 224))
    torch.testing.assert_close(r['weights'][:2], torch.tensor([4/7, 3/7]))
    assert r['weights'][2:].sum() == 0 and r['selected_count'] == 2
    assert not r['weights'].requires_grad and not r['mapped_boxes'].requires_grad
    torch.testing.assert_close(r['raw_weights'][:2], torch.tensor([.8, .6]))
    one = region_patch_weights(boxes, scores, (224, 224), topk=1)
    assert one['weights'][0] == 1


def test_odd_crop_box_mapping_and_uniform_fallback():
    # 333x517 ->224x347, floor left61. Invert mapping for exactly first patch.
    box = torch.tensor([[61*517/347, 0, 93*517/347, 32*333/224]])
    r = region_patch_weights(box, torch.tensor([.9]), (333, 517))
    torch.testing.assert_close(r['mapped_boxes'], torch.tensor([[0., 0, 32, 32]]), atol=1e-5, rtol=0)
    assert r['weights'][0] > .99999
    for boxes, scores in ((torch.empty(0, 4), torch.empty(0)),
                           (torch.tensor([[0., 0, 10, 20]]), torch.tensor([.9]))):
        r = region_patch_weights(boxes, scores, (333, 517))
        assert r['uniform_fallback'] and not r['support'].any()
        torch.testing.assert_close(r['weights'], torch.full((49,), 1/49))


def test_original_detector_called_once_and_weights_are_fixed():
    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.))
            self.calls = 0
        def forward(self, image):
            self.calls += 1
            assert not torch.is_grad_enabled() and not self.training
            return [{'boxes': image.new_tensor([[0., 0, 32, 32]]), 'scores': self.scale[None]}]
    detector = Detector()
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    r = original_region_weights(detector, x)
    assert detector.calls == 1 and r['weights'][0] == 1
    assert not detector.scale.requires_grad and detector.scale.item() == 1
    assert x.grad is None and not r['weights'].requires_grad
