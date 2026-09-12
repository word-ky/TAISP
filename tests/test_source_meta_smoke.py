import torch

from taisp import DifferentiableISP
from taisp.analysis import source_meta_smoke
from taisp.analysis.meta_gradient import MockQuadratic
from taisp.models.parameter_predictor import ParameterPredictor


def test_outer_targets_change_outer_only_and_episodes_reset(monkeypatch):
    torch.manual_seed(20260913)
    x = torch.linspace(.1, .6, 3*9*11).reshape(1, 3, 9, 11)
    predictor, isp = ParameterPredictor(), DifferentiableISP()
    native, clip = MockQuadratic(.9), MockQuadratic(.4)
    seen = []

    def outer(source, image, targets, *, seed):
        assert seed == 20260913
        seen.append(targets)
        loss = (image-targets).square().mean()
        return loss, {'mock_outer': loss}

    monkeypatch.setattr(source_meta_smoke, 'detector_task_loss', outer)
    first, a, _ = source_meta_smoke.source_outer_episode(x, .2, predictor, isp, native, clip, None)
    source_meta_smoke.source_outer_episode(x*.8, .7, predictor, isp, native, clip, None)
    repeated, b, _ = source_meta_smoke.source_outer_episode(x, .7, predictor, isp, native, clip, None)
    assert seen == [.2, .7, .7] and a != b
    torch.testing.assert_close(first.phi, repeated.phi, rtol=0, atol=0)
    torch.testing.assert_close(first.enhanced, repeated.enhanced, rtol=0, atol=0)
    a.backward()
    assert predictor.head.weight.grad.norm() > 0 and isp.phi.grad is None
    assert torch.equal(isp.phi, torch.zeros(8))
