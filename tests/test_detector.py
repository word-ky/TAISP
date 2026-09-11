import ast
import os
from pathlib import Path

import pytest
import torch

from taisp import DifferentiableISP
from taisp.analysis.oracle import detector_task_loss
from taisp.models.detector import load_detector


def test_deployment_has_no_analysis_imports():
    for path in (Path("taisp/tta").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "analysis" not in (node.module or "")
            if isinstance(node, ast.Import):
                assert all("analysis" not in alias.name for alias in node.names)


@pytest.mark.skipif(os.environ.get("TAISP_REAL_MODELS") != "1", reason="explicit pretrained-model integration run")
def test_real_detector_oracle_freeze_gradients_rng_and_inference():
    torch.manual_seed(4)
    detector = load_detector("cuda:0")
    isp = DifferentiableISP().cuda()
    image = 0.2 + 0.5 * torch.rand(1, 3, 256, 320, device="cuda")
    targets = [{"boxes": torch.tensor([[30., 40., 180., 200.]], device="cuda"),
                "labels": torch.tensor([1], device="cuda")}]
    before = {key: value.cpu().clone() for key, value in detector.state_dict().items()}
    rng_before = torch.cuda.get_rng_state()
    loss, components = detector_task_loss(detector, isp(image), targets)
    grad = torch.autograd.grad(loss, isp.phi)[0]
    assert set(components) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert torch.equal(rng_before, torch.cuda.get_rng_state())
    repeat, _ = detector_task_loss(detector, isp(image), targets)
    torch.testing.assert_close(repeat, loss, atol=1e-6, rtol=1e-6)
    assert all(not m.training for m in detector.modules())
    assert all(not p.requires_grad and p.grad is None for p in detector.parameters())
    for key, value in detector.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[key], atol=0, rtol=0)
    with torch.no_grad():
        predictions = detector(image)
    assert set(predictions[0]) == {"boxes", "scores", "labels"}
    assert predictions[0]["boxes"].shape[-1] == 4
