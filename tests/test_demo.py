from pathlib import Path

import pytest
import torch
import yaml

from taisp.demo import run_demo


@pytest.mark.parametrize("predictor", [False, True])
def test_configured_demo(predictor):
    settings = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    settings["use_predictor"] = predictor
    result = run_demo(settings)
    assert result["loss_after"] < result["loss_before"]
    assert result["physical_after"]["brightness"] != result["physical_before"]["brightness"]
    assert result["repeatedly_near_zero_gradient_coordinates"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_demo_matches_cpu():
    settings = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    cpu = run_demo(settings)
    gpu = run_demo({**settings, "device": "cuda:0"})
    torch.testing.assert_close(torch.tensor(cpu["phi_after"]), torch.tensor(gpu["phi_after"]),
                               atol=1e-5, rtol=1e-4)
    assert gpu["loss_after"] < gpu["loss_before"]
    assert gpu["repeatedly_near_zero_gradient_coordinates"] == []
