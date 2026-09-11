"""Synthetic demo, not a real detector/CLIP experiment."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import torch
from torch import nn
import yaml

from taisp import DifferentiableISP
from taisp.losses import MockImageEncoder, SemanticDirectionLoss
from taisp.models import ParameterPredictor
from taisp.tta import AdaptConfig, adapt


def run_demo(settings: dict) -> dict:
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    device = torch.device(settings["device"])
    size = settings["image_size"]
    # Generate on CPU so seed yields the same image on CPU and CUDA.
    image = (0.15 + 0.3 * torch.rand(1, 3, size, size)).to(device)
    isp = DifferentiableISP().to(device)
    semantic = SemanticDirectionLoss(MockImageEncoder(), torch.ones(3)).to(device)
    downstream = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Tanh(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    ).to(device)
    predictor = ParameterPredictor().to(device) if settings["use_predictor"] else None
    config = AdaptConfig(**settings["adapt"])
    result = adapt(image, isp, semantic, downstream=downstream,
                   predictor=predictor, config=config)
    gradients = torch.tensor([d["gradient_per_coordinate"] for d in result.diagnostics[:-1]])
    if config.steps > 0:
        active = gradients.reshape(config.steps, 8).abs().amax(dim=0)
        near_zero = [name for name, value in zip(isp.parameter_names, active) if value < 1e-8]
    else:
        near_zero = []
    return {
        "kind": "synthetic_mock_smoke_only",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "cuda": torch.version.cuda},
        "settings": {**settings, "adapt": asdict(config)},
        "parameter_names": list(isp.parameter_names),
        "phi_before": result.phi0.cpu().tolist(),
        "phi_after": result.phi.cpu().tolist(),
        "physical_before": result.diagnostics[0]["physical"],
        "physical_after": result.diagnostics[-1]["physical"],
        "loss_before": result.diagnostics[0]["total"],
        "loss_after": result.diagnostics[-1]["total"],
        "prediction": result.prediction.cpu().tolist(),
        "repeatedly_near_zero_gradient_coordinates": near_zero,
        "diagnostics": result.diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--device", help="Override YAML device, e.g. cuda:0")
    parser.add_argument("--output", type=Path, help="Save exact JSON diagnostics")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device is not None:
        settings["device"] = args.device
    result = run_demo(settings)
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
