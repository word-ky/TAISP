"""Eight-dimensional image-formation state, with zero as identity."""

import math

import torch
from torch import nn

from . import ops


class DifferentiableISP(nn.Module):
    """RGB BCHW [0,1] -> RGB BCHW [0,1]; phi is raw shape (8,) or (B,8).

    External phi enables functional inner updates without mutating module state.
    The owned Parameter provides a serializable identity/source initialization.
    """

    parameter_names = (
        "gamma", "red_gain", "green_gain", "blue_gain", "contrast",
        "brightness", "tone", "sharpening",
    )
    num_parameters = 8

    def __init__(self):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(self.num_parameters))

    @staticmethod
    def decode(phi: torch.Tensor) -> dict[str, torch.Tensor]:
        bounded = phi.tanh()
        return {
            "gamma": (math.log(2) * bounded[..., 0]).exp(),
            "gains": (math.log(2) * bounded[..., 1:4]).exp(),
            "contrast": (math.log(2) * bounded[..., 4]).exp(),
            "brightness": 0.25 * bounded[..., 5],
            "tone": 0.5 * bounded[..., 6],
            "sharpening": 0.5 * bounded[..., 7],
        }

    def forward(self, image: torch.Tensor, phi: torch.Tensor | None = None):
        p = self.decode(self.phi if phi is None else phi)
        x = ops.gamma(image, p["gamma"].reshape(-1, 1, 1, 1))
        x = ops.white_balance(x, p["gains"].reshape(-1, 3, 1, 1))
        x = ops.contrast(x, p["contrast"].reshape(-1, 1, 1, 1))
        x = ops.brightness(x, p["brightness"].reshape(-1, 1, 1, 1))
        x = ops.tone(x, p["tone"].reshape(-1, 1, 1, 1))
        x = ops.sharpen(x, p["sharpening"].reshape(-1, 1, 1, 1))
        return x.clamp(0, 1)
