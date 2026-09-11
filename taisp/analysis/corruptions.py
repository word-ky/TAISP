"""Controlled shifts used only by offline experiments, never by adaptation."""

import torch

CASES = (("gamma", 1), ("gamma", 2), ("contrast", 1), ("contrast", 2),
         ("color_cast", 1), ("color_cast", 2))


def corrupt(image: torch.Tensor, family: str, severity: int) -> torch.Tensor:
    if family == "gamma":
        return image.pow({1: 1.5, 2: 2.0}[severity])
    if family == "contrast":
        return (image - 0.5) * {1: 0.6, 2: 0.3}[severity] + 0.5
    if family == "color_cast":
        gains = image.new_tensor({1: [1.2, 1., 0.8], 2: [1.4, 1., 0.6]}[severity])
        return (image * gains.view(1, 3, 1, 1)).clamp(0, 1)
    raise ValueError(f"Unknown experiment corruption: {family}")
