"""Differentiable RGB operators; scalar parameters broadcast over BCHW."""

import torch
from torch.nn import functional as F


def gamma(image: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    # Shifted normalized power keeps black/white endpoints and finite slopes.
    eps = image.new_tensor(1e-6)
    offset = eps.pow(exponent)
    return ((image + eps).pow(exponent) - offset) / (
        (1 + eps).pow(exponent) - offset
    )


def white_balance(image: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    return image * gains


def contrast(image: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (image - 0.5) * scale + 0.5


def brightness(image: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return image + offset


def tone(image: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    return image + amount * image * (1 - image)


def sharpen(image: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    blur = F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return image + amount * (image - blur)
