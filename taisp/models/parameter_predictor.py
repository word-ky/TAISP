import torch
from torch import nn


class ParameterPredictor(nn.Module):
    """Small optional source-trained initializer; returns raw phi, shape (B,8).

    Initially outputs identity. Deployment must not optimize these weights.
    Training this initializer is deliberately outside the T001 baseline.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(hidden, 8)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))
