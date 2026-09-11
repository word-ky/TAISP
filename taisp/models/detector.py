"""Frozen COCO detector inference; oracle labels are handled only in analysis."""

from torch import nn


class FrozenDetector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, image):
        """BCHW -> torchvision list of boxes/scores/labels in original pixels."""
        self.model.eval()
        return self.model(list(image))


def load_detector(device="cpu"):
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
    )

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, weights_backbone=None,
    )
    return FrozenDetector(model).to(device).eval()
