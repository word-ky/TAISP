"""TAISP: adapt the image-processing state while keeping the model frozen."""

from .isp.module import DifferentiableISP

__all__ = ["DifferentiableISP"]
