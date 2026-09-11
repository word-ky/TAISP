"""Frozen OpenAI CLIP guidance; no annotations or corruption inputs."""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic import SemanticDirectionLoss

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
POSITIVE_PROMPTS = (
    "a clear natural photograph",
    "a well-lit photograph with natural colors",
    "a sharp and clear photograph with good visibility",
)
NEGATIVE_PROMPTS = (
    "a dark underexposed photograph",
    "a poorly lit photograph with low visibility",
    "a low contrast washed-out photograph",
    "a hazy photograph with faded details",
    "a photograph with an unnatural color cast",
    "a photograph with distorted and unbalanced colors",
)


class CLIPTensorPreprocess(nn.Module):
    """Shortest-side 224 bicubic antialias -> center crop -> CLIP RGB normalize.

    Input/output remain floating tensors. Bicubic overshoot is preserved (no
    additional clipping); this is the documented differentiable tensor variant.
    """

    def __init__(self, size=224):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, image):
        h, w = image.shape[-2:]
        # Fix shortest side exactly; float 612 * (224 / 612) rounds below
        # 224, which would produce a 223x223 crop and the wrong patch count.
        rh, rw = ((self.size, self.size * w // h) if h <= w
                  else (self.size * h // w, self.size))
        x = F.interpolate(image, size=(rh, rw), mode="bicubic", align_corners=False, antialias=True)
        # Transformers' pinned CLIPImageProcessor uses floor, not round, when
        # an odd difference leaves two possible center-crop placements.
        top, left = (rh - self.size) // 2, (rw - self.size) // 2
        return (x[..., top:top + self.size, left:left + self.size] - self.mean) / self.std


class FrozenCLIPEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.preprocess = CLIPTensorPreprocess()

    def forward(self, image):
        self.model.eval()
        return F.normalize(self.model.get_image_features(pixel_values=self.preprocess(image)), dim=-1)


def load_clip_guidance(device="cpu", *, local_files_only=False):
    """Generic bank direction: equal prompt weighting, independent of test type.

    All model weights are float32/frozen; tensor image path remains differentiable.
    """
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION,
                                      local_files_only=local_files_only).float().to(device)
    model.eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION,
                                              local_files_only=local_files_only)
    prompts = list(POSITIVE_PROMPTS + NEGATIVE_PROMPTS)
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text = F.normalize(model.get_text_features(**tokens), dim=-1)
        positive = F.normalize(text[:len(POSITIVE_PROMPTS)].mean(0), dim=0)
        negative = F.normalize(text[len(POSITIVE_PROMPTS):].mean(0), dim=0)
        direction = F.normalize(positive - negative, dim=0)
    encoder = FrozenCLIPEncoder(model).to(device)
    return SemanticDirectionLoss(encoder, direction).to(device)
