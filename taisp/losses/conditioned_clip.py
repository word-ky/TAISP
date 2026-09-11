"""Image-only degradation inference; frozen, detached for each fresh episode."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .clip_semantic import CLIP_MODEL, CLIP_REVISION, POSITIVE_PROMPTS, NEGATIVE_PROMPTS
from .semantic import SemanticDirectionLoss

CONCEPTS = ('darkness', 'low_contrast', 'color_cast')
NEGATIVE_BANKS = tuple(NEGATIVE_PROMPTS[i:i+2] for i in (0, 2, 4))
COORDINATE_MASKS = ((1, 0, 0, 0, 0, 1, 1, 0),
                    (0, 0, 0, 0, 1, 0, 1, 0),
                    (0, 1, 1, 1, 0, 0, 0, 0))


@dataclass
class ConditionedEpisode:
    guidance: SemanticDirectionLoss
    weights: torch.Tensor
    similarities: torch.Tensor
    gate: torch.Tensor


class CLIPConditioner(nn.Module):
    def __init__(self, encoder, positive, negatives, temperature=0.05):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.register_buffer('positive', F.normalize(positive.detach().clone(), dim=-1))
        self.register_buffer('negatives', F.normalize(negatives.detach().clone(), dim=-1))
        self.register_buffer('masks', positive.new_tensor(COORDINATE_MASKS))
        self.temperature = temperature

    @torch.no_grad()
    def prepare(self, original):
        """Compute once from the original image. No labels or corruption IDs."""
        self.encoder.eval()
        similarities = F.normalize(self.encoder(original), dim=-1) @ self.negatives.T
        weights = torch.softmax(similarities / self.temperature, dim=-1)
        direction = self.positive - weights @ self.negatives
        guidance = SemanticDirectionLoss(self.encoder, direction)
        return ConditionedEpisode(guidance, weights, similarities, weights @ self.masks)


def load_conditioner(generic_guidance, temperature=0.05, *, local_files_only=False):
    """Reuse the already-loaded T002 CLIP encoder and exact original prompt text."""
    from transformers import CLIPTokenizer

    encoder = generic_guidance.encoder
    device = next(encoder.parameters()).device
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION,
                                              local_files_only=local_files_only)
    tokens = tokenizer(list(POSITIVE_PROMPTS + NEGATIVE_PROMPTS), padding=True,
                       return_tensors='pt').to(device)
    with torch.no_grad():
        text = F.normalize(encoder.model.get_text_features(**tokens), dim=-1)
        positive = F.normalize(text[:3].mean(0), dim=-1)
        negatives = F.normalize(text[3:].reshape(3, 2, -1).mean(1), dim=-1)
    return CLIPConditioner(encoder, positive, negatives, temperature).to(device)
