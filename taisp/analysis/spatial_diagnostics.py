"""T004 analysis partitions and label-free gradient-magnitude matching."""
import torch


def norm_match(gradient, reference):
    norm = gradient.norm()
    return gradient * (reference.norm()/norm) if norm.item() > 0 else gradient.clone()


def partition_gradients(image, isp, guidance, support):
    phi = torch.zeros(8, device=image.device, requires_grad=True)
    scores = guidance.patch_scores(image, isp(image, phi))
    weighted = scores * guidance.weights
    object_loss = (weighted*support).sum(-1).mean()
    background_loss = (weighted*(~support)).sum(-1).mean()
    g_object = torch.autograd.grad(object_loss, phi, retain_graph=True)[0]
    g_background = torch.autograd.grad(background_loss, phi)[0]
    return g_object.detach(), g_background.detach()


@torch.no_grad()
def partition_losses(image, enhanced, guidance, support):
    scores = guidance.patch_scores(image, enhanced).flatten()
    weighted = scores * guidance.weights.flatten()
    return {'patch_scores': scores.tolist(), 'weighted_contributions': weighted.tolist(),
            'object_weighted_sum': weighted[support].sum().item(),
            'background_weighted_sum': weighted[~support].sum().item(),
            'object_unweighted_mean': scores[support].mean().item() if support.any() else None,
            'background_unweighted_mean': scores[~support].mean().item() if (~support).any() else None}
