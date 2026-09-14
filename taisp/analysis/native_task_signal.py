"""Neutral native-loss call for detached pseudo targets; no annotation imports."""
import torch

KEYS=('loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg')


def native_task_loss(detector,image,targets,*,seed=20260930):
    # Mechanical copy of the validated oracle call; the original stays unchanged.
    model=detector.model
    model.eval()
    devices=[image.device.index or 0] if image.is_cuda else []
    try:
        model.training=model.rpn.training=model.roi_heads.training=True
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            components=model(list(image),targets)
        return sum(components.values()),components
    finally:
        model.eval()


def pseudo_targets(hard):
    return [{'boxes':hard.boxes.detach().clone(),'labels':hard.labels.detach().clone().long()}]


def pseudo_native_loss(detector,image,targets):
    if not len(targets[0]['boxes']):
        zero=image.sum()*0
        return zero,{k:zero for k in KEYS}
    return native_task_loss(detector,image,targets,seed=20260930)
