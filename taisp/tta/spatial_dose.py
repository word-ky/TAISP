"""R034: two ISP states share one direction; only their dose is redistributed."""
import time

import torch

from taisp.isp.spatial import compose,regional_gradients
from .adapt import AdaptResult


def dose_step(go, gb, co, cb, *, lr=.1, eps=1e-12):
    gs=go+gb;norm=gs.norm();zero=norm.item()==0
    u=gs/(norm+eps)
    ao,ab=go @ u,gb @ u
    c=(ao-ab)/(ao.abs()+ab.abs()+eps)
    multipliers=torch.stack((1+.5*c,1-.5*c))
    radius=(co+cb).norm();v=-lr*radius*u
    delta=multipliers[:,None]*v
    assert -1-1e-6<=c.item()<=1+1e-6
    assert bool(((multipliers>=.5-1e-6)&(multipliers<=1.5+1e-6)).all())
    assert abs(multipliers.mean().item()-1)<=1e-6
    return delta,dict(c=c.item(),multipliers=multipliers.tolist(),common_pseudo_gradient=gs.tolist(),
        common_pseudo_norm=norm.item(),shared_direction=u.tolist(),projection_obj=ao.item(),projection_bg=ab.item(),
        common_clip_gradient=(co+cb).tolist(),common_clip_norm=radius.item(),base_delta=v.tolist(),zero_common_pseudo=zero)


@torch.enable_grad()
def adapt_spatial_dose(image, isp, detector_loss, clip_loss, mask, *, steps=3, lr=.1, eps=1e-12):
    detector_loss.eval().requires_grad_(False);clip_loss.eval().requires_grad_(False)
    initial=image.new_zeros((2,8));states=initial.clone();mask=mask.detach();history=[]
    for step in range(steps+1):
        if image.is_cuda:torch.cuda.synchronize(image.device)
        start=time.perf_counter()
        enhanced=compose(image,isp,mask,*states).detach().requires_grad_(True)
        pseudo,clip=detector_loss(image,enhanced),clip_loss(image,enhanced)
        cp=torch.autograd.grad(pseudo,enhanced,retain_graph=True)[0].detach()
        cc=torch.autograd.grad(clip,enhanced)[0].detach()
        gradients=regional_gradients(image,isp,mask,states,{'pseudo':cp,'clip':cc})
        go,gb=gradients['pseudo'].to(image);co,cb=gradients['clip'].to(image)
        delta,d=dose_step(go,gb,co,cb,lr=lr,eps=eps)
        assert bool(torch.isfinite(delta).all() and torch.isfinite(states).all())
        if image.is_cuda:torch.cuda.synchronize(image.device)
        history.append(dict(step=step,total=pseudo.item(),semantic=pseudo.item(),clip_loss=clip.item(),
            consistency=0.,regularization=0.,phi=states.tolist(),phi_obj_norm=states[0].norm().item(),
            phi_bg_norm=states[1].norm().item(),phi_difference_norm=(states[0]-states[1]).norm().item(),
            object_gradient=go.tolist(),background_gradient=gb.tolist(),
            clip_object_gradient=co.tolist(),clip_background_gradient=cb.tolist(),
            detector_gradient=(go+gb).tolist(),detector_gradient_norm=(go+gb).norm().item(),
            clip_gradient_norm=(co+cb).norm().item(),gradient_norm=(delta/lr).norm().item(),
            state_delta=delta.tolist(),support_count=len(detector_loss.boxes),update_applied=step<steps,
            step_seconds=time.perf_counter()-start,**d))
        if step<steps:states=(states+delta).detach()
    return AdaptResult(initial,states.detach(),enhanced.detach(),None,history)
