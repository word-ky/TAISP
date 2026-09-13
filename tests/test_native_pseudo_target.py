import torch
from torch import nn

from taisp import DifferentiableISP
from taisp.analysis.native_pseudo_target import KEYS,NativePseudoTargetLoss
from taisp.models.detector_signal import select_predictions
from taisp.tta.trust_radius import adapt_clip_radius


class ToyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.7))
        self.rpn=nn.Identity();self.roi_heads=nn.Identity()
        self.seen=[]

    def forward(self,images,targets):
        assert self.training and self.rpn.training and self.roi_heads.training
        self.seen.append({k:v.clone() for k,v in targets[0].items()})
        y=images[0]
        return dict(zip(KEYS,[(y-self.weight).square().mean(),y.square().mean(),y.mean(),(1-y).square().mean()]))


class ToyDetector(nn.Module):
    def __init__(self):
        super().__init__();self.model=ToyNative()


class ToyClip(nn.Module):
    def forward(self,original,enhanced):
        return (enhanced-.3).square().mean()


def support():
    prediction={'boxes':torch.tensor([[1.,1.,5.,5.],[0.,0.,3.,3.],[2.,2.,6.,6.]],requires_grad=True),
                'scores':torch.tensor([.5,.8,.4],requires_grad=True),'labels':torch.tensor([1,3,7])}
    return select_predictions(prediction,.5,20)


def test_teacher_selection_detached_targets_unit_losses_and_no_score_weights():
    selected=support();source=ToyDetector().eval().requires_grad_(False)
    loss=NativePseudoTargetLoss(source,selected)
    assert torch.equal(loss.boxes,selected['boxes']) and torch.equal(loss.labels,selected['labels'])
    assert not loss.boxes.requires_grad and loss.labels.dtype==torch.long
    x=torch.full((1,3,8,8),.4,requires_grad=True)
    value=loss(x,x)
    assert set(loss.loss_history[0])==set(KEYS)
    torch.testing.assert_close(value,torch.tensor(sum(loss.loss_history[0].values())))
    selected['scores'].zero_();selected['boxes'].add_(10)
    torch.testing.assert_close(value,loss(x,x),rtol=0,atol=0)
    assert set(source.model.seen[0])=={'boxes','labels'}
    assert torch.equal(source.model.seen[0]['boxes'],loss.boxes)
    assert not any(m.training for m in source.modules())
    assert torch.isfinite(torch.autograd.grad(value,x)[0]).all()


def test_k3_only_phi_changes_and_model_modes_state_restore():
    source=ToyDetector().eval().requires_grad_(False)
    states={k:v.clone() for k,v in source.state_dict().items()}
    isp=DifferentiableISP();clip=ToyClip();x=torch.linspace(.1,.8,3*9*11).reshape(1,3,9,11)
    native=NativePseudoTargetLoss(source,support())
    result=adapt_clip_radius(x,isp,native,clip,steps=3,lr=.1)
    assert len(result.diagnostics)==len(native.loss_history)==4
    assert sum(d['update_applied'] for d in result.diagnostics)==3
    assert result.phi.norm()>0 and torch.equal(isp.phi,torch.zeros(8))
    assert all(torch.equal(v,states[k]) for k,v in source.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert all(not m.training for m in source.modules())


def test_empty_support_exact_no_update_and_current_path_unchanged():
    selected={k:v[:0] for k,v in support().items()}
    source=ToyDetector().eval().requires_grad_(False)
    native=NativePseudoTargetLoss(source,selected)
    isp=DifferentiableISP();clip=ToyClip();x=torch.rand(1,3,8,9)
    result=adapt_clip_radius(x,isp,native,clip)
    assert not source.model.seen
    assert torch.equal(result.phi,torch.zeros(8)) and torch.equal(result.enhanced,isp(x,torch.zeros(8)))
    assert all(d['gradient_norm']==0 for d in result.diagnostics)
    class CurrentLoss(ToyClip):
        boxes=torch.ones(1,4)
    current=CurrentLoss()
    a=adapt_clip_radius(x,isp,current,clip)
    adapt_clip_radius(x,isp,NativePseudoTargetLoss(source,support()),clip)
    b=adapt_clip_radius(x,isp,current,clip)
    assert torch.equal(a.phi,b.phi) and torch.equal(a.enhanced,b.enhanced)


def test_fixed_advancement_rejects_small_gain_bad_blocks_and_clean_drop():
    import copy
    from taisp.analysis.run_t018a import advancement, CASE_NAMES, METHODS
    metrics={group:{f'{case}_{method}':{'AP':.3+(.002 if method=='nativePT_ours' else 0)}
                    for case in CASE_NAMES for method in METHODS}
             for group in ['aggregate']+[f'block{i}' for i in range(4)]}
    assert advancement(metrics)['gate']['passed']
    small=copy.deepcopy(metrics)
    for case in CASE_NAMES[:-1]:
        small['aggregate'][f'{case}_nativePT_ours']['AP']=.3009
    assert not advancement(small)['gate']['passed']
    blocks=copy.deepcopy(metrics)
    for group in ('block0','block1'):
        for case in CASE_NAMES[:-1]:
            blocks[group][f'{case}_nativePT_ours']['AP']=.299
    assert not advancement(blocks)['gate']['passed']
    clean=copy.deepcopy(metrics)
    clean['aggregate']['clean_s0_nativePT_ours']['AP']=.2989
    assert not advancement(clean)['gate']['passed']
