import pytest
import torch

from test_native_pseudo_target import ToyDetector, ToyClip, support
from taisp import DifferentiableISP
from taisp.analysis.native_pseudo_target import NativePseudoTargetLoss, KEYS
from taisp.analysis.native_component_study import CANDIDATES, component_selection
from taisp.analysis.oracle import detector_task_loss
from taisp.analysis.run_t018a import CASE_NAMES
from taisp.tta.trust_radius import adapt_clip_radius

DECLARED={
    'native_cls':('loss_classifier',),
    'native_conf':('loss_classifier','loss_objectness'),
    'native_roi':('loss_classifier','loss_box_reg'),
    'native_conf_roi':('loss_classifier','loss_objectness','loss_box_reg'),
}


@pytest.mark.parametrize('candidate',CANDIDATES)
def test_exact_active_sum_gradient_detached_support_and_K3(candidate):
    source=ToyDetector().eval().requires_grad_(False);selected=support()
    native=NativePseudoTargetLoss(source,selected,component_set=candidate)
    x=torch.linspace(.1,.8,3*8*9).reshape(1,3,8,9).requires_grad_(True)
    _,parts=detector_task_loss(source,x,[{'boxes':selected['boxes'],'labels':selected['labels']}],seed=20260912)
    expected=sum(parts[k] for k in DECLARED[candidate]); actual=native(x,x)
    assert torch.equal(actual,expected)
    assert torch.equal(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])
    assert tuple(native.loss_history[0])==DECLARED[candidate]==native.active_keys
    assert 'loss_rpn_box_reg' not in native.loss_history[0]
    assert not native.boxes.requires_grad and not native.labels.requires_grad
    before={k:v.clone() for k,v in source.state_dict().items()}
    isp=DifferentiableISP(); result=adapt_clip_radius(x.detach(),isp,native,ToyClip())
    assert sum(d['update_applied'] for d in result.diagnostics)==3
    assert result.phi.norm()>0 and torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None
    assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert all(not m.training for m in source.modules())
    assert torch.equal(native.boxes,selected['boxes']) and torch.equal(native.labels,selected['labels'])


@pytest.mark.parametrize('candidate',CANDIDATES)
def test_subset_empty_support_is_exact_zero_update(candidate):
    selected={k:v[:0] for k,v in support().items()};source=ToyDetector().eval().requires_grad_(False)
    native=NativePseudoTargetLoss(source,selected,component_set=candidate)
    isp=DifferentiableISP();x=torch.rand(1,3,8,9)
    result=adapt_clip_radius(x,isp,native,ToyClip())
    assert torch.count_nonzero(result.phi)==0 and torch.equal(result.enhanced,isp(x,torch.zeros(8)))
    assert not source.model.seen and all(all(v==0 for v in d.values()) for d in native.loss_history)


def test_default_full_matches_legacy_oracle_exactly():
    source=ToyDetector().eval().requires_grad_(False);selected=support();x=torch.rand(1,3,8,9)
    old,_=detector_task_loss(source,x,[{'boxes':selected['boxes'],'labels':selected['labels']}],seed=20260912)
    full=NativePseudoTargetLoss(source,selected)
    assert torch.equal(full(x,x),old) and tuple(full.loss_history[0])==KEYS


def metrics_example():
    ap={'no_adapt':.3,'current_ours':.3,'native_cls':.302,'native_conf':.30195,'native_roi':.299,'native_conf_roi':.299}
    return {g:{f'{c}_{m}':{'AP':v,'AP50':.5,'AP75':.41 if m=='native_conf' else .4}
               for c in CASE_NAMES for m,v in ap.items()} for g in ['aggregate']+[f'block{i}' for i in range(4)]}


def test_selection_point01_tie_then_AP75_and_primary_metric():
    m=metrics_example();result=component_selection(m,True)
    assert result['gate']['selected']=='native_conf'
    assert result['gate']['eligible_candidates']==['native_cls','native_conf']
    for c in CASE_NAMES[:-1]:m['aggregate'][f'{c}_native_cls']['AP']=.3023
    assert component_selection(m,True)['gate']['selected']=='native_cls'


def test_selection_uses_positive_blocks_after_AP75_tie():
    m=metrics_example()
    for c in CASE_NAMES[:-1]:
        m['aggregate'][f'{c}_native_conf']['AP75']=.4
        m['block3'][f'{c}_native_cls']['AP']=.299
    assert component_selection(m,True)['gate']['selected']=='native_conf'


def test_none_eligible_if_materiality_or_isolation_fails():
    m=metrics_example()
    assert component_selection(m,False)['gate']['selected'] is None
    for c in CASE_NAMES[:-1]:
        for candidate in CANDIDATES:m['aggregate'][f'{c}_{candidate}']['AP']=.3005
    assert component_selection(m,True)['gate']['decision']=='close_fixed_component_subset_branch'
