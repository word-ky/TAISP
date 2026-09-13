import copy
import inspect

import pytest
import torch
from torch import nn

from taisp import DifferentiableISP
from taisp.analysis.gradient_transport import fit_folds, heldout_alignment, transport_advancement
from taisp.tta.gradient_transport import adapt_gradient_transport
from taisp.tta.trust_radius import adapt_clip_radius


def pairs():
    torch.manual_seed(23)
    q = torch.eye(8, dtype=torch.float64).roll(1, dims=1)
    records = []
    for image in range(12):
        for case in ('clean_s0', 'gamma_s1'):
            p = torch.randn(8, dtype=torch.float64)
            records.append(dict(image_id=image, block=image//3, case=case,
                                pseudo_gradient=p.tolist(), task_gradient=(p @ q).tolist()))
    return records, q


def test_known_reflection_recovery_row_convention_norm_and_serialized_reload():
    import json
    rows, q = pairs()
    fits = json.loads(json.dumps(fit_folds(rows)))
    for f in fits.values():
        got = torch.tensor(f['Q'], dtype=torch.float64)
        torch.testing.assert_close(got, q, rtol=0, atol=1e-10)
        assert f['determinant'] < 0
        p = torch.tensor(rows[0]['pseudo_gradient'], dtype=torch.float64)
        torch.testing.assert_close(p @ got, torch.tensor(rows[0]['task_gradient'], dtype=torch.float64))
        torch.testing.assert_close((p @ got).norm(), p.norm())
        assert not set(f['train_image_ids']) & set(f['heldout_image_ids'])
    assert heldout_alignment(rows, fits)['groups']['overall']['improved_fraction_defined'] == 1


def test_heldout_task_gradient_changes_cannot_change_own_matrix():
    rows, _ = pairs(); original = fit_folds(rows)
    changed = copy.deepcopy(rows)
    for r in changed:
        if r['block'] == 0:
            r['task_gradient'] = [v*7+3 for v in r['task_gradient']]
    assert fit_folds(changed)['0']['Q'] == original['0']['Q']
    rows[1]['block'] = 1
    with pytest.raises(AssertionError, match='image conditions'):
        fit_folds(rows)


def test_zero_vectors_retained_and_reported():
    rows, _ = pairs(); rows[0]['pseudo_gradient'] = [0.]*8
    fits = fit_folds(rows)
    assert fits['1']['zero_pseudo_rows'] == 1 and fits['1']['excluded_rows'] == 0
    summary = heldout_alignment(rows, fits)['groups']['overall']
    assert summary['undefined_zero'] == 1 and summary['episodes'] == 24


class Loss(nn.Module):
    def __init__(self, empty=False):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(.3)); self.boxes = torch.ones(0 if empty else 1, 4)
    def forward(self, original, enhanced):
        return (enhanced-self.weight).square().mean() if len(self.boxes) else enhanced.sum()*0


@pytest.mark.parametrize('empty', [False, True])
def test_runtime_identity_equivalence_k3_frozen_only_phi_and_no_labels(empty):
    x = torch.linspace(.1,.8,3*9*11).reshape(1,3,9,11); isp = DifferentiableISP()
    pseudo, clip = Loss(empty), Loss(); original = pseudo.weight.detach().clone()
    current = adapt_clip_radius(x, isp, pseudo, clip)
    candidate = adapt_gradient_transport(x, isp, pseudo, clip, torch.eye(8))
    assert torch.equal(current.phi, candidate.phi) and torch.equal(current.enhanced, candidate.enhanced)
    assert torch.equal(isp.phi, torch.zeros(8)) and isp.phi.grad is None
    assert torch.equal(pseudo.weight, original) and pseudo.weight.grad is None and not pseudo.weight.requires_grad
    assert clip.weight.grad is None and not clip.weight.requires_grad
    q = torch.eye(8).roll(1, dims=1)
    rotated = adapt_gradient_transport(x, isp, pseudo, clip, q)
    for d in rotated.diagnostics:
        torch.testing.assert_close(torch.tensor(d['transported_gradient']), torch.tensor(d['detector_gradient']) @ q)
        assert d['norm_preservation_passed']
    assert sum(d['update_applied'] for d in rotated.diagnostics) == 3
    if empty:
        assert torch.equal(rotated.phi, torch.zeros(8)) and torch.equal(rotated.enhanced, isp(x, torch.zeros(8)))
    assert set(inspect.signature(adapt_gradient_transport).parameters) == {'image','isp','detector_loss','clip_loss','q','steps','lr','eps'}


def test_fixed_point15_performance_gate_and_isolation():
    from taisp.analysis.run_t018a import CASE_NAMES
    methods = ['no_adapt', 'current_ours', 'grad_transport_ours']
    metrics = {g: {f'{c}_{m}': {k: .3+(.0016 if m == methods[-1] else 0) for k in ('AP','AP50','AP75')}
                   for c in CASE_NAMES for m in methods} for g in ['aggregate']+[f'block{i}' for i in range(4)]}
    assert transport_advancement(metrics, True)['gate']['passed']
    assert not transport_advancement(metrics, False)['gate']['passed']
    for c in CASE_NAMES[:-1]:
        metrics['aggregate'][f'{c}_{methods[-1]}']['AP'] = .3014
    assert not transport_advancement(metrics, True)['gate']['passed']
