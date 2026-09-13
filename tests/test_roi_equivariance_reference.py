import copy

import pytest

from taisp.analysis.roi_equivariance_reference import metrics, aggregate, verify_lock
from taisp.analysis.roi_equivariance import write, sha


def test_scores_and_zero_convention():
    candidate = {'support_count': 1, 'gradients': {'roi_feat_eq': {'object': [3., 4.]}}}
    refs = {'task': {'object': [2., 0.], 'global': [0., 2.]},
            'pseudo': {'object': [0., 1.], 'global': [0., -1.]}}
    m = metrics(candidate, refs)['roi_feat_eq']
    assert m['S_c'] == pytest.approx(1.2)
    assert m['Delta_global'] == pytest.approx(3.2)
    assert m['cosine_candidate_task'] == pytest.approx(.6)
    candidate['gradients']['roi_feat_eq']['object'] = [0., 0.]
    m = metrics(candidate, refs)['roi_feat_eq']
    assert m['S_c'] == 0 and m['cosine_candidate_task'] is None and m['material_near_zero']


def test_frozen_conjunction_no_clean_only_promotion():
    rows = []
    for i in range(32):
        m = {'S_c': 1., 'Delta_global': 1., 'Delta_obj': .5, 'zero_gradient': False,
             'near_zero_gradient': False, 'material_near_zero': False}
        rows.append({'case': 'gamma_s2' if i % 2 else 'clean_s0', 'block': i//8,
            'integrity_passed': True, 'metrics': {k: copy.deepcopy(m) for k in ('roi_feat_eq','roi_logit_eq')}})
    assert aggregate(rows)['disposition'] == 'exact_tie_needs_review'
    for i in range(1, 32, 2):
        for k in rows[i]['metrics']:
            rows[i]['metrics'][k]['Delta_global'] = -.1
    result = aggregate(rows)
    assert result['nomination'] is None
    assert result['disposition'] == 'close_fixed_roi_flip_equivariance_family'


def test_candidate_receipts_checked_before_reference_inputs(tmp_path):
    for i in range(34):
        write(tmp_path/f'{i}.json', {})
    write(tmp_path/'records.json', [])
    lock = {'candidate_commit': 'preexisting-commit',
            'files': {p.name: sha(p) for p in tmp_path.glob('*.json')}}
    assert verify_lock(tmp_path, lock) == []
    write(tmp_path/'0.json', {'changed': True})
    with pytest.raises(AssertionError):
        verify_lock(tmp_path, lock)
