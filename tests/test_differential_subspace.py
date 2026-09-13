import copy

import numpy as np
import pytest

from taisp.analysis.differential_subspace import decompose, episode, triage, describe
from taisp.analysis.spatial_action import geometry


@pytest.mark.parametrize('obj,bg', [([1., 2.], [1., 2.]), ([1., 2.], [-1., -2.]),
                                  ([1., 2.], [0., 0.]), ([0., 0.], [0., 0.]),
                                  ([2., -3.], [-4., 5.])])
def test_orthogonal_decomposition_and_pythagoras(obj, bg):
    receipt, shared, diff = decompose(obj, bg)
    np.testing.assert_allclose(shared + diff, np.r_[obj, bg], atol=1e-14)
    assert abs(np.dot(shared, diff)) < 1e-14
    assert receipt['vector_reconstruction_error'] < 1e-14
    assert receipt['energy_reconstruction_error'] < 1e-12
    np.testing.assert_allclose(receipt['s'], (np.array(obj) + bg) / 2)
    np.testing.assert_allclose(receipt['d'], (np.array(obj) - bg) / 2)


def test_productivity_components_known_zero_and_one_region_cases():
    cases = [([1., 0.], [1., 0.], [2., 0.], [2., 0.]),
             ([1., 0.], [-1., 0.], [-2., 0.], [2., 0.]),
             ([1., 0.], [0., 0.], [3., 0.], [0., 0.]),
             ([1., 2.], [-3., 4.], [0., 0.], [0., 0.])]
    for values in cases:
        saved = geometry(*values)['D_spatial']
        r = episode(*values, saved)
        assert r['metrics']['D_reconstruction_error'] < 1e-12
    shared = episode(*cases[0], geometry(*cases[0])['D_spatial'])
    assert shared['metrics']['cos_diff'] is None and shared['metrics']['C_diff'] == 0
    one_region = episode(*cases[2], geometry(*cases[2])['D_spatial'])
    np.testing.assert_allclose(one_region['metrics']['R_extra'], np.sqrt(2), atol=3e-12)
    np.testing.assert_allclose(one_region['metrics']['C_shared'], .5, atol=1e-12)
    np.testing.assert_allclose(one_region['metrics']['C_diff'], .5, atol=1e-12)
    opposing = episode(*cases[1], geometry(*cases[1])['D_spatial'])
    assert opposing['metrics']['cos_diff'] == pytest.approx(-1)
    assert opposing['diff_dot_sign'] == -1
    summary = describe([{'analysis': shared}])
    assert summary['zero_norm_counts']['either_diff'] == 1
    assert summary['metrics']['cos_diff']['valid_count'] == 0
    assert summary['metrics']['cos_diff']['median'] is None
    assert summary['metrics']['C_diff']['positive_count'] == 0


def test_frozen_triage_inclusive_and_strict_boundaries():
    scopes = {'overall': {'metrics': {'R_extra': {'median': 1.10},
                                      'C_diff': {'positive_count': 20, 'median': .01}}},
              'corrupted': {'metrics': {'R_extra': {'median': 1.10},
                                        'C_diff': {'positive_count': 10, 'median': .01}}}}
    scopes.update({f'block{i}': {'metrics': {'R_extra': {'median': 1.06 if i < 3 else 1.05},
                                            'C_diff': {'median': .01 if i < 3 else 0.}}} for i in range(4)})
    assert triage(scopes)['task_relevant'] and triage(scopes)['pseudo_differentially_useful']
    for group, metric, key, value in [('overall', 'R_extra', 'median', 1.099),
                                     ('corrupted', 'R_extra', 'median', 1.099),
                                     ('block2', 'R_extra', 'median', 1.05)]:
        changed = copy.deepcopy(scopes)
        changed[group]['metrics'][metric][key] = value
        assert triage(changed)['decision'] == 'close_partition'
    for group, key, value in [('overall', 'positive_count', 19), ('corrupted', 'positive_count', 9),
                              ('overall', 'median', 0.), ('block2', 'median', 0.)]:
        changed = copy.deepcopy(scopes)
        changed[group]['metrics']['C_diff'][key] = value
        assert triage(changed)['decision'] == 'close_fixed_pseudo_spatial_branch'
