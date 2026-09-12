import copy
import math

from taisp.analysis.deterministic_replay import first_mismatch, determinism_gate, spearman, decision


def test_exact_gate_detects_one_ulp_and_signed_zero_without_tolerance():
    r = {'episodes':[{'loss':1.,'phi':[0.,-.1]}], 'geometry':{'cosine':-.7}}
    repeats = [copy.deepcopy(r) for _ in range(3)]
    assert determinism_gate(repeats)['passed']
    repeats[2]['episodes'][0]['loss'] = math.nextafter(1.,2.)
    gate = determinism_gate(repeats)
    assert not gate['passed'] and gate['first_mismatch_repeat'] == 2
    assert gate['first_mismatch']['path'] == 'episodes[0].loss'
    mismatch = first_mismatch([0.],[-0.])
    assert mismatch['reference_float64_hex'] != mismatch['actual_float64_hex']


def test_spearman_uses_average_tie_ranks_and_handles_constant():
    assert abs(spearman([1.,1.,3.,4.],[-1.,-1.,-3.,-4.])+1) < 1e-12
    assert spearman([1.,1.,1.],[2.,3.,4.]) is None


def test_predeclared_decision_branches_no_fitted_threshold():
    def probes(joint,clean,corrupt):
        return {n:{'delta_comparison':{'actual_group_means':{'clean':d[0],'corrupted':d[1]}}}
                for n,d in zip(('joint','clean','corrupted'),(joint,clean,corrupt))}
    assert decision(probes((-1,-1),(-1,1),(-1,-1)))['rule'] == 'functionally_active_source_objective_conflict'
    assert decision(probes((-1,-1),(-1,-1),(1,-1)))['corrupt_only_cross_harm']
    assert decision(probes((-1,-1),(-1,-1),(-1,-1)))['rule'].startswith('all_three_improve_both')
    assert decision(probes((-1,1),(-1,-1),(-1,-1)))['rule'] == 'asymmetric_joint_finite_step_interaction'
    assert decision(probes((0,0),(0,0),(0,0)))['rule'] == 'unresolved_pattern'
