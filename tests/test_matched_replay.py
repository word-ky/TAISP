import ast
import inspect

from taisp.analysis import matched_replay as replay


def test_fixed_rotation_and_inside_pair_order():
    schedule = replay.schedule()
    assert len(schedule) == 8
    for i,row in enumerate(schedule):
        assert row['cycle'] == i+1
        assert row['roles'] == (['baseline','probe'] if i%2==0 else ['probe','baseline'])
    assert schedule[0]['pairs'] == ['null','joint','clean','corrupted']
    assert schedule[3]['pairs'] == ['corrupted','null','joint','clean']
    for name in schedule[0]['pairs']:
        assert sorted(row['pairs'].index(name) for row in schedule) == [0,0,1,1,2,2,3,3]


def test_seven_of_eight_and_strict_null_floor_boundaries():
    assert replay.resolved_effect([-2.]*7+[2.],[1.]*8)['direction'] == 'improvement'
    assert not replay.resolved_effect([-2.]*6+[2.]*2,[1.]*8)['resolved']
    assert not replay.resolved_effect([-1.]*8,[1.]*8)['resolved']
    assert not replay.resolved_effect([0.]*8,[0.]*8)['resolved']


def test_logical_roles_survive_reversed_execution_order_and_null_correction():
    cycles = []
    for spec in replay.schedule():
        pairs = []
        for name in spec['pairs']:
            evaluations = [{'role':role,'episodes':[{'outer_loss':10.+(1. if name=='null' else -2.)*(role=='probe')} for _ in range(8)]}
                           for role in spec['roles']]
            pairs.append({'pair':name,'evaluations':evaluations})
        cycles.append({'cycle':spec['cycle'],'pairs':pairs})
    summary = replay.summarize(cycles)
    assert summary['null']['joint']['signed']['values'] == [1.]*8
    assert summary['effects']['joint']['clean']['paired']['values'] == [-2.]*8
    assert summary['effects']['joint']['clean']['corrected']['values'] == [-3.]*8
    assert summary['decision']['rule'].startswith('resolved_both_group_improvement')
    summary['effects']['clean']['corrupted']['corrected']['direction'] = 'worsening'
    assert replay.decision(summary['effects'])['rule'] == 'functionally_active_source_objective_conflict'
    for p in summary['effects'].values():
        for g in p.values():
            g['corrected']['direction'] = 'unresolved'
    assert replay.decision(summary['effects'])['rule'] == 'measurement_limited_unresolved_or_mixed'


def test_checkpoint_replay_has_no_optimizer_or_gradient_direction_recomputation():
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(inspect.getsource(replay))) if isinstance(n,ast.Call)]
    assert not any('optim' in c or c.endswith('.step') or c in ('one_step','directions','torch.autograd.grad') for c in calls)
