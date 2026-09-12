import ast
import inspect

import torch

from taisp.analysis import repeatability


def test_population_repeat_stats_and_descriptive_effect_count():
    values = [1.,2.,3.,4.]
    result = repeatability.effect_scale(2.,values)
    assert result['control']['mean'] == 2.5
    assert abs(result['control']['std_population']-1.25**.5) < 1e-12
    assert result['control']['range'] == 3
    assert result['count_equal_or_larger_abs_change'] == 2
    assert result['repeat_denominator_including_r0'] == 4
    zero = repeatability.effect_scale(1.,[2.]*12)
    assert zero['abs_effect_over_std'] is None and zero['abs_effect_over_range'] is None


def test_common_mode_reconstruction_and_translation_invariant_distances():
    features = torch.arange(128,dtype=torch.double).reshape(8,16)/100
    gw = torch.arange(128,dtype=torch.float32).reshape(8,16)/1000
    gb = torch.linspace(-.1,.1,8)
    flat = torch.cat((gw.flatten(),gb))
    learned = torch.zeros_like(flat).add_(flat,alpha=-1e-3).double()
    wh = features@learned[:-8].reshape(8,16).T
    output = wh+learned[-8:]
    prior = {'gradient_audit':{'episodes':[{'head_weight_gradient':gw.tolist(),'head_bias_gradient':gb.tolist()} for _ in range(8)]},
             'probes':{'joint':{'conditioning':{'features':features.tolist(),'weight_term':wh.tolist(),
                        'bias':learned[-8:].tolist(),'outputs':output.tolist()}}}}
    result = repeatability.common_mode(prior)
    assert result['max_saved_output_difference'] < 1e-9
    assert abs(result['common_mode_removed_upper_bound']['centered_energy_fraction']-1) < 1e-12
    for name in ('wh_only','bias_removed','common_mode_removed_upper_bound'):
        torch.testing.assert_close(torch.tensor(result[name]['pairwise_distances']),
                                   torch.tensor(result['full_original_output']['pairwise_distances']))
    assert result['wh_only']['centered_energy_fraction'] < 1


def test_primary_module_never_calls_optimizer_or_probe():
    tree = ast.parse(inspect.getsource(repeatability))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any('optim' in name or name.endswith('.step') or name == 'one_step' for name in calls)
