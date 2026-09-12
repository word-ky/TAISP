import torch

from taisp.analysis.gradient_conflict import geometry, directions, one_step, delta_comparison, conditioning
from taisp.models.parameter_predictor import ParameterPredictor


def test_geometry_and_mean_direction_prediction():
    gradients = torch.tensor([[1., 0.], [-2., 0.]]*4)
    result = geometry(gradients, gradients)
    assert result['same_image_clean_corrupted_cosines'] == [-1.]*4
    assert result['head']['cosine'] == -1 and result['head']['dot'] == -2
    assert abs(result['head']['clean_to_corrupted_norm_ratio']-.5) < 1e-10
    vector = directions(gradients)['joint']
    torch.testing.assert_close(vector, torch.tensor([-.5, 0.]))
    predicted = -1e-3*(gradients@vector)
    torch.testing.assert_close(predicted, torch.tensor([.0005, -.001]*4))
    summary = delta_comparison(predicted, predicted*2)
    assert summary['sign_agreement_fraction'] == 1
    assert abs(summary['pearson_r']-1) < 1e-12


def test_independent_copies_literal_sgd_and_centered_energy():
    torch.manual_seed(20260913)
    original = ParameterPredictor().double()
    gradient = torch.linspace(-.2, .3, original.head.weight.numel()+8, dtype=torch.double)
    a, b = one_step(original, gradient), one_step(original, -gradient)
    flat = lambda p: torch.cat((p.head.weight.flatten(), p.head.bias.flatten()))
    torch.testing.assert_close(flat(a), -1e-3*gradient)
    torch.testing.assert_close(flat(b), 1e-3*gradient)
    assert torch.count_nonzero(flat(original)) == 0
    assert all(torch.equal(p, q) for p, q in zip(a.features.parameters(), original.features.parameters()))
    x = torch.linspace(.1, .7, 3*9*11, dtype=torch.double).reshape(1, 3, 9, 11)
    same = conditioning(a, [x]*8)
    assert max(same['population_std_per_coordinate']) < 1e-18
    assert same['centered_energy_over_total'] < 1e-25
    assert abs(same['weight_term_energy']+same['bias_energy']+same['cross_energy']-same['total_output_energy']) < 1e-18
    varied = conditioning(a, [x*(.5+.05*i) for i in range(8)])
    assert varied['centered_energy_over_total'] > 0
    torch.testing.assert_close(torch.tensor(varied['outputs']), torch.cat([a(x*(.5+.05*i)) for i in range(8)]).float())
