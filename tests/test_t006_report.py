import torch

from scripts.report_t006 import detector_rows, joint_outcome
from taisp.analysis.run_t006 import detector_diagnostics


def test_target_projection_joint_outcomes_and_signed_taylor():
    row = {'g_sem': [1., 2.], 'norm_matched': {'gradient': [.1, .2]},
           'source': {'g_det': [3., 4.], 'gradient_cosine': .7, 'det_loss_delta_sem1': -1.,
                      'norm_matched': {'det_loss_delta': 0.}},
           'target': {'g_det': [5., 6.], 'gradient_cosine': .8, 'det_loss_delta_sem1': 1.,
                      'norm_matched': {'det_loss_delta': -2.}}}
    target = detector_rows([row], 'target', True)[0]
    assert target['g_det'] == [5., 6.] and target['g_sem'] == [.1, .2]
    assert target['det_loss_delta_sem1'] == -2.
    assert joint_outcome(row) == 'source_only'
    assert joint_outcome(row, True) == 'target_only'
    row['target']['det_loss_delta_sem1'] = -1.
    assert joint_outcome(row) == 'both'
    row['source']['det_loss_delta_sem1'] = row['target']['det_loss_delta_sem1'] = 0.
    assert joint_outcome(row) == 'neither'
    a, b = torch.tensor([1., -2.]), torch.tensor([3., 4.])
    d = detector_diagnostics(a, b, 2., {'1': 1., '3': 0., 'matched': 3.}, a/2, .1)
    assert d['coordinate_dot'] == [3., -8.]
    assert d['linear_prediction'] == .5 and d['norm_matched']['linear_prediction'] == .25
    assert d['det_loss_delta_sem1'] == -1. and d['norm_matched']['det_loss_delta'] == 1.
