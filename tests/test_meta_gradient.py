import ast
import inspect

import torch

from taisp.analysis import meta_gradient
from taisp.analysis.oracle import detector_task_loss
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.training import initialization
from taisp.tta import trust_radius


def test_exact_reference_matches_fd_and_fo_forward():
    torch.set_num_threads(1)
    row = meta_gradient.audit_episode(6)
    assert row['all_gradients_finite']
    assert row['cos_exact_fd'] >= .999
    torch.testing.assert_close(torch.tensor(row['exact']), torch.tensor(row['fd']), rtol=1e-5, atol=1e-8)
    assert row['trajectory_max_abs_difference'] == 0
    assert row['image_max_abs_difference'] == 0


def test_predeclared_gate_boundaries():
    rows = [dict(cos_fo_exact=.5, cos_exact_fd=1., all_gradients_finite=True) for _ in range(12)]
    for row in rows[:3]:
        row['cos_fo_exact'] = -.1
    assert meta_gradient.summarize(rows)['part_b_allowed']
    rows[3]['cos_fo_exact'] = 0.
    assert not meta_gradient.summarize(rows)['part_b_allowed']
    rows[3]['cos_fo_exact'] = .5
    rows[0]['cos_exact_fd'] = .998
    assert not meta_gradient.summarize(rows)['part_b_allowed']
    rows[0].update(cos_exact_fd=1., all_gradients_finite=False)
    assert not meta_gradient.summarize(rows)['part_b_allowed']


def test_labels_and_exact_reference_stay_out_of_deployment():
    assert 'targets' in inspect.signature(detector_task_loss).parameters
    assert list(inspect.signature(trust_radius.adapt_clip_radius).parameters) == [
        'image', 'isp', 'detector_loss', 'clip_loss', 'steps', 'lr', 'eps']
    assert list(inspect.signature(DetectorNativeLoss.forward).parameters) == ['self', 'original', 'enhanced']
    for module in (trust_radius, initialization):
        tree = ast.parse(inspect.getsource(module))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not names & {'targets', 'annotations', 'detector_task_loss', 'reference_unroll'}
        imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any('analysis' in name for name in imports)
