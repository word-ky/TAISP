import ast
import inspect
import os
from pathlib import Path

import pytest
import torch

from taisp.losses.detector_native import DetectorNativeLoss
from taisp.tta import adapt, AdaptConfig


def test_target_not_in_deployment_imports_or_objective_signature():
    assert list(inspect.signature(DetectorNativeLoss.forward).parameters) == ['self', 'original', 'enhanced']
    assert list(inspect.signature(DetectorNativeLoss.__init__).parameters) == ['self', 'detector', 'support', 'variant', 'js_coefficient']
    for folder in ('taisp/tta', 'taisp/losses', 'taisp/models'):
        for path in Path(folder).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert 'analysis' not in (node.module or '')
                if isinstance(node, ast.Import):
                    assert all('analysis' not in alias.name for alias in node.names)


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit real-model integration')
def test_real_target_oracle_and_source_adaptation_isolation():
    from taisp import DifferentiableISP
    from taisp.analysis.fcos_target import load_fcos_target, target_task_loss, target_metadata
    from taisp.models.detector import load_detector
    torch.manual_seed(42)
    torch.set_num_threads(1)
    # CPU source repeats avoid the documented CUDA backward nondeterminism.
    source, isp = load_detector('cpu'), DifferentiableISP()
    image = .2+.5*torch.rand(1, 3, 257, 319)
    support = {'base': {'boxes': torch.tensor([[20., 30., 170., 230.]]),
                        'labels': torch.tensor([1]), 'scores': torch.tensor([.8])}}
    guidance = DetectorNativeLoss(source, support, 'det_pseudo')
    config = AdaptConfig(steps=1, regularization_weight=0)
    absent = adapt(image, isp, guidance, config=config)
    target = load_fcos_target('cuda:0')
    state = {k: v.cpu().clone() for k, v in target.state_dict().items()}
    def forbidden(*args):
        raise AssertionError('target called during source adaptation')
    hook = target.model.register_forward_pre_hook(forbidden)
    present = adapt(image, isp, guidance, config=config)
    hook.remove()
    torch.testing.assert_close(absent.phi, present.phi, atol=0, rtol=0)
    torch.testing.assert_close(absent.enhanced, present.enhanced, atol=0, rtol=0)
    x, gpu_isp = image.cuda(), DifferentiableISP().cuda()
    annotations = [{'boxes': support['base']['boxes'].cuda(), 'labels': support['base']['labels'].cuda()}]
    rng = torch.cuda.get_rng_state()
    loss, components = target_task_loss(target, gpu_isp(x), annotations)
    gradient = torch.autograd.grad(loss, gpu_isp.phi)[0]
    assert set(components) == {'classification', 'bbox_regression', 'bbox_ctrness'}
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert all(not m.training for m in target.modules())
    with torch.no_grad():
        prediction = target(x)[0]
    assert set(prediction) == {'boxes', 'scores', 'labels'}
    assert all(not p.requires_grad and p.grad is None for p in target.parameters())
    assert all(torch.equal(v.cpu(), state[k]) for k, v in target.state_dict().items())
    meta = target_metadata(target)
    assert meta['sha256'] == '99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7'
