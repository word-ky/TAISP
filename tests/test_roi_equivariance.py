import ast
import inspect

import torch

from taisp import DifferentiableISP
from taisp.analysis import roi_equivariance as eq
from taisp.analysis.common_jacobian import common_reference


def test_flip_mapping_and_roundtrip_preserves_pair_order():
    boxes = torch.tensor([[1., 2., 5., 8.], [0., 0., 3., 2.]])
    mapped = eq.flip_boxes(boxes, 10)
    assert torch.equal(mapped, torch.tensor([[5., 2., 9., 8.], [7., 0., 10., 2.]]))
    assert torch.equal(eq.flip_boxes(mapped, 10), boxes)


def test_objectives_identical_and_permuted_pairs():
    torch.manual_seed(61)
    z, logits = torch.randn(3, 8), torch.randn(3, 5)
    losses = eq.paired_losses(z, z, logits, logits)
    for v in losses.values():
        assert torch.allclose(v, torch.zeros(3), atol=2e-7)
    zf, lf = torch.randn_like(z), torch.randn_like(logits)
    a = eq.paired_losses(z, zf, logits, lf)
    b = eq.paired_losses(z[[2, 0, 1]], zf[[2, 0, 1]], logits[[2, 0, 1]], lf[[2, 0, 1]])
    for k in a:
        assert torch.equal(a[k][[2, 0, 1]], b[k])
        assert torch.isfinite(a[k]).all() and (a[k] >= 0).all()


def test_empty_support_is_exact_zero_without_detector_call():
    image = torch.rand(1, 3, 6, 7)
    cs, values, pairs = eq.candidate_cotangents(None, image, torch.empty(0, 4))
    refs, _ = eq.common_jvp(image, DifferentiableISP(), torch.zeros(1, 1, 6, 7), cs)
    assert pairs == []
    for k in cs:
        assert torch.count_nonzero(cs[k]) == 0 and values[k]['empty_support']
        assert refs[k]['object'] == [0.] * 8


def test_common_jvp_is_identical_to_accepted_a1():
    torch.manual_seed(62)
    image = .2+.6*torch.rand(1, 3, 12, 14)
    isp = DifferentiableISP()
    mask, _ = eq.support_mask(image, torch.tensor([[1., 2., 7., 9.]]))
    cs = {'candidate': torch.randn_like(image), 'reference': torch.randn_like(image)}
    actual, _ = eq.common_jvp(image, isp, mask, cs)
    expected, _ = common_reference(image, isp, mask, cs)
    assert actual == expected


def test_candidate_import_and_input_boundary():
    tree = ast.parse(inspect.getsource(eq))
    imported = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(any(x in m for x in ('oracle', 'common_jacobian', 'source_meta', 'reference')) for m in imported)
    assert list(inspect.signature(eq.candidate_cotangents).parameters) == ['detector', 'image', 'boxes']
