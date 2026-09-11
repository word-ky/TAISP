import torch

from taisp.analysis.run_t003 import choose_variant
from taisp.losses.conditioned_clip import CLIPConditioner
from taisp.losses.semantic import MockImageEncoder


def test_oracle_variants_are_explicit_analysis_choices():
    c = CLIPConditioner(MockImageEncoder(), torch.ones(3), torch.eye(3))
    e = c.prepare(torch.full((1, 3, 6, 7), .3))
    soft, gate = choose_variant('soft', e, c, 'gamma')
    assert soft is e.guidance and gate is None
    sg, mask = choose_variant('soft_gate', e, c, 'gamma')
    assert sg is soft
    torch.testing.assert_close(mask, e.gate)
    og, mask = choose_variant('oracle_prompt', e, c, 'contrast')
    assert mask is None
    both, mask = choose_variant('oracle_both', e, c, 'contrast')
    torch.testing.assert_close(og.direction, both.direction)
    torch.testing.assert_close(mask, torch.tensor([0., 0, 0, 0, 1, 0, 1, 0]))
    mixed, mask = choose_variant('soft_oracle_gate', e, c, 'color_cast')
    assert mixed is soft
    torch.testing.assert_close(mask, torch.tensor([0., 1, 1, 1, 0, 0, 0, 0]))
