import pytest

from taisp.analysis.dose import half_dose_criteria, ratio_summary
from taisp.analysis.replication import ap_contrasts


def test_half_dose_all_five_criteria_and_boundaries():
    args = ([.01, .02], [4, 4], [True, True], [4, 4], [-.1, 0, 0], .65, 1.)
    assert all(half_dose_criteria(*args).values())
    for index, replacement in [(0, [0, .02]), (1, [3, 4]), (2, [False, True]),
                               (3, [4, 3]), (4, [-.10001, 0, 0]), (5, .65001)]:
        changed = list(args)
        changed[index] = replacement
        assert not all(half_dose_criteria(*changed).values())


def test_paired_ratios_keep_zero_denominators_explicit():
    q = ratio_summary([1., 0., 2., 2.], [2., 0., 0., 2.])
    assert q['positive_denominator_count'] == 2
    assert q['both_zero_count'] == q['nonzero_numerator_zero_denominator_count'] == 1
    assert q['ratio']['mean'] == .75


def test_half_endpoint_contrasts_reuse_existing_macro_with_negative_clean():
    cases = ['gamma_s1', 'gamma_s2', 'clean_s0']
    variants = ['no_adapt', 'det_pseudo_clip_radius', 'det_pseudo_half_dose']
    metrics = {f'target_{c}_{v}': {m: .3 + (delta if v == variants[-1] else 0)
        for m in ('AP', 'AP50', 'AP75')} for c, delta in zip(cases, [.02, -.01, -.03]) for v in variants}
    q = ap_contrasts(metrics, cases, ['target'], variants, references=variants[:2])
    r = q['target'][variants[-1]]['no_adapt']
    assert r['macro_corruption_AP_delta'] == pytest.approx(.5)
    assert r['positive_corruption_count'] == 1 and r['clean_AP_delta'] == pytest.approx(-3.)
