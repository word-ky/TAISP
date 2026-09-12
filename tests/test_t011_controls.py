import copy
import hashlib
import json

import numpy as np
import pytest

from taisp.analysis.random_controls import matched_mask, randomization, selection_information, stratum_indices


def fixture():
    rows = [{'image_id': i, 'case': case} for i in range(1, 9) for case in ('a', 'b')]
    groups = {'aggregate': list(range(1, 9)), 'block_1': [1, 2, 3, 4], 'block_2': [5, 6, 7, 8]}
    candidate = np.array([r['image_id'] % 2 == 0 for r in rows])
    return rows, groups, candidate


def test_exact_stratum_counts_repeatability_seed_variation_and_identity_order():
    rows, groups, candidate = fixture()
    strata = stratum_indices(rows, groups)
    first = matched_mask(rows, strata, candidate, 2026091600)
    assert np.array_equal(first, matched_mask(rows, strata, candidate, 2026091600))
    assert not np.array_equal(first, matched_mask(rows, strata, candidate, 2026091601))
    for indexes in strata.values():
        assert first[indexes].sum() == candidate[indexes].sum() == 2
    order = np.arange(len(rows))[::-1]
    reordered = [rows[j] for j in order]
    other = matched_mask(reordered, stratum_indices(reordered, groups), candidate[order], 2026091600)
    assert np.array_equal(other, first[order])


def test_zero_full_strata_and_forbidden_outcome_fields():
    rows, groups, candidate = fixture()
    strata = stratum_indices(rows, groups)
    candidate[strata['a', 'block_1']] = False
    candidate[strata['b', 'block_2']] = True
    mask = matched_mask(rows, strata, candidate, 2026091600)
    assert not mask[strata['a', 'block_1']].any()
    assert mask[strata['b', 'block_2']].all()
    poisoned = copy.deepcopy(rows)
    for row in poisoned:
        row.update(scores={'support_confidence': 999}, phi_norm_3=999, annotations='forbidden', target_AP=-999)
    assert np.array_equal(mask, matched_mask(poisoned, strata, candidate, 2026091600))


def test_randomization_ties_corrected_tail_and_not_a_confidence_interval():
    r = randomization(2., [0., 1., 2., 2.])
    assert r['random']['mean'] == 1.25
    assert r['random']['median'] == 1.5
    assert r['strict_empirical_percentile'] == 50.
    assert r['random_equal_candidate'] == 2
    assert r['one_sided_tail'] == pytest.approx(3/5)
    assert r['candidate_minus_random']['mean'] == .75
    full = randomization(201., np.arange(200))
    assert full['one_sided_tail'] == pytest.approx(1/201)


def test_both_targets_strict_p95_block_medians_and_clean_are_required():
    assert selection_information([.1, .2], [.09, .19], [4, 5], [-.1, 0, .01])
    assert not selection_information([.1, .2], [.1, .19], [5, 5], [0, 0, 0])
    assert not selection_information([.1, .2], [.09, .19], [4, 3], [0, 0, 0])
    assert not selection_information([.1, .2], [.09, .19], [5, 5], [-.1001, 0, 0])
    assert not selection_information([0, .2], [-.1, .19], [5, 5], [0, 0, 0])


def test_preparation_preserves_candidate_vector_and_all_200_masks(tmp_path):
    from scripts.prepare_t011 import prepare
    from taisp.analysis.random_controls import CANDIDATE
    rows, groups, candidate = fixture()
    for r in rows:
        r.update(case='clean_s0' if r['case'] == 'a' else 'gamma_s1', phi_norm_3=.1,
                 source_setup_seconds=.05, adapt_seconds_3=.3)
    original, output = tmp_path/'original', tmp_path/'output'
    original.mkdir()
    (original/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    np.savez_compressed(original/'decisions.npz', selected=candidate[None])
    (original/'grid.json').write_text(json.dumps([{'name': CANDIDATE,
        'decision_sha256': hashlib.sha256(candidate.tobytes()).hexdigest(), 'cutoff': {'cutoff_score': .5}}]))
    (original/'manifest.json').write_text(json.dumps({'source_revision': 'fixture', 'full_cohort': False,
        'image_ids': list(range(1, 9)), 'cases': ['clean_s0', 'gamma_s1'], 'groups': groups}))
    prepare(original, output)
    masks = np.load(output/'decisions.npz')['selected']
    assert masks.shape == (203, 16) and np.array_equal(masks[2], candidate)
    assert not masks[0].any() and masks[1].all()
    for indexes in stratum_indices(rows, groups).values():
        assert np.all(masks[3:, indexes].sum(axis=1) == candidate[indexes].sum())
    meta = json.loads((output/'manifest.json').read_text())
    assert meta['labels_or_prediction_files_read'] is False and meta['random_draws'] == 200


def test_configuration_batches_cover_each_config_once_and_keep_old_default():
    from scripts.analyze_t010 import configuration_ranges
    assert configuration_ranges(44, None) == [None]
    ranges = configuration_ranges(203, 25)
    assert len(ranges) == 9 and ranges[-1] == (200, 203)
    assert [j for start, stop in ranges for j in range(start, stop)] == list(range(203))
