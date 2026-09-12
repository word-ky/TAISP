import copy
import json
import math

import numpy as np
import pytest

from taisp.analysis.gating import compose_predictions, extract_scores, latency_estimate, rank_mask


def receipt():
    return {'diagnostics': [{'detector_gradient': [3., 4.], 'clip_gradient': [2., 0.],
                            'detector_gradient_norm': 5., 'clip_gradient_norm': 2., 'total': .7}],
            'support_count': 2, 'support': {'scores': [.8, .6]}}


def test_exact_identity_scores_ignore_postupdate_and_evaluation_fields():
    r = receipt()
    expected = {'clip_norm': 2., 'det_norm': 5., 'log_norm_ratio': math.log((5+1e-12)/(2+1e-12)),
                'gradient_cosine': .6, 'pseudo_loss': .7, 'support_count': 2, 'support_confidence': .7}
    assert extract_scores(r) == pytest.approx(expected)
    r.update({'family': 'clean', 'severity': 9, 'image_id': 999, 'annotations': 'forbidden',
              'target_AP': 999, 'phi_norm_3': 999})
    r['diagnostics'].append({'total': 999, 'gradient_norm': 999})
    assert extract_scores(r) == pytest.approx(expected)


def test_zero_gradient_and_empty_support_conventions():
    r = receipt()
    r['diagnostics'][0].update(detector_gradient=[0., 0.], detector_gradient_norm=0., total=0.)
    r.update(support_count=0, support={'scores': []})
    s = extract_scores(r)
    assert s['gradient_cosine'] == s['support_confidence'] == s['pseudo_loss'] == 0
    assert s['log_norm_ratio'] == pytest.approx(math.log(1e-12/(2+1e-12)))


def test_rank_orientations_ties_and_exact_coverage():
    values, ids = [1, 3, 3, 0, 3, 2, 4, 5], [9, 7, 2, 8, 2, 6, 5, 4]
    high, h = rank_mask(values, ids, .5, 'high')
    low, _ = rank_mask(values, ids, .5, 'low')
    assert np.flatnonzero(high).tolist() == [2, 4, 6, 7]
    assert h == {'selected': 4, 'total': 8, 'cutoff_score': 3., 'boundary_image_id': 2, 'boundary_row_ordinal': 4}
    assert np.flatnonzero(low).tolist() == [0, 2, 3, 5]
    for coverage in (.25, .5, .75):
        mask, _ = rank_mask(np.zeros(7000), np.repeat(np.arange(1000), 7), coverage, 'high')
        assert int(mask.sum()) == int(7000*coverage)
    assert not rank_mask(values, ids, 0, 'high')[0].any()
    assert rank_mask(values, ids, 1, 'low')[0].all()


def predictions(offset):
    return [{'image_id': i, 'category_id': 1, 'bbox': [offset, 0, 10, 10], 'score': score}
            for i, score in [(1, .8), (1, .7), (3, .6)]]


def test_composition_exact_anchors_empty_images_native_order_and_no_mutation():
    raw, hybrid = predictions(0), predictions(1)
    before = copy.deepcopy((raw, hybrid))
    assert compose_predictions(raw, hybrid, set(), [1, 2, 3]) == raw
    assert compose_predictions(raw, hybrid, {1, 2, 3}, [1, 2, 3]) == hybrid
    chosen = compose_predictions(raw, hybrid, {1, 2}, [1, 2, 3])
    assert chosen == hybrid[:2]+raw[2:]
    chosen[0]['area'] = 100  # Official loadRes adds fields to its input dictionaries.
    assert (raw, hybrid) == before
    assert compose_predictions(raw, [], {1}, [1, 2, 3]) == raw[2:]


def test_same_image_decisions_across_detectors_and_fixed_blocks():
    selected, ids = {1}, [1, 2, 3]
    for offset in (0, 10, 20):
        raw, hybrid = predictions(offset), predictions(offset+1)
        composed = compose_predictions(raw, hybrid, selected, ids)
        assert [p['bbox'][0] for p in composed] == [offset+1, offset+1, offset]
        assert compose_predictions(raw, hybrid, selected, [3]) == raw[2:]


def test_latency_keeps_identity_signal_cost_for_skipped_images():
    r = {'source_setup_seconds': .05, 'adapt_seconds_3': .3, 'identity_step_seconds': .07}
    assert latency_estimate(r, 'no_adapt', False) == .05
    assert latency_estimate(r, 'support_count', False) == .05
    assert latency_estimate(r, 'gradient_cosine', False) == pytest.approx(.12)
    assert latency_estimate(r, 'pseudo_loss', True) == pytest.approx(.35)
    assert latency_estimate(r, 'full_hybrid', True) == pytest.approx(.35)


def test_preparation_keeps_one_global_mask_and_block_membership(tmp_path):
    from scripts.prepare_t010 import prepare
    study, output = tmp_path/'input', tmp_path/'output'
    study.mkdir()
    (study/'environment.json').write_text(json.dumps({'evaluated_image_ids': [1, 2, 3, 4],
        'cases': ['gamma_s1', 'clean_s0'], 'source_revision': 'frozen'}))
    (study/'subset.json').write_text(json.dumps({'replication_blocks': {'block_1': [1, 2], 'block_2': [3, 4]}}))
    rows = []
    for i in range(1, 5):
        for family, severity in [('gamma', 1), ('clean', 0)]:
            r = receipt()
            r['diagnostics'][0].update(phi=[0.]*8, step_seconds=.07)
            r.update(image_id=i, family=family, severity=severity, variant='det_pseudo_clip_radius',
                     phi_norm_3=.1, source_setup_seconds=.05, adapt_seconds_3=.3)
            rows.append(r)
    (study/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # The pre-AP stage must not need or inspect these files.
    (study/'metrics.json').write_text('invalid forbidden outcomes')
    prepare(study, output)
    matrix = np.load(output/'decisions.npz')['selected']
    assert matrix.shape == (44, 8) and not matrix[0].any() and matrix[1].all()
    assert matrix[2].sum() == 2 and matrix[2, :2].all()
    safety = json.loads((output/'safety.json').read_text())['clip_norm_high_25']
    assert safety['block_1']['all']['coverage'] == .5
    assert safety['block_2']['all']['coverage'] == 0  # No separate block recalibration.
    assert safety['aggregate']['clean_s0']['effective_phi3']['mean'] == pytest.approx(.025)
    meta = json.loads((output/'manifest.json').read_text())
    assert meta['configurations'] == 44 and meta['labels_or_prediction_files_read'] is False


def test_official_gated_AP_anchors_and_fixed_blocks():
    pytest.importorskip('pycocotools')
    from pycocotools.coco import COCO
    from taisp.analysis.replication import replication_ap
    coco = COCO()
    coco.dataset = {'info': {}, 'images': [{'id': i, 'width': 100, 'height': 100} for i in (1, 2)],
        'categories': [{'id': 1, 'name': 'object'}], 'annotations': [
            {'id': i, 'image_id': i, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0} for i in (1, 2)]}
    coco.createIndex()
    raw = [{'image_id': i, 'category_id': 1, 'bbox': [50, 50, 10, 10], 'score': .9} for i in (1, 2)]
    hybrid = [{**r, 'bbox': [0, 0, 10, 10]} for r in raw]
    blocks = {'block_1': [1], 'block_2': [2]}
    empty = replication_ap(coco, [1, 2], compose_predictions(raw, hybrid, set(), [1, 2]), blocks)
    full = replication_ap(coco, [1, 2], compose_predictions(raw, hybrid, {1, 2}, [1, 2]), blocks)
    mixed = replication_ap(coco, [1, 2], compose_predictions(raw, hybrid, {1}, [1, 2]), blocks)
    assert empty['aggregate']['AP'] == 0
    assert full['aggregate']['AP'] == pytest.approx(1)
    assert 0 < mixed['aggregate']['AP'] < 1
    assert mixed['block_1']['AP'] == pytest.approx(1) and mixed['block_2']['AP'] == 0
    assert all('area' not in p for p in raw+hybrid)
