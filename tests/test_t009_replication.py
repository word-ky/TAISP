import pytest

from taisp.analysis.replication import ap_contrasts, replication_ap


def test_official_replication_blocks_filter_other_images():
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = {'info': {}, 'images': [{'id': i, 'width': 100, 'height': 100} for i in [1, 2]],
                    'categories': [{'id': 1, 'name': 'object'}], 'annotations': [
                        {'id': i, 'image_id': i, 'category_id': 1, 'bbox': [10, 10, 40, 40], 'area': 1600, 'iscrowd': 0} for i in [1, 2]]}
    coco.createIndex()
    predictions = [{'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 40, 40], 'score': .9}]
    r = replication_ap(coco, [1, 2], predictions, {'block_1': [1], 'block_2': [2]})
    assert r['block_1']['AP'] == pytest.approx(1)
    assert r['block_2']['AP'] == 0
    assert 0 < r['aggregate']['AP'] < 1


def test_macro_deltas_use_all_corruptions_and_keep_negative_clean():
    methods = ['no_adapt', 'global_generic', 'det_pseudo', 'det_pseudo_clip_radius']
    cases = ['gamma_s1', 'gamma_s2', 'clean_s0']
    metrics = {}
    for case in cases:
        for variant in methods:
            value = .3
            if variant == 'det_pseudo_clip_radius':
                value += {'gamma_s1': .02, 'gamma_s2': -.01, 'clean_s0': -.03}[case]
            metrics[f'ssd_{case}_{variant}'] = {m: value for m in ('AP', 'AP50', 'AP75')}
    r = ap_contrasts(metrics, cases, ['ssd'], methods)['ssd']['det_pseudo_clip_radius']['no_adapt']
    assert r['macro_corruption_AP_delta'] == pytest.approx(.5)
    assert r['positive_corruption_count'] == 1
    assert r['clean_AP_delta'] == pytest.approx(-3)
