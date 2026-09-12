import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(os.environ.get('TAISP_REAL_MODELS') != '1', reason='explicit real-model integration')
def test_real_ssd_mapping_serialization_freeze_repeat_and_hybrid_isolation(tmp_path):
    from taisp import DifferentiableISP
    from taisp.analysis.coco import COCOSubset, prediction_records, subset_ap
    from taisp.analysis.ssd_target import load_ssd_target, ssd_metadata
    from taisp.models.detector import load_detector
    from taisp.models.detector_signal import select_predictions
    from taisp.losses.detector_native import DetectorNativeLoss
    from taisp.losses.clip_semantic import load_clip_guidance
    from taisp.tta.trust_radius import adapt_clip_radius

    torch.manual_seed(20260912)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    data = COCOSubset(os.environ.get('TAISP_TEST_DATA_ROOT', '/home/liujianhua/wjq/TAISP/shared/coco200'))
    image_id = data.ids[0]
    image, _ = data.load(image_id, 'cpu')
    source, clip, isp = load_detector('cpu'), load_clip_guidance('cpu', local_files_only=True), DifferentiableISP()

    def episode():
        with torch.no_grad():
            support = select_predictions(source(image)[0], .5, 20)
        loss = DetectorNativeLoss(source, {'base': support}, 'det_pseudo')
        return support, adapt_clip_radius(image, isp, loss, clip, steps=3, lr=.1, eps=1e-12)

    absent_support, absent = episode()
    assert len(absent_support['boxes']) > 0 and absent.phi.norm() > 0
    target = load_ssd_target('cuda:0')
    state = {k: v.cpu().clone() for k, v in target.state_dict().items()}
    def forbidden(*args):
        raise AssertionError('SSD invoked during adaptation')
    hook = target.model.register_forward_pre_hook(forbidden)
    present_support, present = episode()
    hook.remove()
    for key in absent_support:
        torch.testing.assert_close(absent_support[key], present_support[key], atol=0, rtol=0)
    torch.testing.assert_close(absent.phi, present.phi, atol=0, rtol=0)
    torch.testing.assert_close(absent.enhanced, present.enhanced, atol=0, rtol=0)
    x = present.enhanced.cuda()
    with torch.no_grad():
        first, second = target(x)[0], target(x)[0]
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert all(not m.training for m in target.modules())
    assert all(not p.requires_grad and p.grad is None for p in target.parameters())
    assert all(torch.equal(v.cpu(), state[k]) for k, v in target.state_dict().items())
    meta = ssd_metadata(target)
    for category_id, category in data.coco.cats.items():
        assert meta['categories'][category_id] == category['name']
    assert set(first['labels'].tolist()) <= set(data.coco.cats)
    records = prediction_records(image_id, first)
    assert records and all(r['bbox'][2] >= 0 and r['bbox'][3] >= 0 for r in records)
    path = tmp_path/'predictions.json'
    path.write_text(json.dumps(records))
    metrics = subset_ap(data.coco, [image_id], json.loads(path.read_text()))
    assert all(0 <= metrics[name] <= 1 for name in ('AP', 'AP50', 'AP75'))
    pin = Path('research_log/T009_ssd_pin.json')
    if pin.exists():
        assert meta == json.loads(pin.read_text())
    if os.environ.get('TAISP_SSD_PIN_OUT'):
        Path(os.environ['TAISP_SSD_PIN_OUT']).write_text(json.dumps(meta, indent=2)+'\n')
