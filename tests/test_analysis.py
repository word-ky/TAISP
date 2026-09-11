import torch

from taisp.analysis.corruptions import CASES, corrupt


def test_corruption_equations_and_severity():
    x = torch.full((1, 3, 2, 3), 0.25)
    assert len(CASES) == 6
    torch.testing.assert_close(corrupt(x, "gamma", 2), x.square())
    torch.testing.assert_close(corrupt(x, "contrast", 2), (x - 0.5) * 0.3 + 0.5)
    cast = corrupt(x, "color_cast", 2)
    torch.testing.assert_close(cast[0, :, 0, 0], torch.tensor([0.35, 0.25, 0.15]))
    assert corrupt(x, "gamma", 2).mean() < corrupt(x, "gamma", 1).mean()
    for family, severity in CASES:
        a, b = corrupt(x, family, severity), corrupt(x, family, severity)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_official_subset_ap_known_perfect_and_empty():
    import pytest
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from taisp.analysis.coco import subset_ap

    coco = COCO()
    coco.dataset = {"info": {}, "images": [{"id": 1, "width": 100, "height": 100}],
                    "categories": [{"id": 1, "name": "object"}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                        "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}]}
    coco.createIndex()
    prediction = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "score": 0.9}]
    assert abs(subset_ap(coco, [1], prediction)["AP"] - 1) < 1e-10
    assert subset_ap(coco, [1], [])["AP"] == 0
