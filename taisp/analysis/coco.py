"""Annotated subset loading and official evaluation, analysis-only."""

import json
from pathlib import Path

import torch


class COCOSubset:
    def __init__(self, root):
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.manifest = json.loads((self.root / "subset.json").read_text())
        self.coco = COCO(str(self.root / "instances_val2017.json"))
        self.ids = self.manifest["image_ids"]

    def load(self, image_id, device):
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        info = self.coco.imgs[image_id]
        # PIL decoding occurs before the differentiable ISP/CLIP path.
        with Image.open(self.root / "val2017" / info["file_name"]) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div(255).unsqueeze(0).to(device)
        boxes, labels = [], []
        for ann in self.coco.imgToAnns[image_id]:
            x, y, w, h = ann["bbox"]
            if ann.get("iscrowd", 0) or w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        targets = [{"boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
                    "labels": torch.tensor(labels, dtype=torch.int64, device=device)}]
        return tensor, targets


def prediction_records(image_id, prediction):
    boxes = prediction["boxes"].detach().cpu().clone()
    boxes[:, 2:] -= boxes[:, :2]
    return [{"image_id": image_id, "category_id": label, "bbox": box, "score": score}
            for box, label, score in zip(boxes.tolist(), prediction["labels"].tolist(),
                                         prediction["scores"].tolist())]


def subset_ap(coco, image_ids, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if predictions:
        results = coco.loadRes(predictions)
    else:
        # Empty predictions are a meaningful zero-AP method, not a failed run.
        results = COCO()
        results.dataset = {"images": [coco.imgs[i] for i in image_ids],
                           "categories": coco.dataset["categories"], "annotations": []}
        results.createIndex()
    evaluator = COCOeval(coco, results, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    return {name: float(value) for name, value in zip(names, evaluator.stats)}
