"""Observed smoke mismatch: repeat frozen detector gradients with/without warmup."""
import json
from pathlib import Path
import os

import torch
from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.conditioned_clip import load_conditioner
from taisp.models.detector import load_detector
from taisp.analysis.coco import COCOSubset
from taisp.analysis.corruptions import corrupt
from taisp.analysis.oracle import detector_task_loss

torch.manual_seed(20260912)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
root = Path('/home/liujianhua/wjq/TAISP')
data = COCOSubset(root / 'shared/coco200')
base = json.loads((root / 'runs/20260912-013248-taisp-t002-coco200-final/artifacts/study/samples.jsonl').read_text().splitlines()[0])
generic = load_clip_guidance('cuda:0', local_files_only=True)
conditioner = load_conditioner(generic, local_files_only=True)
detector = load_detector('cuda:0')
isp = DifferentiableISP().cuda()
clean, targets = data.load(base['image_id'], 'cuda:0')
x = corrupt(clean, base['family'], base['severity'])
saved = x.new_tensor(base['g_det'])
rows = []
previous = None
for stage in ('cold1', 'cold2', 'after_clean1', 'after_clean2'):
    if stage == 'after_clean1':
        with torch.no_grad():
            detector(clean)
    phi = torch.zeros(8, device=x.device, requires_grad=True)
    loss, _ = detector_task_loss(detector, isp(x, phi), targets, seed=20260912+base['image_id'])
    grad = torch.autograd.grad(loss, phi)[0]
    rows.append(dict(stage=stage, loss=loss.item(), saved_loss=base['det_loss_before'],
        gradient=grad.tolist(), saved_gradient=saved.tolist(),
        max_error_saved=(grad-saved).abs().max().item(),
        max_error_previous=(grad-previous).abs().max().item() if previous is not None else None))
    previous = grad
out = Path(os.environ['AUTODL_ARTIFACTS_DIR']) / 'gradient_repeat.json'
out.write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
