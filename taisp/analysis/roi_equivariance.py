"""R039 analysis-only fixed-ROI objectives. No annotated data or oracle imports."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from taisp import DifferentiableISP
from taisp.isp.spatial import support_mask
from taisp.models.detector import load_detector
from taisp.models.detector_signal import flip_boxes, select_predictions
from .corruptions import corrupt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{list(tensor.shape)}\n'.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def isolated(model):
    return (all(not m.training for m in model.modules()) and
            all(not p.requires_grad and p.grad is None for p in model.parameters()))


def setup():
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    source, isp = load_detector('cuda:0'), DifferentiableISP().to('cuda:0')
    digest = state_hash(source)
    assert digest == '73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    weight = Path(torch.hub.get_dir())/'checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    assert sha(weight) == '258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    return source, isp, {'source_revision': os.environ.get('TAISP_SOURCE_REVISION'),
        'python': platform.python_version(), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'seed': 20260913, 'source_state_sha256': digest, 'weight_sha256': sha(weight),
        'deterministic_algorithms': False, 'cudnn_benchmark': False,
        'device': 'cuda:0', 'optimizer_steps': 0, 'AP_calls': 0}


def images(manifest):
    for i, info in enumerate(manifest['images']):
        assert sha(info['path']) == info['sha256']
        with Image.open(info['path']) as im:
            clean = pil_to_tensor(im.convert('RGB')).float().div(255).unsqueeze(0).cuda()
        case = manifest['corrupted_cases'][i]
        family, severity = case.rsplit('_s', 1)
        for name, image in [('clean_s0', clean), (case, corrupt(clean, family, int(severity)))]:
            yield info, name, image


def fixed_roi_representation(detector, image, boxes):
    """Same transform/backbone/pool/head as fixed_roi_logits, preserving ROI order."""
    model = detector.model
    height, width = image.shape[-2:]
    transformed, _ = model.transform(list(image), None)
    rh, rw = transformed.image_sizes[0]
    scaled = boxes.detach()*boxes.new_tensor([rw/width, rh/height, rw/width, rh/height])
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    pooled = model.roi_heads.box_roi_pool(features, [scaled], transformed.image_sizes)
    encoded = model.roi_heads.box_head(pooled)
    logits, _ = model.roi_heads.box_predictor(encoded)
    return encoded, logits


def paired_losses(z, z_flip, logits, logits_flip):
    feature = 1-(F.normalize(z, dim=-1)*F.normalize(z_flip, dim=-1)).sum(-1)
    lp, lq = F.log_softmax(logits, -1), F.log_softmax(logits_flip, -1)
    lm = torch.logaddexp(lp, lq)-math.log(2.)
    js = (lp.exp()*(lp-lm)+lq.exp()*(lq-lm)).sum(-1)/2
    return {'roi_feat_eq': feature, 'roi_logit_eq': js}


def candidate_cotangents(detector, image, boxes):
    if not len(boxes):
        return ({k: torch.zeros_like(image) for k in ('roi_feat_eq', 'roi_logit_eq')},
                {k: {'loss': 0., 'per_object': [], 'empty_support': True}
                 for k in ('roi_feat_eq', 'roi_logit_eq')}, [])
    y = image.detach().requires_grad_(True)
    mapped = flip_boxes(boxes, image.shape[-1])
    assert torch.allclose(flip_boxes(mapped, image.shape[-1]), boxes, atol=1e-4, rtol=0)
    z, logits = fixed_roi_representation(detector, y, boxes)
    zf, logitsf = fixed_roi_representation(detector, y.flip(-1), mapped)
    values = paired_losses(z, zf, logits, logitsf)
    cotangents, receipts = {}, {}
    for j, (name, per_object) in enumerate(values.items()):
        loss = per_object.mean()
        cotangents[name] = torch.autograd.grad(loss, y, retain_graph=j == 0)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(cotangents[name]).all()
        receipts[name] = {'loss': loss.item(), 'per_object': per_object.detach().cpu().tolist(),
                          'empty_support': False}
    pairs = [{'index': i, 'original_box': b, 'flipped_box': fb}
             for i, (b, fb) in enumerate(zip(boxes.cpu().tolist(), mapped.cpu().tolist()))]
    return cotangents, receipts, pairs


def common_jvp(image, isp, mask, cotangents):
    """Pure port of accepted A1 common_reference, float32 JVP/float64 reductions."""
    x, m = image.detach(), mask.detach().double()
    cs = {k: c.detach().double() for k, c in cotangents.items()}
    refs = {k: {'global': [], 'object': [], 'background': []} for k in cs}
    phi = x.new_zeros(8)
    primal_checks, columns = [], []
    for k in range(8):
        direction = x.new_zeros(8)
        direction[k] = 1
        primal, column = torch.func.jvp(lambda p: isp(x, p), (phi,), (direction,))
        assert torch.isfinite(column).all()
        primal_checks.append(bool(torch.allclose(primal, x, atol=2e-7, rtol=1e-6)))
        jd = column.detach().double()
        columns.append(jd.norm().item())
        for name, c in cs.items():
            refs[name]['global'].append((c*jd).sum().item())
            refs[name]['object'].append((c*m*jd).sum().item())
            refs[name]['background'].append((c*(1-m)*jd).sum().item())
    assert all(primal_checks)
    for r in refs.values():
        assert all(abs(g-o-b) <= 1e-12+1e-10*(abs(o)+abs(b))
                   for g, o, b in zip(r['global'], r['object'], r['background']))
    return refs, {'primal_identity_checks': primal_checks, 'column_l2_norms': columns,
                  'jacobian_dtype': str(image.dtype), 'reduction_dtype': 'torch.float64',
                  'device': str(image.device), 'columns': 8}


def mask_receipt(mask, rectangles):
    return {'shape': list(mask.shape), 'rectangles': rectangles,
            'area_fraction': mask.double().mean().item(),
            'uint8_sha256': hashlib.sha256(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()}


def run(manifest_path, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, isp, env = setup()
    env['image_manifest_sha256'] = sha(manifest_path)
    write(output/'environment.json', env)
    rows = []
    for index, (info, case, image) in enumerate(images(json.loads(manifest_path.read_text()))):
        with torch.no_grad():
            selected = select_predictions(source(image)[0], .5, 20)
        mask, rectangles = support_mask(image, selected['boxes'])
        # Evaluate at the actual frozen ISP identity output, not a changed image pipeline.
        identity = isp(image, image.new_zeros(8)).detach()
        cs, objectives, pairs = candidate_cotangents(source, identity, selected['boxes'])
        refs, numerical = common_jvp(image, isp, mask, cs)
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case,
               'block': info['block'], 'support_count': len(selected['boxes']),
               'supports': {k: v.cpu().tolist() for k, v in selected.items()},
               'mask': mask_receipt(mask, rectangles), 'pairs': pairs,
               'objectives': objectives, 'gradients': refs, 'jacobian': numerical,
               'isolation': isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())}
        assert row['isolation']
        write(output/f'record_{index:02d}.json', row)
        rows.append(row)
        print(json.dumps({'stage': 'candidate', 'episode': index, 'supports': len(selected['boxes'])}), flush=True)
    assert len(rows) == 32 and state_hash(source) == env['source_state_sha256']
    # Executed in its own interpreter; task/reference modules must never have been loaded.
    forbidden = [k for k in sys.modules if k.startswith('taisp.') and
                 any(s in k for s in ('oracle', 'source_meta', 'common_jacobian', 'roi_equivariance_reference'))]
    assert not forbidden, forbidden
    write(output/'records.json', rows)
    write(output/'completion.json', {'status': 'candidate_complete', 'episodes': len(rows),
        'source_hash_after': state_hash(source), 'frozen_eval_grad_none': isolated(source),
        'forbidden_modules_loaded': forbidden, 'seconds': time.perf_counter()-started})
    write(output/'sha256.json', {p.name: sha(p) for p in sorted(output.glob('*.json'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
