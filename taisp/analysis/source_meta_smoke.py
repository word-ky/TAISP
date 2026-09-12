"""T013-B fixed train2017 outer smoke; annotations stay in analysis only."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions
from taisp.models.parameter_predictor import ParameterPredictor
from taisp.training.initialization import adapt_from_initialization
from .corruptions import corrupt
from .oracle import detector_task_loss


def source_outer_episode(image, targets, predictor, isp, native, clip, source):
    initial = predictor(image).squeeze(0)
    result = adapt_from_initialization(image, isp, native, clip, initial)
    loss, components = detector_task_loss(source, result.enhanced, targets, seed=20260913)
    return result, loss, components


def load_episodes(manifest, device):
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    episodes = []
    for info, corrupted_case in zip(manifest['images'], manifest['corrupted_cases']):
        path = Path(info['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        assert '/train2017/' in info['coco_url']
        with Image.open(path) as image:
            clean = pil_to_tensor(image.convert('RGB')).float().div(255).unsqueeze(0).to(device)
        boxes, labels = [], []
        for ann in info['annotations']:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
        targets = [{'boxes': torch.tensor(boxes, dtype=torch.float32, device=device),
                    'labels': torch.tensor(labels, dtype=torch.long, device=device)}]
        family, severity = corrupted_case.rsplit('_s', 1)
        for case, image in [('clean_s0', clean), (corrupted_case, corrupt(clean, family, int(severity)))]:
            episodes.append({'image_id': info['image_id'], 'case': case, 'image': image, 'targets': targets})
    return episodes


def gradient_norm(parameters):
    return torch.sqrt(sum(p.grad.square().sum() for p in parameters)).item()


def parameter_vector(predictor):
    return torch.cat([p.detach().flatten() for p in predictor.parameters()])


def frozen_unchanged(models, states):
    return all(all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
               and all(not m.training for m in model.modules())
               and all(not p.requires_grad and p.grad is None for p in model.parameters())
               for model, state in zip(models, states))


def group_statistics(rows, clean):
    chosen = [row for row in rows if (row['case'] == 'clean_s0') == clean]
    return {**{f'{key}_{stat}': (sum(values)/len(values) if stat == 'mean' else max(values))
               for key in ('phi0_norm', 'phi3_norm', 'saturation_rate')
               for values in [[r[key] for r in chosen]] for stat in ('mean', 'max')},
            'empty_support_count': sum(row['support_count'] == 0 for row in chosen)}


def run(manifest_path, audit_path, output):
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['summary']['part_b_allowed']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['split'] == 'train2017' and len(manifest['images']) == 4
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    started = time.perf_counter()
    source, clip = load_detector(device), load_clip_guidance(device, local_files_only=True)
    isp, predictor = DifferentiableISP().to(device), ParameterPredictor().to(device)
    models = (source, clip)
    states = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    episodes = load_episodes(manifest, device)
    support_receipts = []
    for episode in episodes:
        with torch.no_grad():
            selected = select_predictions(source(episode['image'])[0], threshold=.5, topk=20)
        episode['native'] = DetectorNativeLoss(source, {'base': selected}, 'det_pseudo')
        support_receipts.append({'image_id': episode['image_id'], 'case': episode['case'],
                                 **{k: v.cpu().tolist() for k, v in selected.items()}})
    (output/'supports.json').write_text(json.dumps(support_receipts, indent=2)+'\n', encoding='utf-8')
    optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-3)
    original_parameters = parameter_vector(predictor).clone()
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        episode_rows = []
        step_started = time.perf_counter()
        for episode in episodes:
            result, loss, components = source_outer_episode(
                episode['image'], episode['targets'], predictor, isp, episode['native'], clip, source)
            (loss/8).backward()
            episode_rows.append({
                'image_id': episode['image_id'], 'case': episode['case'], 'outer_loss': loss.item(),
                'components': {k: v.item() for k, v in components.items()},
                'phi0': result.phi0.detach().cpu().tolist(), 'phi3': result.phi.detach().cpu().tolist(),
                'phi0_norm': result.phi0.norm().item(), 'phi3_norm': result.phi.norm().item(),
                'saturation_rate': result.diagnostics[-1]['saturation_rate'],
                'input_saturation_rate': ((episode['image'] <= 1e-4) | (episode['image'] >= 1-1e-4)).float().mean().item(),
                'support_count': len(episode['native'].boxes),
            })
        head_norm, trunk_norm = gradient_norm(predictor.head.parameters()), gradient_norm(predictor.features.parameters())
        finite = all(torch.isfinite(p.grad).all().item() for p in predictor.parameters())
        unchanged = frozen_unchanged(models, states)
        assert finite and head_norm > 0 and unchanged
        assert torch.count_nonzero(isp.phi) == 0 and isp.phi.grad is None
        assert max(row['saturation_rate'] for row in episode_rows) < 1
        torch.cuda.synchronize()
        row = {'step': step, 'optimizer_step_applied': step < 3,
               'mean_outer_loss': sum(r['outer_loss'] for r in episode_rows)/8,
               'mean_components': {k: sum(r['components'][k] for r in episode_rows)/8
                                   for k in episode_rows[0]['components']},
               'head_gradient_norm': head_norm, 'trunk_gradient_norm': trunk_norm,
               'predictor_parameter_change_norm': (parameter_vector(predictor)-original_parameters).norm().item(),
               'all_predictor_gradients_finite': finite, 'frozen_models_unchanged': unchanged,
               'clean': group_statistics(episode_rows, True), 'corrupted': group_statistics(episode_rows, False),
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'elapsed_seconds': time.perf_counter()-step_started, 'episodes': episode_rows}
        steps.append(row)
        with (output/'steps.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'episodes'}), flush=True)
        if step < 3:
            optimizer.step()
    assert (parameter_vector(predictor)-original_parameters).norm() > 0
    torch.save(predictor.state_dict(), output/'predictor_after_three_steps.pt')
    receipt = {'source_revision': os.environ.get('TAISP_SOURCE_REVISION'),
               'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
               'seed': 20260913, 'sampling_seed': 20260913, 'python': platform.python_version(),
               'torch': torch.__version__, 'device': device, 'gpu': torch.cuda.get_device_name(0),
               'optimizer': 'SGD', 'outer_lr': 1e-3, 'optimizer_steps': 3,
               'inner_steps': 3, 'inner_lr': .1, 'norm_epsilon': 1e-12,
               'image_ids': manifest['image_ids'], 'episode_count': 8, 'steps': steps,
               'elapsed_seconds': time.perf_counter()-started,
               'interpretation': 'Source-only overfit/gradient smoke; no generalization or detection-performance claim.'}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.audit, args.output)
