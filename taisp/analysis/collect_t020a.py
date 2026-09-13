"""Separate source-labelled gradient collection and cross-fit process (no AP/TTT)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

from taisp import DifferentiableISP
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions
from .deterministic_replay import state_hash
from .differential_subspace import write_json
from .gradient_transport import fit_folds, heldout_alignment
from .oracle import detector_task_loss
from .run_t018a import load_image, CASES_ALL, corrupt
from .source_meta_smoke import frozen_unchanged


def source_targets(annotations, device):
    boxes, labels = [], []
    for a in annotations:
        x,y,w,h = a['bbox']
        if a.get('iscrowd',0) or w <= 0 or h <= 0:
            continue
        boxes.append([x,y,x+w,y+h]); labels.append(a['category_id'])
    return [{'boxes': torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1,4),
             'labels': torch.tensor(labels, dtype=torch.long, device=device)}]


def collect(config, manifest_path, output, smoke=False):
    manifest = json.loads(manifest_path.read_text())
    images = [dict(r, block=i) for i,r in enumerate(manifest['images'][:2])] if smoke else manifest['images']
    assert len(manifest['images']) == config['count'] and manifest['overlap_prior_source'] == manifest['overlap_val'] == 0
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed']); torch.set_num_threads(config['threads']); torch.backends.cudnn.benchmark = False
    source = load_detector(config['device']); isp = DifferentiableISP().to(config['device'])
    before = state_hash(source)
    assert before == '73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    states = [{k:v.clone() for k,v in source.state_dict().items()}]
    annotation_bytes = Path(manifest['annotation_file']).read_bytes()
    assert hashlib.sha256(annotation_bytes).hexdigest() == manifest['annotation_sha256']
    by_image = {r['image_id']: [] for r in images}
    for a in json.loads(annotation_bytes)['annotations']:
        if a['image_id'] in by_image:
            by_image[a['image_id']].append(a)
    del annotation_bytes
    write_json(output/'environment.json', {'source_state_sha256': before, 'config': config, 'smoke': smoke,
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cohort_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'annotation_sha256': manifest['annotation_sha256'], 'CLIP': 'not loaded or called in source pair collection',
        'boundary': 'source-labelled analysis process; runtime receives only fold matrices/provenance, never task gradients/labels'})
    rows = []; started = time.perf_counter()
    with (output/'gradient_pairs.jsonl').open('w') as handle:
        for i,info in enumerate(images):
            clean = load_image(info, config['device']); targets = source_targets(by_image[info['image_id']], config['device'])
            for family,severity in CASES_ALL:
                image = clean if family == 'clean' else corrupt(clean,family,severity)
                with torch.no_grad():
                    selected = select_predictions(source(image)[0], config['support_threshold'], config['support_topk'])
                support = {k:v.clone() for k,v in selected.items()}
                loss = DetectorNativeLoss(source, {'base':selected}, 'det_pseudo').eval().requires_grad_(False)
                phi = isp.phi.detach().clone().requires_grad_(True)
                pseudo = loss(image, isp(image,phi))
                gp = torch.autograd.grad(pseudo,phi)[0].detach()
                phi = isp.phi.detach().clone().requires_grad_(True)
                task, components = detector_task_loss(source, isp(image,phi), targets, seed=config['seed'])
                gt = torch.autograd.grad(task,phi)[0].detach()
                isolation = {'source_unchanged_frozen_eval_grad_none': frozen_unchanged((source,),states),
                    'isp_identity_grad_none': bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None),
                    'support_unchanged_detached': all(torch.equal(v,support[k]) and not v.requires_grad for k,v in selected.items()),
                    'finite_gradients': bool(torch.isfinite(gp).all() and torch.isfinite(gt).all())}
                assert all(isolation.values()), isolation
                row = dict(image_id=info['image_id'],block=info['block'],case=f'{family}_s{severity}',
                    pseudo_gradient=gp.tolist(),task_gradient=gt.tolist(),pseudo_norm=gp.norm().item(),task_norm=gt.norm().item(),
                    pseudo_loss=pseudo.item(),task_loss=task.item(),task_components={k:v.item() for k,v in components.items()},
                    support={k:v.tolist() for k,v in selected.items()},support_count=len(selected['boxes']),isolation=isolation)
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();rows.append(row)
            print(f'paired gradients {i+1}/{len(images)} image_id={info["image_id"]} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    collection_seconds = time.perf_counter()-started
    start_fit = time.perf_counter()
    fits = fit_folds(rows, eps=config['radius_eps'], device=config['device'])
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter()-start_fit
    write_json(output/'fits.json', fits)
    write_json(output/'alignment.json', heldout_alignment(rows,fits))
    isolation = {'source_hash_unchanged': state_hash(source)==before,
                 'source_frozen_eval_grad_none': frozen_unchanged((source,),states),
                 'all_episode_checks_passed': all(all(r['isolation'].values()) for r in rows),
                 'image_fold_isolation': all(not set(f['train_image_ids']) & set(f['heldout_image_ids']) for f in fits.values())}
    assert all(isolation.values())
    write_json(output/'completion.json',dict(status='completed',smoke=smoke,images=len(images),pairs=len(rows),
        folds=len(fits),collection_seconds=collection_seconds,fit_seconds=fit_seconds,isolation=isolation,
        zero_pseudo=sum(r['pseudo_norm']==0 for r in rows),zero_task=sum(r['task_norm']==0 for r in rows)))
    print('T020-A source collection/fit complete; runtime starts in separate process',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/t020a.yaml'))
    p.add_argument('--manifest',type=Path,default=Path('research_log/T020A_train_cohort.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();collect(yaml.safe_load(a.config.read_text()),a.manifest,a.output,a.smoke)
