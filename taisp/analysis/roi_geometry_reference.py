"""T025-A post-candidate-lock source reference; unchanged accepted oracle."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from taisp.losses.detector_native import DetectorNativeLoss
from .oracle import detector_task_loss
from .roi_equivariance import (sha, write, setup, images, support_mask, mask_receipt,
                               common_jvp, isolated, state_hash)
from .common_jacobian import reference_checks
from .roi_equivariance_reference import metrics, norm, score, verify_lock


NAME = 'roi_bbox_exposure_stability'


def aggregate(rows):
    def group(chosen):
        vals = [r['metrics'][NAME] for r in chosen]
        near_images = {r['image_id'] for r in chosen if r['metrics'][NAME]['material_near_zero']}
        return {'n': len(vals), 'S_geom_positive': sum(v['S_geom'] > 0 for v in vals),
            'Delta_global_positive': sum(v['Delta_global'] > 0 for v in vals),
            'median_Delta_global': statistics.median(v['Delta_global'] for v in vals),
            'mean_Delta_global': statistics.mean(v['Delta_global'] for v in vals),
            'median_Delta_obj': statistics.median(v['Delta_obj'] for v in vals),
            'zero_gradients': sum(v['zero_gradient'] for v in vals),
            'near_zero_gradients': sum(v['near_zero_gradient'] for v in vals),
            'nonempty_near_zero': sum(v['material_near_zero'] for v in vals),
            'nonempty_near_zero_distinct_images': len(near_images),
            'roi_box_positive': sum(v['roi_box_diagnostic']['score'] > 0 for v in vals),
            'localization_positive': sum(v['localization_diagnostic']['score'] > 0 for v in vals)}
    overall = group(rows)
    bad = group([r for r in rows if r['case'] != 'clean_s0'])
    blocks = {str(i): group([r for r in rows if r['block'] == i]) for i in range(4)}
    conditions = {c: group([r for r in rows if r['case'] == c]) for c in sorted({r['case'] for r in rows})}
    gate = {'S_overall': overall['S_geom_positive'] >= 30, 'S_corrupt': bad['S_geom_positive'] >= 15,
        'Delta_overall': overall['Delta_global_positive'] >= 30, 'Delta_corrupt': bad['Delta_global_positive'] >= 15,
        'median_overall': overall['median_Delta_global'] > 0, 'median_corrupt': bad['median_Delta_global'] > 0,
        'blocks': sum(b['median_Delta_global'] > 0 for b in blocks.values()) >= 3,
        'no_systematic_near_zero': overall['nonempty_near_zero_distinct_images'] < 2,
        'integrity': all(r['integrity_passed'] for r in rows)}
    passed = all(gate.values())
    return {'candidate': NAME, 'overall': overall, 'corrupt': bad, 'blocks': blocks, 'conditions': conditions,
        'gate': gate, 'passed': passed, 'status': 'NEEDS_REVIEW', 'nomination': NAME if passed else None,
        'disposition': 'nominate_for_research_review_only' if passed else 'close_exposure_pair_roi_box_geometry',
        'runtime_or_AP_authorized': False}


def run(cohort_path, candidate_root, lock_path, output):
    lock = json.loads(lock_path.read_text())
    candidates = verify_lock(candidate_root, lock)
    assert len(candidates) == 48 and len(lock['files']) == 52
    assert sha(cohort_path) == lock['cohort_sha256']
    manifest = json.loads(cohort_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, isp, env = setup()
    env.update({'candidate_lock_sha256': sha(lock_path), 'candidate_commit': lock['candidate_commit'],
                'ordering': 'all48 candidate receipts verified before annotation load',
                'cohort_sha256': sha(cohort_path)})
    write(output/'environment.json', env)
    assert sha(manifest['annotation_file']) == manifest['annotation_sha256']
    data = json.loads(Path(manifest['annotation_file']).read_text())
    annotations = {i: [] for i in manifest['image_ids']}
    for ann in data['annotations']:
        if (ann['image_id'] in annotations and not ann.get('iscrowd', 0) and
            ann['bbox'][2] > 0 and ann['bbox'][3] > 0 and ann.get('area', 1) > 0):
            annotations[ann['image_id']].append(ann)
    del data
    rows = []
    for index, (info, case, image) in enumerate(images(manifest)):
        candidate = candidates[index]
        assert (info['image_id'], case, info['block']) == (candidate['image_id'], candidate['case'], candidate['block'])
        support = {k: torch.tensor(v, device='cuda:0', dtype=torch.long if k == 'labels' else torch.float32)
                   for k,v in candidate['supports'].items()}
        support['boxes'] = support['boxes'].reshape(-1,4)
        assert candidate['roi_order'] == list(range(candidate['support_count']))
        assert candidate['objectives'][NAME]['class_indices'] == candidate['supports']['labels']
        mask, rectangles = support_mask(image, support['boxes'])
        assert mask_receipt(mask, rectangles) == candidate['mask']
        anns = annotations[info['image_id']]
        boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in anns]
        targets = [{'boxes': image.new_tensor(boxes).reshape(-1,4),
                    'labels': torch.tensor([a['category_id'] for a in anns], device=image.device)}]
        native = DetectorNativeLoss(source, {'base': support}, 'det_pseudo')
        y = isp(image, image.new_zeros(8)).detach().requires_grad_(True)
        task, components = detector_task_loss(source, y, targets, seed=20260913)
        ct = torch.autograd.grad(task, y, retain_graph=True)[0].detach()
        cb = torch.autograd.grad(components['loss_box_reg'], y, retain_graph=True)[0].detach()
        cl = torch.autograd.grad(components['loss_box_reg']+components['loss_rpn_box_reg'], y)[0].detach()
        pseudo = native(image, y)
        cp = torch.autograd.grad(pseudo, y)[0].detach()
        cs = {'task': ct, 'pseudo': cp, 'roi_box': cb, 'localization': cl}
        assert all(torch.isfinite(c).all() for c in cs.values())
        refs, jacobian = common_jvp(image, isp, mask, cs)
        checks = {}
        for name,c in [('task',ct),('pseudo',cp)]:
            phi = image.new_zeros(8, requires_grad=True)
            direct = torch.autograd.grad((isp(image,phi)*c).sum(),phi)[0].cpu().tolist()
            r = refs[name]
            checks[name] = reference_checks(r['global'],r['object'],r['background'],direct)
        m = metrics(candidate,refs)[NAME]
        m['S_geom'] = m.pop('S_c')
        g = candidate['gradients'][NAME]['object']
        for name in ('roi_box','localization'):
            t = refs[name]['object']
            m[name+'_diagnostic'] = {'score': score(t,g), 'task_gradient_norm': norm(t),
                'cosine': math.fsum(a*b for a,b in zip(t,g))/(norm(t)*norm(g)) if norm(t) and norm(g) else None}
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case, 'block': info['block'],
            'mask': candidate['mask'], 'support_count': candidate['support_count'],
            'task_loss': task.item(), 'task_components': {k:v.item() for k,v in components.items()},
            'pseudo_loss': pseudo.item(), 'reference_gradients': refs, 'jacobian': jacobian,
            'reference_checks': checks, 'metrics': {NAME:m},
            'integrity_passed': candidate['isolation'] and isolated(source) and isp.phi.grad is None and
                not bool(isp.phi.any()) and all(c[k]['passed'] for c in checks.values() for k in ('partition','parity'))}
        write(output/f'record_{index:02d}.json',row)
        assert row['integrity_passed'], 'Observed reference numerical/isolation blocker; stop.'
        rows.append(row)
        print(json.dumps({'stage':'reference','episode':index,'integrity':True}),flush=True)
    assert state_hash(source) == env['source_state_sha256'] and isolated(source)
    write(output/'records.json',rows)
    write(output/'summary.json',aggregate(rows))
    write(output/'completion.json',{'status':'NEEDS_REVIEW','episodes':len(rows),
        'source_hash_after':state_hash(source),'seconds':time.perf_counter()-started,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.glob('*.json'))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--candidate-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.cohort,a.candidate_root,a.candidate_lock,a.output)
