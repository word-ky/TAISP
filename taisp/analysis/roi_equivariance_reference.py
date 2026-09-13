"""R039 separate post-lock source reference process; no candidate regeneration."""
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


EPS = 1e-12


def norm(x):
    return math.sqrt(math.fsum(v*v for v in x))


def score(t, g):
    return math.fsum(a*b for a, b in zip(t, g))/(norm(g)+EPS)


def metrics(candidate, refs):
    to, ts = refs['task']['object'], refs['task']['global']
    po, ps = refs['pseudo']['object'], refs['pseudo']['global']
    result = {}
    for name, r in candidate['gradients'].items():
        g = r['object']
        sc, sg, so = score(to, g), score(ts, ps), score(to, po)
        lengths = {k: norm(v) for k, v in {'candidate': g, 'task_object': to,
            'task_global': ts, 'pseudo_object': po, 'pseudo_global': ps}.items()}
        result[name] = {'S_c': sc, 'S_global': sg, 'Delta_global': sc-sg,
            'S_obj_current': so, 'Delta_obj': sc-so, 'norms': lengths,
            'cosine_candidate_task': (math.fsum(a*b for a,b in zip(to,g))/(norm(to)*norm(g))
                                      if norm(to) and norm(g) else None),
            'zero_gradient': norm(g) == 0, 'near_zero_gradient': norm(g) <= EPS,
            'material_near_zero': candidate['support_count'] > 0 and norm(g) <= EPS}
    return result


def aggregate(rows):
    results = {}
    for name in ('roi_feat_eq', 'roi_logit_eq'):
        def group(chosen):
            vals = [r['metrics'][name] for r in chosen]
            return {'n': len(vals), 'S_c_positive': sum(v['S_c'] > 0 for v in vals),
                'Delta_global_positive': sum(v['Delta_global'] > 0 for v in vals),
                'median_Delta_global': statistics.median(v['Delta_global'] for v in vals),
                'mean_Delta_global': statistics.mean(v['Delta_global'] for v in vals),
                'median_Delta_obj': statistics.median(v['Delta_obj'] for v in vals),
                'zero_gradients': sum(v['zero_gradient'] for v in vals),
                'near_zero_gradients': sum(v['near_zero_gradient'] for v in vals),
                'material_near_zero': sum(v['material_near_zero'] for v in vals)}
        all_, bad = group(rows), group([r for r in rows if r['case'] != 'clean_s0'])
        blocks = {str(i): group([r for r in rows if r['block'] == i]) for i in range(4)}
        conditions = {c: group([r for r in rows if r['case'] == c]) for c in sorted({r['case'] for r in rows})}
        gate = {'S_c_overall': all_['S_c_positive'] >= 20, 'S_c_corrupt': bad['S_c_positive'] >= 10,
            'Delta_overall': all_['Delta_global_positive'] >= 20,
            'Delta_corrupt': bad['Delta_global_positive'] >= 10,
            'median_overall': all_['median_Delta_global'] > 0,
            'median_corrupt': bad['median_Delta_global'] > 0,
            'blocks': sum(b['median_Delta_global'] > 0 for b in blocks.values()) >= 3,
            'no_material_near_zero': all_['material_near_zero'] == 0,
            'integrity': all(r['integrity_passed'] for r in rows)}
        results[name] = {'overall': all_, 'corrupt': bad, 'blocks': blocks, 'conditions': conditions,
                         'gate': gate, 'passed': all(gate.values())}
    passing = [k for k in results if results[k]['passed']]
    def key(k):
        a, c = results[k]['overall'], results[k]['corrupt']
        return c['Delta_global_positive'], c['median_Delta_global'], a['Delta_global_positive'], a['median_Delta_global']
    ranked = sorted(passing, key=key, reverse=True)
    tied = len(ranked) == 2 and key(ranked[0]) == key(ranked[1])
    return {'candidates': results, 'nomination': ranked[0] if ranked and not tied else None,
            'disposition': 'exact_tie_needs_review' if tied else
                ('nominate_for_research_review_only' if ranked else 'close_fixed_roi_flip_equivariance_family'),
            'status': 'NEEDS_REVIEW', 'runtime_or_AP_authorized': False}


def verify_lock(root, lock):
    assert lock['candidate_commit'] and len(lock['files']) >= 35
    for name, digest in lock['files'].items():
        assert sha(root/name) == digest, name
    return json.loads((root/'records.json').read_text())


def run(cohort_path, candidate_root, lock_path, output):
    # No annotation read/model load is reached until every committed receipt hash passes.
    lock = json.loads(lock_path.read_text())
    candidates = verify_lock(candidate_root, lock)
    manifest = json.loads(cohort_path.read_text())
    assert len(candidates) == 32
    assert sha(cohort_path) == lock['cohort_sha256']
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, isp, env = setup()
    env.update({'candidate_lock_sha256': sha(lock_path), 'candidate_commit': lock['candidate_commit'],
                'ordering': 'all candidate hashes verified before annotation load', 'cohort_sha256': sha(cohort_path)})
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
                   for k, v in candidate['supports'].items()}
        support['boxes'] = support['boxes'].reshape(-1, 4)
        mask, rectangles = support_mask(image, support['boxes'])
        assert mask_receipt(mask, rectangles) == candidate['mask']
        anns = annotations[info['image_id']]
        boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in anns]
        targets = [{'boxes': image.new_tensor(boxes).reshape(-1,4),
                    'labels': torch.tensor([a['category_id'] for a in anns], device=image.device)}]
        native = DetectorNativeLoss(source, {'base': support}, 'det_pseudo')
        y = isp(image, image.new_zeros(8)).detach().requires_grad_(True)
        task, components = detector_task_loss(source, y, targets, seed=20260913)
        ct = torch.autograd.grad(task, y)[0].detach()
        pseudo = native(image, y)
        cp = torch.autograd.grad(pseudo, y)[0].detach()
        assert torch.isfinite(ct).all() and torch.isfinite(cp).all()
        refs, jacobian = common_jvp(image, isp, mask, {'task': ct, 'pseudo': cp})
        checks = {}
        for name, c in [('task', ct), ('pseudo', cp)]:
            phi = image.new_zeros(8, requires_grad=True)
            direct = torch.autograd.grad((isp(image, phi)*c).sum(), phi)[0].cpu().tolist()
            r = refs[name]
            checks[name] = reference_checks(r['global'], r['object'], r['background'], direct)
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case, 'block': info['block'],
            'mask': candidate['mask'], 'support_count': candidate['support_count'],
            'task_loss': task.item(), 'task_components': {k: v.item() for k,v in components.items()},
            'pseudo_loss': pseudo.item(), 'reference_gradients': refs, 'jacobian': jacobian,
            'reference_checks': checks, 'metrics': metrics(candidate, refs),
            'integrity_passed': candidate['isolation'] and isolated(source) and isp.phi.grad is None and
                not bool(isp.phi.any()) and all(c[k]['passed'] for c in checks.values() for k in ('partition','parity'))}
        write(output/f'record_{index:02d}.json', row)
        assert row['integrity_passed'], 'Observed reference numerical/isolation blocker; stop, no rescue.'
        rows.append(row)
        print(json.dumps({'stage': 'reference', 'episode': index, 'integrity': True}), flush=True)
    assert state_hash(source) == env['source_state_sha256'] and isolated(source)
    write(output/'records.json', rows)
    write(output/'summary.json', aggregate(rows))
    write(output/'completion.json', {'status': 'NEEDS_REVIEW', 'episodes': len(rows),
        'source_hash_after': state_hash(source), 'seconds': time.perf_counter()-started, 'AP_calls': 0})
    write(output/'sha256.json', {p.name: sha(p) for p in sorted(output.glob('*.json'))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--candidate-root', type=Path, required=True)
    p.add_argument('--candidate-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.cohort, a.candidate_root, a.candidate_lock, a.output)
