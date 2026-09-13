"""T026-A separate post-lock clean-memory reference; no candidate regeneration."""
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
        result[name] = {'S_mem': sc, 'S_global': sg, 'Delta_global': sc-sg,
            'S_obj_current': so, 'Delta_obj': sc-so, 'norms': lengths,
            'cosine_candidate_task': (math.fsum(a*b for a,b in zip(to,g))/(norm(to)*norm(g))
                                      if norm(to) and norm(g) else None),
            'zero_gradient': norm(g) == 0, 'near_zero_gradient': norm(g) <= EPS,
            'material_near_zero': candidate['support_count'] > 0 and norm(g) <= EPS}
    return result


NAME = 'class_conditional_clean_roi_memory'


def describe(values):
    if not values:
        return {'n':0,'positive':0,'positive_fraction':None,'mean':None,'median':None,'quantiles':None}
    ordered=sorted(values)
    def quantile(q):
        x=(len(ordered)-1)*q;i=int(x);j=min(i+1,len(ordered)-1)
        return ordered[i]+(ordered[j]-ordered[i])*(x-i)
    positive=sum(v>0 for v in values)
    return {'n':len(values),'positive':positive,'positive_fraction':positive/len(values),
        'mean':statistics.mean(values),'median':statistics.median(values),
        'quantiles':{str(q):quantile(q) for q in (0.,.25,.5,.75,1.)}}


def aggregate(rows):
    def group(chosen):
        vals=[r['metrics'][NAME] for r in chosen]
        d={k:describe([v[k] for v in vals]) for k in ('S_mem','Delta_global','Delta_obj')}
        d['candidate_norm']=describe([v['norms']['candidate'] for v in vals])
        d['retrieval_cosine']=describe([v for r in chosen for v in r['retrieval']['cosines']])
        return {'n':len(chosen),'distributions':d,
            'S_mem_positive':d['S_mem']['positive'],'Delta_global_positive':d['Delta_global']['positive'],
            'median_Delta_global':d['Delta_global']['median'],
            'zero_gradients':sum(v['zero_gradient'] for v in vals),
            'near_zero_gradients':sum(v['near_zero_gradient'] for v in vals),
            'nonempty_near_zero_images':len({r['image_id'] for r in chosen if r['metrics'][NAME]['material_near_zero']}),
            'distinct_memory_entries':len({v for r in chosen for v in r['retrieval']['entry_ids']}),
            'distinct_classes':len({v for r in chosen for v in r['retrieval']['classes']}),
            'anchor_diversity':{k:describe([r['retrieval'][k] for r in chosen if r['retrieval'][k] is not None])
                for k in ('unique_anchor_hashes','anchor_pairwise_mean_cosine','anchor_pairwise_min_cosine')}}
    overall=group(rows);corrupt=group([r for r in rows if r['case']!='clean_s0'])
    blocks={str(i):group([r for r in rows if r['block']==i]) for i in range(4)}
    conditions={c:group([r for r in rows if r['case']==c]) for c in sorted({r['case'] for r in rows})}
    gate={'S_overall':overall['S_mem_positive']>=30,'S_corrupt':corrupt['S_mem_positive']>=15,
        'Delta_overall':overall['Delta_global_positive']>=30,'Delta_corrupt':corrupt['Delta_global_positive']>=15,
        'median_overall':overall['median_Delta_global']>0,'median_corrupt':corrupt['median_Delta_global']>0,
        'blocks':sum(b['median_Delta_global']>0 for b in blocks.values())>=3,
        'no_material_near_zero':overall['nonempty_near_zero_images']<2,
        'integrity':all(r['integrity_passed'] for r in rows),
        'memory_health':all(r['memory_health_passed'] for r in rows)}
    passed=all(gate.values())
    return {'candidate':NAME,'overall':overall,'corrupt':corrupt,'blocks':blocks,'conditions':conditions,
        'gate':gate,'passed':passed,'status':'NEEDS_REVIEW','nomination':NAME if passed else None,
        'disposition':'nominate_for_research_review_only' if passed else 'close_class_conditional_clean_roi_memory',
        'runtime_or_AP_authorized':False}


def verify_lock(root, lock):
    assert lock['candidate_commit'] and len(lock['files']) == 100
    for name, digest in lock['files'].items():
        assert sha(root/name) == digest, name
    return json.loads((root/'records.json').read_text())


def run(cohort_path, candidate_root, lock_path, memory_root, memory_lock, output):
    # No annotation read/model load is reached until every committed receipt hash passes.
    lock = json.loads(lock_path.read_text())
    candidates = verify_lock(candidate_root, lock)
    manifest = json.loads(cohort_path.read_text())
    assert len(candidates) == 48
    assert sha(cohort_path) == lock['cohort_sha256']
    memlock=json.loads(memory_lock.read_text())
    assert memlock['memory_commit']==lock['memory_commit']
    assert sha(memory_lock)==lock['memory_lock_sha256']
    for name,digest in memlock['files'].items():assert sha(memory_root/name)==digest,name
    health=json.loads((memory_root/'health.json').read_text())
    entries=json.loads((memory_root/'entries.json').read_text())
    assert health['passed'] and not {e['image_id'] for e in entries}&set(manifest['image_ids'])
    for candidate in candidates:
        obj=candidate['objectives'][NAME]
        assert obj['predicted_classes']==candidate['supports']['labels']
        for label,ids in zip(obj['predicted_classes'],obj['retrieval_ids']):
            assert len(ids)==4 and len(set(ids))==4 and all(entries[i]['class_id']==label for i in ids)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, isp, env = setup()
    env.update({'candidate_lock_sha256': sha(lock_path), 'candidate_commit': lock['candidate_commit'],
                'ordering': 'all candidate hashes verified before annotation load', 'cohort_sha256': sha(cohort_path), 'memory_commit':memlock['memory_commit'], 'memory_health':health})
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
        obj=candidate['objectives'][NAME]
        retrieval={'entry_ids':sorted({i for ids in obj['retrieval_ids'] for i in ids}),
            'classes':sorted(set(obj['predicted_classes'])),
            'cosines':[v for values in obj['similarities'] for v in values],
            'unique_anchor_hashes':len(set(obj['anchor_hashes'])),
            'anchor_pairwise_mean_cosine':obj['anchor_pairwise_mean_cosine'],
            'anchor_pairwise_min_cosine':obj['anchor_pairwise_min_cosine']}
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case, 'block': info['block'],
            'mask': candidate['mask'], 'support_count': candidate['support_count'],
            'task_loss': task.item(), 'task_components': {k: v.item() for k,v in components.items()},
            'pseudo_loss': pseudo.item(), 'reference_gradients': refs, 'jacobian': jacobian,
            'reference_checks': checks, 'metrics': metrics(candidate, refs), 'retrieval':retrieval, 'memory_health_passed':health['passed'],
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
    p.add_argument('--memory-root', type=Path, required=True)
    p.add_argument('--memory-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.cohort, a.candidate_root, a.candidate_lock, a.memory_root, a.memory_lock, a.output)
