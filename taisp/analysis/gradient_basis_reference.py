"""R052 post-candidate-lock task reference and oracle span ceiling only."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch
from .orthogonal_candidates import rng_state
from .gradient_basis_math import basis, scores, analyze
from .roi_equivariance import setup, images, sha, write, common_jvp, state_hash, isolated


def verify_inputs(cohort_path, candidate_root, lock_path, permutations_path):
    assert 'taisp.analysis.oracle' not in sys.modules
    assert 'taisp.analysis.common_jacobian' not in sys.modules
    lock = json.loads(lock_path.read_text())
    assert len(lock['candidate_commit']) == 40 and len(lock['files']) == 243
    assert sha(cohort_path) == lock['cohort_sha256']
    assert sha(permutations_path) == lock['permutations_sha256']
    assert sha(candidate_root/'sha256.json') == lock['manifest_sha256']
    for f, h in lock['files'].items():
        assert sha(candidate_root/f) == h, f
    candidates = json.loads((candidate_root/'records.json').read_text())
    manifest = json.loads(cohort_path.read_text())
    permutations = json.loads(permutations_path.read_text())
    assert permutations['cohort_sha256'] == sha(cohort_path)
    assert len(permutations['pair_source_indices']) == 256 and permutations['seed'] == 20261005
    expected = [(im['image_id'], case, im['block'], im['family'])
                for im, cc in zip(manifest['images'], manifest['corrupted_cases']) for case in ['clean_s0', cc]]
    assert len(candidates) == 240 and [(c['image_id'], c['case'], c['block'], c['family']) for c in candidates] == expected
    assert all(c['integrity_passed'] and c['rng']['restored'] and c['native_rng']['restored']
               and all(c['objectives'][k]['parity']['passed'] for k in ['hard', 'clip', 'native']) for c in candidates)
    env = json.loads((candidate_root/'environment.json').read_text())
    completion = json.loads((candidate_root/'completion.json').read_text())
    assert env['source_revision'] == lock['candidate_code_commit']
    assert env['manifest_sha256'] == lock['candidate_images_sha256']
    assert env['source_state_sha256'] == completion['source_hash_after'] == lock['source_state_sha256']
    assert env['clip_state_sha256'] == completion['clip_hash_after'] == lock['clip_state_sha256']
    assert not completion['GT_loaded'] and not completion['forbidden_modules_loaded']
    return candidates, manifest, permutations, {'verified_candidate_files': 243, 'candidate_commit': lock['candidate_commit'],
        'candidate_lock_sha256': sha(lock_path), 'cohort_sha256': sha(cohort_path), 'permutations_sha256': sha(permutations_path),
        'candidate_records_sha256': sha(candidate_root/'records.json'), 'source_state_sha256': lock['source_state_sha256'],
        'before_annotations_and_oracle': True}


def run(cohort_path, candidate_root, lock_path, permutations_path, output):
    candidates, manifest, permutations, preflight = verify_inputs(cohort_path, candidate_root, lock_path, permutations_path)
    output.mkdir(parents=True, exist_ok=False)
    # Construct every candidate span before importing oracle or opening source annotations.
    bases = [basis(*[c['objectives'][k]['gradient'] for k in ['hard', 'clip', 'native']]) for c in candidates]
    write(output/'preflight.json', preflight)
    from .oracle import detector_task_loss
    from .common_jacobian import reference_checks
    started = time.perf_counter()
    source, isp, env = setup()
    assert env['source_state_sha256'] == preflight['source_state_sha256']
    assert sha(manifest['annotation_file']) == manifest['annotation_sha256']
    data = json.loads(Path(manifest['annotation_file']).read_text())
    ids = set(manifest['image_ids'])
    anns = {i: [] for i in ids}
    for a in data['annotations']:
        if a['image_id'] in ids and not a.get('iscrowd', 0) and a['bbox'][2] > 0 and a['bbox'][3] > 0 and a.get('area', 1) > 0:
            anns[a['image_id']].append(a)
    del data
    env.update(candidate_lock_sha256=sha(lock_path), annotation_sha256=manifest['annotation_sha256'],
               permutations_sha256=sha(permutations_path), cohort_sha256=sha(cohort_path))
    write(output/'environment.json', env)
    rows = []
    for index, (info, case, image) in enumerate(images(manifest)):
        c = candidates[index]
        assert (c['image_id'], c['case']) == (info['image_id'], case)
        aa = anns[info['image_id']]
        boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in aa]
        targets = [{'boxes': image.new_tensor(boxes).reshape(-1, 4),
                    'labels': torch.tensor([a['category_id'] for a in aa], device=image.device)}]
        before = state_hash(source); rb = rng_state(image.device)
        y = isp(image, image.new_zeros(8)).detach().requires_grad_(True)
        loss, parts = detector_task_loss(source, y, targets, seed=20260913)
        ct = torch.autograd.grad(loss, y)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(ct).all()
        refs, jac = common_jvp(image, isp, image.new_ones(1, 1, *image.shape[-2:]), {'task': ct})
        r = refs['task']; task = r['global']
        phi = image.new_zeros(8, requires_grad=True)
        direct = torch.autograd.grad((isp(image, phi)*ct).sum(), phi)[0].cpu().tolist()
        checks = reference_checks(task, r['object'], r['background'], direct)
        after = state_hash(source); ra = rng_state(image.device)
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case, 'family': info['family'], 'block': info['block'],
               'task_gradient': task, 'task_loss': loss.item(), 'task_components': {k: v.item() for k, v in parts.items()},
               'basis': bases[index], **scores(bases[index], task), 'pairwise_cosines': c['pairwise_cosines'],
               'candidate_zero_flags': c['zero_flags'], 'jacobian': jac, 'reference_checks': checks,
               'state_before': before, 'state_after': after, 'rng': {'before': rb, 'after': ra, 'restored': rb == ra},
               'candidate_recomputed': False,
               'integrity_passed': before == after == env['source_state_sha256'] and rb == ra and isolated(source)
                   and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values())
                   and all(math.isfinite(v) for v in task)}
        write(output/f'record_{index:03d}.json', row)
        assert row['integrity_passed'], 'reference integrity failure'
        rows.append(row)
        print(json.dumps({'stage': 'reference', 'episode': index}), flush=True)
    assert len(rows) == 240 and sha(candidate_root/'records.json') == preflight['candidate_records_sha256']
    write(output/'records.json', rows)
    summary, null = analyze(rows, permutations['pair_source_indices'])
    write(output/'summary.json', summary)
    write(output/'null_episode_scores.json', null)
    write(output/'completion.json', {'status': 'NEEDS_REVIEW', 'decision': summary['decision'], 'episodes': 240,
          'seconds': time.perf_counter()-started, 'source_hash_after': state_hash(source), 'candidate_recomputed': False, 'AP_calls': 0})
    write(output/'sha256.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cohort', 'candidate-root', 'candidate-lock', 'permutations', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(a.cohort, a.candidate_root, a.candidate_lock, a.permutations, a.output)
