"""R052 label-free H/C/N basis; reuse accepted R047 and generic CLIP."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
from huggingface_hub import hf_hub_download
from taisp.losses.clip_semantic import (load_clip_guidance, CLIP_MODEL, CLIP_REVISION,
                                      POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
from taisp.losses.detector_native import DetectorNativeLoss
from .pseudo_native_candidates import gradients, norm, component_sum_check
from .native_task_signal import pseudo_targets
from .orthogonal_candidates import rng_state
from .roi_equivariance import setup, images, select_predictions, common_jvp, sha, write, isolated, state_hash


def label_free():
    forbidden = [k for k in sys.modules if k.startswith('taisp.') and
                 any(s in k for s in ('annotation', 'oracle', 'reference', 'source_meta', 'common_jacobian'))]
    assert not forbidden, forbidden
    return forbidden


def clip_gradient(image, isp, clip):
    y = isp(image, image.new_zeros(8)).detach().requires_grad_(True)
    loss = clip(image, y)
    ct = torch.autograd.grad(loss, y)[0].detach()
    assert torch.isfinite(loss) and torch.isfinite(ct).all()
    refs, jac = common_jvp(image, isp, image.new_ones(1, 1, *image.shape[-2:]), {'clip': ct})
    g = refs['clip']['global']
    phi = image.new_zeros(8, requires_grad=True)
    direct = torch.autograd.grad((isp(image, phi)*ct).sum(), phi)[0].cpu().tolist()
    n, nd = norm(g), norm(direct)
    cos = math.fsum(a*b for a, b in zip(g, direct))/(n*nd) if n and nd else (1. if n == nd == 0 else 0.)
    rel = norm([a-b for a, b in zip(g, direct)])/max(n, 1e-12)
    return {'gradient': g, 'norm': n, 'loss': loss.item(), 'direct_reverse': direct,
            'parity': {'cosine': cos, 'relative_l2': rel, 'passed': cos >= .999999 and rel <= 1e-5}}, jac


def run(manifest_path, output):
    label_free()
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == {'images', 'corrupted_cases', 'episode_order'}
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, isp, env = setup()
    weight = Path(hf_hub_download(CLIP_MODEL, 'pytorch_model.bin', revision=CLIP_REVISION, local_files_only=True))
    wh = sha(weight)
    assert wh == 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    clip = load_clip_guidance('cuda:0', local_files_only=True).eval()
    ch = state_hash(clip)
    assert ch == '6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c'
    env.update(clip_model=CLIP_MODEL, clip_revision=CLIP_REVISION, clip_weight_sha256=wh,
               clip_state_sha256=ch, positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS,
               manifest_sha256=sha(manifest_path), native_seed=20260930, code_sha256=sha(__file__))
    write(output/'environment.json', env)
    rows = []
    for index, (info, case, image) in enumerate(images(manifest)):
        label_free()
        before = {'source': state_hash(source), 'clip': state_hash(clip)}
        rb = rng_state(image.device)
        with torch.no_grad():
            support = select_predictions(source(image)[0], .5, 20)
        hard = DetectorNativeLoss(source, {'base': support}, 'det_pseudo')
        targets = pseudo_targets(hard)
        values, jac, native_rng = gradients(image, isp, source, hard, targets)
        values['clip'], clip_jac = clip_gradient(image, isp, clip)
        after = {'source': state_hash(source), 'clip': state_hash(clip)}
        ra = rng_state(image.device)
        target_match = (torch.equal(targets[0]['boxes'], hard.boxes) and torch.equal(targets[0]['labels'], hard.labels)
                        and all(not v.requires_grad for v in targets[0].values()))
        support_match = torch.equal(hard.boxes, support['boxes']) and torch.equal(hard.labels, support['labels'])
        supports = {k: v.cpu().tolist() for k, v in support.items()}
        pairs = {}
        for a, b in [('hard', 'clip'), ('hard', 'native'), ('clip', 'native')]:
            na, nb = values[a]['norm'], values[b]['norm']
            pairs[a+'_'+b] = math.fsum(x*y for x, y in zip(values[a]['gradient'], values[b]['gradient']))/(na*nb) if na and nb else None
        row = {'episode_index': index, 'image_id': info['image_id'], 'case': case, 'block': info['block'], 'family': info['family'],
               'objectives': values, 'pairwise_cosines': pairs, 'zero_flags': {k: values[k]['norm'] == 0 for k in ('hard', 'clip', 'native')},
               'support_count': len(hard.boxes), 'supports': supports,
               'supports_sha256': hashlib.sha256(json.dumps(supports, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'pseudo_targets': {k: v.cpu().tolist() for k, v in targets[0].items()}, 'hard_weights': hard.weights.cpu().tolist(),
               'support_match': support_match, 'target_match': target_match, 'jacobian': jac, 'clip_jacobian': clip_jac,
               'native_rng': native_rng, 'rng': {'before': rb, 'after': ra, 'restored': rb == ra},
               'state_before': before, 'state_after': after, 'component_additivity_diagnostic': component_sum_check(values),
               'integrity_passed': before == after == {'source': env['source_state_sha256'], 'clip': ch} and rb == ra
                   and target_match and support_match and native_rng['restored'] and isolated(source) and isolated(clip)
                   and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['parity']['passed'] for v in values.values())}
        write(output/f'record_{index:03d}.json', row)
        assert row['integrity_passed'], 'candidate integrity failure'
        rows.append(row)
        print(json.dumps({'stage': 'candidate', 'episode': index, 'integrity': True}), flush=True)
    assert len(rows) == 240
    write(output/'records.json', rows)
    write(output/'completion.json', {'status': 'candidate_complete', 'episodes': 240, 'seconds': time.perf_counter()-started,
          'source_hash_after': state_hash(source), 'clip_hash_after': state_hash(clip), 'GT_loaded': False,
          'forbidden_modules_loaded': label_free(), 'AP_calls': 0})
    write(output/'sha256.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.output)
