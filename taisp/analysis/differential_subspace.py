"""R026: shared/differential postmortem from frozen A1 reference vectors only."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from taisp.analysis.spatial_action import EPS


def decompose(obj, bg):
    obj, bg = np.asarray(obj, dtype=np.float64), np.asarray(bg, dtype=np.float64)
    s, d = (obj + bg) / 2, (obj - bg) / 2
    g, shared, diff = np.r_[obj, bg], np.r_[s, s], np.r_[d, -d]
    norm, ns, nd = (float(np.linalg.norm(x)) for x in (g, shared, diff))
    receipt = {'s': s.tolist(), 'd': d.tolist(), 'norm': norm,
               'shared_norm': ns, 'diff_norm': nd,
               'f_diff': nd**2 / (norm**2 + EPS),
               'vector_reconstruction_error': float(np.linalg.norm(g - shared - diff)),
               'energy_reconstruction_error': abs(norm**2 - ns**2 - nd**2),
               'orthogonality_residual': float(np.dot(shared, diff))}
    return receipt, shared, diff


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def episode(task_obj, task_bg, pseudo_obj, pseudo_bg, saved_d_spatial):
    t, ts, td = decompose(task_obj, task_bg)
    p, ps, pd = decompose(pseudo_obj, pseudo_bg)
    shared_dot, diff_dot = float(np.dot(ts, ps)), float(np.dot(td, pd))
    c_shared, c_diff = shared_dot / (p['norm'] + EPS), diff_dot / (p['norm'] + EPS)
    reconstructed = c_shared + c_diff
    error = abs(reconstructed - saved_d_spatial)
    assert error <= 1e-12, 'Saved D_spatial reconstruction failed'
    metrics = {'R_extra': t['norm'] / (t['shared_norm'] + EPS),
               'f_diff_task': t['f_diff'], 'f_diff_pseudo': p['f_diff'],
               'pseudo_diff_amplitude_fraction': p['diff_norm'] / (p['norm'] + EPS),
               'C_shared': c_shared, 'C_diff': c_diff,
               'D_reconstructed': reconstructed, 'D_spatial_saved': saved_d_spatial,
               'D_reconstruction_error': error,
               'cos_shared': cosine(ts, ps), 'cos_diff': cosine(td, pd),
               'shared_dot': shared_dot, 'diff_dot': diff_dot}
    return {'task': t, 'pseudo': p, 'metrics': metrics,
            'diff_dot_sign': int(np.sign(diff_dot)),
            'zero_norm': {'task_diff': t['diff_norm'] == 0,
                          'pseudo_diff': p['diff_norm'] == 0,
                          'either_diff': t['diff_norm'] == 0 or p['diff_norm'] == 0,
                          'either_shared': t['shared_norm'] == 0 or p['shared_norm'] == 0}}


def describe(rows):
    metrics = {}
    for name in rows[0]['analysis']['metrics']:
        values = [r['analysis']['metrics'][name] for r in rows
                  if r['analysis']['metrics'][name] is not None]
        metrics[name] = {'valid_count': len(values),
                         'mean': float(np.mean(values)) if values else None,
                         'median': float(np.median(values)) if values else None,
                         'min': min(values) if values else None,
                         'max': max(values) if values else None,
                         'positive_count': sum(v > 0 for v in values)}
    return {'n': len(rows), 'metrics': metrics,
            'zero_norm_counts': {k: sum(r['analysis']['zero_norm'][k] for r in rows)
                                 for k in rows[0]['analysis']['zero_norm']},
            'diff_dot_sign_counts': {str(s): sum(r['analysis']['diff_dot_sign'] == s for r in rows)
                                     for s in (-1, 0, 1)}}


def triage(scopes):
    overall, corrupt = scopes['overall']['metrics'], scopes['corrupted']['metrics']
    task_blocks = sum(scopes[f'block{i}']['metrics']['R_extra']['median'] > 1.05 for i in range(4))
    pseudo_blocks = sum(scopes[f'block{i}']['metrics']['C_diff']['median'] > 0 for i in range(4))
    task_flags = {'overall_median_R_extra_ge_1_10': overall['R_extra']['median'] >= 1.10,
                  'corrupted_median_R_extra_ge_1_10': corrupt['R_extra']['median'] >= 1.10,
                  'at_least3_block_medians_gt_1_05': task_blocks >= 3}
    pseudo_flags = {'overall_positive_C_diff_ge20': overall['C_diff']['positive_count'] >= 20,
                    'corrupted_positive_C_diff_ge10': corrupt['C_diff']['positive_count'] >= 10,
                    'overall_median_C_diff_positive': overall['C_diff']['median'] > 0,
                    'at_least3_positive_block_medians': pseudo_blocks >= 3}
    task_yes, pseudo_yes = all(task_flags.values()), all(pseudo_flags.values())
    decision = ('close_partition' if not task_yes else
                'close_fixed_pseudo_spatial_branch' if not pseudo_yes else
                'shared_differential_interaction_or_normalization_review')
    return {'task_flags': task_flags, 'pseudo_flags': pseudo_flags,
            'task_relevant': task_yes, 'pseudo_differentially_useful': pseudo_yes,
            'task_blocks_above_1_05': task_blocks, 'positive_C_diff_blocks': pseudo_blocks,
            'decision': decision, 'stop_for_research_review': True}


def summarize(rows):
    groups = {'overall': rows, 'clean': [r for r in rows if r['case'] == 'clean_s0'],
              'corrupted': [r for r in rows if r['case'] != 'clean_s0']}
    groups.update({f'block{i}': [r for r in rows if r['block'] == i] for i in range(4)})
    groups.update({'case/' + case: [r for r in rows if r['case'] == case]
                   for case in sorted({r['case'] for r in rows})})
    scopes = {name: describe(selected) for name, selected in groups.items()}
    return {'scopes': scopes, 'triage': triage(scopes)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(input_root, manifest_path, output):
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((input_root / name).read_bytes()).hexdigest() == expected, name
    raw = json.loads((input_root / 'records.json').read_text())['records']
    assert len(raw) == 32 and [r['episode_index'] for r in raw] == list(range(32))
    assert sum(r['case'] == 'clean_s0' for r in raw) == 16
    assert [r['block'] for r in raw] == [i // 8 for i in range(32)]
    rows = []
    for r in raw:
        assert r['numerical_passed'] and all(r['isolation'].values())
        saved = json.loads((input_root / 'full' / f"record_{r['episode_index']:02d}.json").read_text())
        t, p = r['task']['reference'], r['pseudo']['reference']
        assert t == saved['task']['reference'] and p == saved['pseudo']['reference']
        out = {k: r[k] for k in ('episode_index', 'parent_episode_index', 'image_id', 'case',
                                 'block', 'support_count', 'mask_area_fraction', 'mask_uint8_sha256')}
        out['analysis'] = episode(t['object'], t['background'], p['object'], p['background'],
                                  r['geometry']['D_spatial'])
        rows.append(out)
    result = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'records.json', {'records': rows})
    write_json(output / 'summary.json', result)
    write_json(output / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
               'device': 'cpu', 'dtype': 'float64', 'eps': EPS,
               'source_revision': os.environ.get('TAISP_SOURCE_REVISION'),
               'input_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'inputs': manifest, 'new_model_calls': 0, 'optimizer_steps': 0})
    write_json(output / 'completion.json', {'status': 'completed', 'episodes': len(rows),
               'elapsed_seconds': time.perf_counter() - started, 'triage': result['triage'],
               'max_D_reconstruction_error': max(r['analysis']['metrics']['D_reconstruction_error'] for r in rows),
               'new_model_calls': 0, 'optimizer_steps': 0})
    print(json.dumps({'triage': result['triage'], 'overall': result['scopes']['overall']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('input-root', 'manifest', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    run(args.input_root, args.manifest, args.output)
