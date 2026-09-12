"""Check frozen T009 raw receipts against the committed cohort and update equations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    root = args.study
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    env, manifest = read(root/'environment.json'), read(root/'subset.json')
    assert manifest == read(Path('research_log/T009_subset.json'))
    assert digest(root/'subset.json') == '155bb6f047d374342623f48488bcb2b33601a4ecbd372976a433c1b289546e49'
    previous = read(Path('research_log/remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/environment.json'))
    for key in ('clip_model', 'clip_revision', 'clip_sha256', 'detector_sha256', 'target', 'positive_prompts', 'negative_prompts'):
        assert env[key] == previous[key], key
    assert env['ssd'] == read(Path('research_log/T009_ssd_pin.json'))
    assert env['config']['count'] == 1000
    assert {**env['config'], 'count': 200} == previous['config']
    ids = env['evaluated_image_ids']
    assert ids == manifest['image_ids'] and len(ids) == len(set(ids)) == 1000
    assert all(x['overlap_count'] == 0 for x in manifest['exclusions'])
    assert [i for b in manifest['replication_blocks'].values() for i in b] == manifest['selection_order']
    assert all(len(b) == 200 for b in manifest['replication_blocks'].values()) and len(manifest['replication_blocks']) == 5
    cases = env['cases']
    variants = env['config']['variants']
    expected = {(i, c, v) for i in ids for c in cases for v in variants}
    keys, support, fallbacks = set(), {}, Counter()
    error, transfer_error, samples = 0., 0., 0
    with (root/'samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            case = f"{row['family']}_s{row['severity']}"
            key = (row['image_id'], case, row['variant'])
            assert key not in keys
            keys.add(key)
            support_key = (row['image_id'], case)
            signature = json.dumps(row['support'], sort_keys=True)
            if support_key in support:
                assert signature == support[support_key]
            support[support_key] = signature
            assert row['support_count'] == len(row['support']['boxes'])
            history = row['diagnostics']
            assert len(history) == 4 and history[0]['phi'] == [0.]*8
            for step in range(3):
                expected_phi = np.array(history[step]['phi'])-.1*np.array(history[step]['gradient_per_coordinate'])
                np.testing.assert_allclose(history[step+1]['phi'], expected_phi, atol=1e-7, rtol=1e-5)
                error = max(error, float(np.max(np.abs(expected_phi-history[step+1]['phi']))))
            for step in (1, 2, 3):
                np.testing.assert_allclose(row[f'phi_norm_{step}'], np.linalg.norm(history[step]['phi']), atol=1e-7, rtol=1e-5)
            native = row['variant'] != 'global_generic'
            assert row['no_update_fallback'] == (native and row['support_count'] == 0)
            if row['no_update_fallback']:
                assert all(d['phi'] == [0.]*8 for d in history)
                fallbacks[row['variant']+'/'+case] += 1
            if row['variant'] == 'det_pseudo_clip_radius':
                for step, d in enumerate(history):
                    nd, nc = d['detector_gradient_norm'], d['clip_gradient_norm']
                    scale = nc/(nd+1e-12) if nd else 0.
                    np.testing.assert_allclose(d['scale_factor'], scale, atol=1e-7, rtol=1e-5)
                    update = np.array(d['detector_gradient'])*scale
                    np.testing.assert_allclose(d['gradient_per_coordinate'], update, atol=1e-7, rtol=1e-5)
                    transfer_error = max(transfer_error, float(np.max(np.abs(update-d['gradient_per_coordinate']))))
                    assert d['update_applied'] == (step < 3)
            samples += 1
    assert keys == expected and samples == 21000 and len(support) == 7000
    prediction_names = {f'{d}_{c}_{v}.json' for d in env['detectors'] for c in cases for v in ['no_adapt']+variants}
    assert {p.name for p in (root/'predictions').glob('*.json')} == prediction_names
    files = {}
    allowed_ids = set(ids)
    for name in sorted(prediction_names):
        path = root/'predictions'/name
        predictions = read(path)
        image_ids = {r['image_id'] for r in predictions}
        assert image_ids <= allowed_ids
        files[name] = {'sha256': digest(path), 'records': len(predictions), 'images_with_predictions': len(image_ids)}
    receipt = {'images': 1000, 'sample_rows': samples, 'image_conditions': len(support), 'prediction_files': files,
        'complete_unique_pairing': True, 'shared_original_support': True, 'all_phi0_zero': True,
        'cohort_and_models_match_precommitted_pins': True, 'update_equations_and_terminal_step_verified': True,
        'max_phi_update_float32_rounding_error': error, 'max_hybrid_transfer_float32_rounding_error': transfer_error,
        'fallback_counts': dict(fallbacks), 'samples_bytes': (root/'samples.jsonl').stat().st_size,
        'samples_sha256': digest(root/'samples.jsonl'), 'scope': 'raw receipt integrity only; no AP inference or proxy mining'}
    (root/'receipt_audit.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'prediction_files'}, indent=2))


if __name__ == '__main__':
    main()
