"""Freeze 200 exact condition/block matched controls without reading AP or predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from scripts.report_t005 import distribution
from taisp.analysis.random_controls import CANDIDATE, SEEDS, matched_mask, stratum_indices


def prepare(original, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda n: json.loads((original/n).read_text(encoding='utf-8'))
    meta, old_grid = read('manifest.json'), read('grid.json')
    rows = [json.loads(s) for s in (original/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
    matrix = np.load(original/'decisions.npz')['selected']
    candidate_index = next(j for j, c in enumerate(old_grid) if c['name'] == CANDIDATE)
    candidate = matrix[candidate_index].copy()
    candidate_hash = hashlib.sha256(candidate.tobytes()).hexdigest()
    assert candidate_hash == old_grid[candidate_index]['decision_sha256']
    if meta['full_cohort']:
        assert candidate_hash == '7a02e6d66336ef39fbd4b55c3a5aba564aacee56bd2d7405b62ce9d811a511f6'
        assert old_grid[candidate_index]['cutoff']['cutoff_score'] == .8499477751114789
    strata = stratum_indices(rows, meta['groups'])
    masks = [np.zeros(len(rows), dtype=bool), np.ones(len(rows), dtype=bool), candidate]
    configs = [{'name': name, 'role': 'anchor'} for name in ('no_adapt', 'full_hybrid', CANDIDATE)]
    counts = {f'{case}/{block}': int(candidate[indexes].sum()) for (case, block), indexes in strata.items()}
    for seed in SEEDS:
        mask = matched_mask(rows, strata, candidate, seed)
        assert all(int(mask[indexes].sum()) == int(candidate[indexes].sum()) for indexes in strata.values())
        masks.append(mask)
        configs.append({'name': f'random_{seed}', 'role': 'random', 'seed': seed})
    selected = np.stack(masks)
    for config, mask in zip(configs, selected):
        config['decision_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
    write = lambda n, v: (output/n).write_text(json.dumps(v, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'decisions.npz', selected=selected)
    shutil.copyfile(original/'scores.jsonl', output/'scores.jsonl')
    write('seeds.json', list(SEEDS))
    write('grid.json', configs)
    write('matching.json', {'candidate': CANDIDATE, 'candidate_decision_sha256': candidate_hash,
        'candidate_cutoff': old_grid[candidate_index]['cutoff'], 'stratum_selected_counts': counts,
        'strata': len(strata), 'draws': len(SEEDS), 'all_stratum_counts_equal': True,
        'unique_random_masks': len({c['decision_sha256'] for c in configs if c['role'] == 'random'}),
        'hash_payload': 'TAISP-T011|{seed}|{image_id}|{case}', 'order': 'digest bytes ascending then image ID'})
    phi = np.asarray([r['phi_norm_3'] for r in rows])
    setup = np.asarray([r['source_setup_seconds'] for r in rows])
    adaptation = np.asarray([r['adapt_seconds_3'] for r in rows])
    safety = {c['name']: {} for c in configs}
    for group, ids in meta['groups'].items():
        members = set(ids)
        for config in configs:
            safety[config['name']][group] = {}
        for case in ('clean_s0', 'corrupted_overall'):
            indexes = [j for j, r in enumerate(rows) if r['image_id'] in members and
                       (r['case'] == 'clean_s0' if case == 'clean_s0' else r['case'] != 'clean_s0')]
            for config, mask in zip(configs, selected):
                effective = mask[indexes]*phi[indexes]
                safety[config['name']][group][case] = {
                    'rows': len(indexes), 'selected': int(mask[indexes].sum()),
                    'coverage': float(mask[indexes].mean()), 'nonzero_phi_fraction': float(np.mean(effective > 0)),
                    'effective_phi3': distribution(effective),
                    'latency_estimate_seconds': distribution(setup[indexes]+mask[indexes]*adaptation[indexes])}
    write('safety.json', safety)
    outputs = ('decisions.npz', 'scores.jsonl', 'grid.json', 'seeds.json', 'matching.json', 'safety.json')
    write('manifest.json', {'source_revision': os.environ.get('TAISP_SOURCE_REVISION', 'unrecorded'),
        'original_preparation_revision': meta['source_revision'], 'image_ids': meta['image_ids'],
        'cases': meta['cases'], 'groups': meta['groups'], 'full_cohort': meta['full_cohort'],
        'rows': len(rows), 'configurations': len(configs), 'random_draws': len(SEEDS), 'candidate': CANDIDATE,
        'candidate_decision_sha256': candidate_hash, 'labels_or_prediction_files_read': False,
        'inputs_sha256': {n: hashlib.sha256((original/n).read_bytes()).hexdigest() for n in
                         ('scores.jsonl', 'decisions.npz', 'grid.json', 'manifest.json')},
        'outputs_sha256': {n: hashlib.sha256((output/n).read_bytes()).hexdigest() for n in outputs}})
    print(json.dumps({'images': len(meta['image_ids']), 'observations': len(rows), 'configurations': len(configs),
                      'random_draws': len(SEEDS), 'matched_strata': len(strata), 'all_counts_equal': True}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.original, a.output)


if __name__ == '__main__':
    main()
