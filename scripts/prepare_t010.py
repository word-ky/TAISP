"""Commit-able T010 signals and decisions, computed without predictions or labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.report_t005 import distribution
from taisp.analysis.gating import HYBRID, SCORES, extract_scores, latency_estimate, rank_mask


def prepare(study, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((study/name).read_text(encoding='utf-8'))
    env, manifest = read('environment.json'), read('subset.json')
    ids, cases = env['evaluated_image_ids'], env['cases']
    rows, zero_cosines = [], 0
    with (study/'samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['variant'] != HYBRID:
                continue
            d = r['diagnostics'][0]
            assert d['phi'] == [0.]*8
            scores = extract_scores(r)
            assert all(np.isfinite(v) for v in scores.values())
            zero_cosines += int(d['detector_gradient_norm'] == 0 or d['clip_gradient_norm'] == 0)
            rows.append({'image_id': r['image_id'], 'case': f"{r['family']}_s{r['severity']}",
                         'scores': scores, 'phi_norm_3': r['phi_norm_3'],
                         'source_setup_seconds': r['source_setup_seconds'],
                         'adapt_seconds_3': r['adapt_seconds_3'], 'identity_step_seconds': d['step_seconds']})
    assert {(r['image_id'], r['case']) for r in rows} == {(i, c) for i in ids for c in cases}
    assert len(rows) == len(ids)*len(cases)
    write = lambda name, value: (output/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    (output/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    image_ids = np.array([r['image_id'] for r in rows])
    configs = [{'name': 'no_adapt', 'score': 'no_adapt', 'coverage': 0., 'orientation': None},
               {'name': 'full_hybrid', 'score': 'full_hybrid', 'coverage': 1., 'orientation': None}]
    masks = [np.zeros(len(rows), dtype=bool), np.ones(len(rows), dtype=bool)]
    for score in SCORES:
        for orientation in ('high', 'low'):
            for coverage in (.25, .5, .75):
                mask, cutoff = rank_mask([r['scores'][score] for r in rows], image_ids, coverage, orientation)
                configs.append({'name': f'{score}_{orientation}_{int(coverage*100)}', 'score': score,
                                'coverage': coverage, 'orientation': orientation, 'cutoff': cutoff})
                masks.append(mask)
    decisions = np.stack(masks)
    for config, mask in zip(configs, decisions):
        config['decision_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
    np.savez_compressed(output/'decisions.npz', selected=decisions)
    write('grid.json', configs)
    groups = {'aggregate': ids, **{name: [i for i in members if i in set(ids)]
                                   for name, members in manifest['replication_blocks'].items()}}
    groups = {name: members for name, members in groups.items() if members}
    distributions, safety = {}, {c['name']: {} for c in configs}
    for group, members in groups.items():
        member_set = set(members)
        distributions[group] = {}
        for c in configs:
            safety[c['name']][group] = {}
        for case in ['all', 'corrupted_overall', *cases]:
            indexes = [j for j, r in enumerate(rows) if r['image_id'] in member_set and
                       (case == 'all' or (r['case'] != 'clean_s0' if case == 'corrupted_overall' else r['case'] == case))]
            chosen = [rows[j] for j in indexes]
            distributions[group][case] = {score: distribution([r['scores'][score] for r in chosen]) for score in SCORES}
            for c, mask in zip(configs, decisions):
                selected = mask[indexes]
                phi = np.array([r['phi_norm_3'] for r in chosen])*selected
                safety[c['name']][group][case] = {
                    'rows': len(chosen), 'selected': int(selected.sum()), 'coverage': float(selected.mean()),
                    'nonzero_phi_fraction': float(np.mean(phi > 0)), 'effective_phi3': distribution(phi),
                    'latency_estimate_seconds': distribution([latency_estimate(r, c['score'], bool(s)) for r, s in zip(chosen, selected)])}
    write('score_distributions.json', distributions)
    write('safety.json', safety)
    write('manifest.json', {'source_revision': os.environ.get('TAISP_SOURCE_REVISION', 'unrecorded'),
                           'input_study_revision': env['source_revision'], 'image_ids': ids, 'cases': cases,
                           'groups': groups, 'full_cohort': len(ids) == 1000, 'rows': len(rows),
                           'scores': list(SCORES), 'unavailable_scores': [], 'zero_vector_cosines_coded_zero': zero_cosines,
                           'configurations': len(configs), 'pooled_rank_reference': 'all saved hybrid image-condition rows',
                           'ties': 'ascending image ID then original hybrid row ordinal',
                           'inputs_sha256': {n: hashlib.sha256((study/n).read_bytes()).hexdigest()
                                             for n in ('samples.jsonl', 'subset.json', 'environment.json')},
                           'outputs_sha256': {n: hashlib.sha256((output/n).read_bytes()).hexdigest()
                                              for n in ('scores.jsonl', 'decisions.npz', 'grid.json', 'score_distributions.json', 'safety.json')},
                           'labels_or_prediction_files_read': False})
    print(json.dumps({'images': len(ids), 'rows': len(rows), 'configurations': len(configs),
                      'zero_vector_cosines': zero_cosines, 'labels_or_prediction_files_read': False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.study, args.output)


if __name__ == '__main__':
    main()
