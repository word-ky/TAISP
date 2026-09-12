"""Reconstruct retained-prediction proxies and paired loss/proxy disagreements."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from taisp.analysis.detection_proxies import image_proxies, PROXY_DIRECTION
from scripts.report_t007 import ratio_stratum

H = 'det_pseudo_clip_radius'
FAILURES = ('geometry', 'class_score', 'false_positives', 'duplicates')


class ClusterStats:
    def __init__(self, rows):
        ids = np.array([r['image_id'] for r in rows])
        self.ids, self.index = np.unique(ids, return_inverse=True)
        n = len(self.ids)
        self.weights = np.random.default_rng(20260912).multinomial(n, np.full(n, 1/n), size=2000)

    def estimate(self, values, mask=None):
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        if not valid.any():
            return {'estimate': None, 'ci95': None, 'valid_observations': 0}
        sums = np.bincount(self.index, weights=np.where(valid, values, 0), minlength=len(self.ids))
        counts = np.bincount(self.index, weights=valid.astype(float), minlength=len(self.ids))
        numerator = np.einsum('ij,j->i', self.weights, sums, optimize=False)
        denominator = np.einsum('ij,j->i', self.weights, counts, optimize=False)
        boot = numerator[denominator > 0]/denominator[denominator > 0]
        return {'estimate': float(values[valid].mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
                'valid_observations': int(valid.sum()), 'valid_bootstrap_draws': len(boot)}


def failure_flags(delta):
    worse = lambda key: delta[key] is not None and PROXY_DIRECTION[key]*delta[key] < 0
    return {'geometry': any(worse(k) for k in ('recall50', 'recall75', 'best_iou_mean')),
            'class_score': worse('class_score_mean'),
            'false_positives': worse('fp05') or worse('fp50'), 'duplicates': worse('duplicates')}


def summarize(rows):
    stats = ClusterStats(rows)
    loss = np.array([r['loss_delta'] for r in rows])
    good, bad = loss < 0, loss > 0
    out = {'observations': len(rows), 'image_clusters': len(stats.ids),
           'loss_delta': stats.estimate(loss), 'loss_improves': stats.estimate(good), 'proxies': {}, 'failures': {}}
    for key, direction in PROXY_DIRECTION.items():
        delta = np.array([r['delta'][key] if r['delta'][key] is not None else np.nan for r in rows])
        valid = np.isfinite(delta)
        improvement = delta*direction
        out['proxies'][key] = {'mean_delta': stats.estimate(delta),
            'loss_good_proxy_good': stats.estimate(good & (improvement > 0), valid),
            'loss_good_proxy_bad': stats.estimate(good & (improvement < 0), valid),
            'loss_bad_proxy_good': stats.estimate(bad & (improvement > 0), valid),
            'loss_bad_proxy_bad': stats.estimate(bad & (improvement < 0), valid),
            'ties': stats.estimate((loss == 0) | (improvement == 0), valid),
            'proxy_bad_given_loss_good': stats.estimate(improvement < 0, valid & good),
            'undefined_observations': int((~valid).sum())}
    for name in FAILURES:
        flags = np.array([r['failure'][name] for r in rows])
        out['failures'][name] = {'loss_good_proxy_bad': stats.estimate(flags & good),
                                 'given_loss_good': stats.estimate(flags, good)}
    return out


def select_groups(rows, cases):
    yield 'corrupted_overall', [r for r in rows if r['case'] != 'clean_s0']
    for case in cases:
        yield case, [r for r in rows if r['case'] == case]
    for base in ('corrupted_overall', 'clean_s0'):
        chosen = [r for r in rows if (r['case'] == 'clean_s0') == (base == 'clean_s0')]
        for label in ('[0,1)', '[1,2)', '[2,4)', '[4,inf)', 'clip_zero'):
            rr = [r for r in chosen if r['ratio_bin'] == label]
            if rr:
                yield base+' ratio '+label, rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study, out = args.study, args.output
    out.mkdir(parents=True, exist_ok=True)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    env = read(study/'environment.json')
    annotation_bytes = args.annotations.read_bytes()
    assert hashlib.sha256(annotation_bytes).hexdigest() == env['annotation_sha256']
    annotations = json.loads(annotation_bytes)
    ids = env['evaluated_image_ids']
    selected = set(ids)
    subset = {**annotations, 'images': [i for i in annotations['images'] if i['id'] in selected],
              'annotations': [a for a in annotations['annotations'] if a['image_id'] in selected]}
    (out/'annotations_t007.json').write_text(json.dumps(subset)+'\n', encoding='utf-8')
    gt = defaultdict(list)
    for a in subset['annotations']:
        gt[a['image_id']].append(a)
    variants = env['config']['variants']
    cases = list(read(study/'summary.json')[variants[0]])
    samples = [json.loads(s) for s in (study/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    sr = {(r['image_id'], f"{r['family']}_s{r['severity']}", r['variant']): r for r in samples}
    lookup, input_hashes = {}, {}
    with (out/'proxies.jsonl').open('w', encoding='utf-8') as dest:
        for detector in ('source', 'target'):
            for case in cases:
                for variant, k in [('no_adapt', 0)]+[(v, k) for v in variants for k in (1, 3)]:
                    name = f'{detector}_{case}_'+('before' if k == 0 else variant+str(k))+'.json'
                    raw = (study/'predictions'/name).read_bytes()
                    input_hashes[name] = hashlib.sha256(raw).hexdigest()
                    by_image = defaultdict(list)
                    for r in json.loads(raw):
                        by_image[r['image_id']].append(r)
                    for image_id in ids:
                        row = {'detector': detector, 'case': case, 'variant': variant, 'k': k, 'image_id': image_id,
                               **image_proxies(by_image[image_id], gt[image_id])}
                        lookup[(detector, case, variant, k, image_id)] = row
                        dest.write(json.dumps(row, allow_nan=False)+'\n')
                print(f'Proxies {detector} {case}', flush=True)
    contrasts = [(v, 'no_adapt') for v in variants]+[(H, 'det_pseudo')]
    paired, summary = [], {}
    for detector in ('source', 'target'):
        for v, ref in contrasts:
            contrast = v+'-minus-'+ref
            for k in (1, 3):
                chosen = []
                for case in cases:
                    for image_id in ids:
                        current = lookup[(detector, case, v, k, image_id)]
                        baseline = lookup[(detector, case, ref, 0 if ref == 'no_adapt' else k, image_id)]
                        delta = {f: current[f]-baseline[f] if current[f] is not None and baseline[f] is not None else None for f in PROXY_DIRECTION}
                        loss = sr[(image_id, case, v)][detector][f'det_loss_delta_sem{k}']
                        if ref != 'no_adapt':
                            loss -= sr[(image_id, case, ref)][detector][f'det_loss_delta_sem{k}']
                        row = {'detector': detector, 'contrast': contrast, 'k': k, 'image_id': image_id,
                            'case': case, 'loss_delta': loss, 'delta': delta, 'failure': failure_flags(delta),
                            'ratio_bin': ratio_stratum(sr[(image_id, case, H)])}
                        chosen.append(row)
                paired.extend(chosen)
                key = f'{detector}/{contrast}/K{k}'
                summary[key] = {g: summarize(rr) for g, rr in select_groups(chosen, cases)}
                print('Summaries '+key, flush=True)
    with (out/'paired.jsonl').open('w', encoding='utf-8') as f:
        for r in paired:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    indexed = {(r['detector'], r['contrast'], r['k'], r['case'], r['image_id']): r for r in paired}
    joint = {}
    for v, ref in contrasts:
        contrast = v+'-minus-'+ref
        for k in (1, 3):
            same = []
            for r in paired:
                if r['detector'] != 'source' or r['contrast'] != contrast or r['k'] != k:
                    continue
                t = indexed[('target', contrast, k, r['case'], r['image_id'])]
                sf = {f: r['loss_delta'] < 0 and r['failure'][f] for f in FAILURES}
                tf = {f: t['loss_delta'] < 0 and t['failure'][f] for f in FAILURES}
                same.append({**r, 'both_fail': {f: sf[f] and tf[f] for f in FAILURES},
                             'source_only_fail': any(sf.values()) and not any(tf.values()),
                             'target_only_fail': any(tf.values()) and not any(sf.values()),
                             'both_any_fail': any(sf.values()) and any(tf.values())})
            joint[f'{contrast}/K{k}'] = {}
            for g, rr in select_groups(same, cases):
                stats = ClusterStats(rr)
                joint[f'{contrast}/K{k}'][g] = {'observations': len(rr),
                    **{f: stats.estimate([r[f] for r in rr]) for f in ('source_only_fail', 'target_only_fail', 'both_any_fail')},
                    'same_category': {f: stats.estimate([r['both_fail'][f] for r in rr]) for f in FAILURES}}
    (out/'analysis.json').write_text(json.dumps({'contrasts': summary, 'cross_detector': joint}, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out/'receipt.json').write_text(json.dumps({'source_revision': env['source_revision'], 't007_raw_sha256': hashlib.sha256((study/'samples.jsonl').read_bytes()).hexdigest(),
        'full_annotation_sha256': env['annotation_sha256'], 'prediction_hashes': input_hashes,
        'proxy_rows': len(lookup), 'paired_rows': len(paired), 'image_count': len(ids),
        'no_gt_images': [i for i in ids if not any(not a.get('iscrowd', 0) and a['bbox'][2] > 0 and a['bbox'][3] > 0 for a in gt[i])],
        'scope': 'retained post-NMS native-threshold detections; noncrowd positive-area GT; not AP',
        'bootstrap': '2000 multinomial image-cluster resamples seed20260912; sums/counts preserve case clusters; 95percentile exploratory'}, indent=2)+'\n', encoding='utf-8')
    print('Complete offline T008 analysis', flush=True)


if __name__ == '__main__':
    main()
