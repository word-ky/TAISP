"""Tables, retained TP-score distributions and a figure from offline T008 outputs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_t008 import ClusterStats, H


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fields(stat):
    return {'estimate': stat['estimate'], 'ci_low': stat['ci95'][0] if stat['estimate'] is not None else None,
            'ci_high': stat['ci95'][1] if stat['estimate'] is not None else None,
            'valid_observations': stat['valid_observations']}


def fmt(stat, scale=1, digits=4):
    if stat['estimate'] is None:
        return 'NA'
    v, (lo, hi) = stat['estimate'], stat['ci95']
    return f'{v*scale:+.{digits}f} [{lo*scale:+.{digits}f}, {hi*scale:+.{digits}f}]'


def table(lines, headers, rows):
    lines.extend(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
    lines.extend('| '+' | '.join(map(str, row))+' |' for row in rows)
    lines.append('')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, default=Path('research_log/T008'))
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    out = args.analysis
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    a = read(out/'analysis.json')
    ap = read(args.study/'metrics.json')
    receipt = read(out/'receipt.json')
    cases = ['gamma_s1', 'gamma_s2', 'contrast_s1', 'contrast_s2', 'color_cast_s1', 'color_cast_s2', 'clean_s0']
    paired = [json.loads(s) for s in (out/'paired.jsonl').read_text(encoding='utf-8').splitlines()]
    proxies = [json.loads(s) for s in (out/'proxies.jsonl').read_text(encoding='utf-8').splitlines()]
    scalar, failures, cross = [], [], []
    for contrast, groups in a['contrasts'].items():
        for group, r in groups.items():
            for proxy, stats in r['proxies'].items():
                for name, stat in stats.items():
                    if isinstance(stat, dict):
                        scalar.append({'contrast': contrast, 'group': group, 'proxy': proxy, 'statistic': name, **fields(stat)})
            for name, stats in r['failures'].items():
                for measure, stat in stats.items():
                    if isinstance(stat, dict):
                        failures.append({'contrast': contrast, 'group': group, 'failure': name, 'statistic': measure, **fields(stat)})
    for contrast, groups in a['cross_detector'].items():
        for group, r in groups.items():
            for name, stat in {**{k: v for k, v in r.items() if k not in ('observations', 'same_category')},
                               **{'both_'+k: v for k, v in r['same_category'].items()}}.items():
                cross.append({'contrast': contrast, 'group': group, 'event': name, **fields(stat)})
    write_csv(out/'proxy_statistics.csv', scalar)
    write_csv(out/'failure_statistics.csv', failures)
    write_csv(out/'cross_detector.csv', cross)
    pools = defaultdict(list)
    for r in proxies:
        for iou in ('50', '75'):
            pools[(r['detector'], r['case'], r['variant'], r['k'], iou)].extend(r['tp_scores'+iou])
    distributions = []
    for (det, case, variant, k, iou), scores in pools.items():
        q = np.quantile(scores, [.1, .25, .5, .75, .9]).tolist() if scores else [None]*5
        distributions.append({'detector': det, 'case': case, 'variant': variant, 'k': k, 'iou': iou,
                              'tp_count': len(scores), 'mean': float(np.mean(scores)) if scores else None,
                              **dict(zip(['p10', 'p25', 'median', 'p75', 'p90'], q))})
    write_csv(out/'tp_score_distributions.csv', distributions)
    clean = {}
    for det in ('source', 'target'):
        rr = [r for r in paired if r['detector'] == det and r['case'] == 'clean_s0'
              and r['contrast'] == H+'-minus-no_adapt' and r['k'] == 3]
        stats = ClusterStats(rr)
        valid = [r['delta']['best_iou_mean'] is not None for r in rr]
        clean[det] = {
            'recall_changed': stats.estimate([any(r['delta'][m] not in (None, 0) for m in ('recall50', 'recall75')) for r in rr], valid),
            'iou_changed': stats.estimate([r['delta']['best_iou_mean'] not in (None, 0) for r in rr], valid),
            'geometry_worse': stats.estimate([r['failure']['geometry'] for r in rr], valid),
            'geometry_better': stats.estimate([any(r['delta'][m] is not None and r['delta'][m] > 0 for m in ('recall50', 'recall75', 'best_iou_mean')) for r in rr], valid)}
    (out/'clean_changes.json').write_text(json.dumps(clean, indent=2)+'\n', encoding='utf-8')
    lines = ['# T008 computed tables', '',
             'All bracketed intervals: 2,000 image-cluster bootstrap 95% percentile intervals, exploratory and unadjusted. '+
             'Recall deltas are percentage points; IoU, scores and FP counts keep native units. AP is the existing T007 aggregate in points, without CIs.', '',
             '## Hybrid minus no-adapt at K=3: every condition', '']
    for det in ('source', 'target'):
        lines.extend(['### '+det, ''])
        rows = []
        for case in cases:
            r = a['contrasts'][f'{det}/{H}-minus-no_adapt/K3'][case]
            dap = 100*(ap[f'{det}_{case}_{H}3']['AP']-ap[f'{det}_{case}_before']['AP'])
            rows.append([case, f'{dap:+.3f}', fmt(r['loss_delta']),
                *[fmt(r['proxies'][m]['mean_delta'], 100 if m.startswith('recall') else 1)
                  for m in ('recall50', 'recall75', 'best_iou_mean', 'agnostic_iou_mean', 'class_score_mean', 'fp05', 'fp50', 'duplicates')]])
        table(lines, ['Condition', 'AP3 delta', 'Loss delta', 'Recall50 pp', 'Recall75 pp', 'Best IoU', 'Agnostic IoU', 'Class score', 'FP05', 'FP50', 'Duplicates'], rows)
    lines.extend(['## Overall corrupted and clean: hybrid paired contrasts', ''])
    for ref in ('no_adapt', 'det_pseudo'):
        lines.extend(['### Hybrid minus '+ref, ''])
        rows = []
        for det in ('source', 'target'):
            for k in (1, 3):
                for group in ('corrupted_overall', 'clean_s0'):
                    r = a['contrasts'][f'{det}/{H}-minus-{ref}/K{k}'][group]
                    rows.append([det, k, group, fmt(r['loss_delta']),
                        *[fmt(r['proxies'][m]['mean_delta'], 100 if m.startswith('recall') else 1)
                          for m in ('recall75', 'best_iou_mean', 'class_score_mean', 'fp05', 'fp50')]])
        table(lines, ['Detector', 'K', 'Group', 'Loss', 'Recall75 pp', 'Best IoU', 'Class score', 'FP05', 'FP50'], rows)
    lines.extend(['## Failure percentages conditional on detector-specific loss improvement', '',
                  'Categories overlap. Geometry and score exclude undefined GT cases; FP/duplicate categories retain them. Hybrid-minus-raw loss improvement means relative to raw loss, not relative to the original image.', ''])
    rows = []
    for ref in ('no_adapt', 'det_pseudo'):
        for det in ('source', 'target'):
            for k in (1, 3):
                for group in ('corrupted_overall', *cases):
                    r = a['contrasts'][f'{det}/{H}-minus-{ref}/K{k}'][group]
                    rows.append([ref, det, k, group, *[fmt(v['given_loss_good'], 100, 2) for v in r['failures'].values()]])
    table(lines, ['Reference', 'Detector', 'K', 'Group', 'Geometry %', 'Class score %', 'FP %', 'Duplicate %'], rows)
    lines.extend(['## FCOS hybrid-minus-raw scale strata', '',
                  'Initial ratio det/CLIP; bins are descriptive only. All source strata and every per-proxy loss-sign quadrant are in the CSV/JSON artifacts.', ''])
    rows = []
    for k in (1, 3):
        for group, r in a['contrasts'][f'target/{H}-minus-det_pseudo/K{k}'].items():
            if ' ratio ' in group:
                rows.append([k, group, r['observations'],
                    *[fmt(r['proxies'][m]['mean_delta'], 100 if m.startswith('recall') else 1) for m in ('recall75', 'best_iou_mean', 'fp05', 'fp50')],
                    *[fmt(r['failures'][m]['given_loss_good'], 100, 2) for m in ('geometry', 'false_positives')]])
    table(lines, ['K', 'Stratum', 'N', 'Recall75 pp', 'Best IoU', 'FP05', 'FP50', 'Geometry bad given loss good %', 'FP bad given loss good %'], rows)
    lines.extend(['## Same-image failure coincidence: hybrid minus no-adapt', '',
                  'Each detector must improve its own loss to count as loss-good/proxy-bad. Percentages are unconditional across image-condition episodes, with GT-valid denominators for geometry/class score. FP remains defined on the no-GT image. Any-failure unions include all observable categories.', ''])
    rows = []
    for k in (1, 3):
        for group in ('corrupted_overall', *cases):
            r = a['cross_detector'][f'{H}-minus-no_adapt/K{k}'][group]
            rows.append([k, group, *[fmt(r[m], 100, 2) for m in ('source_only_fail', 'target_only_fail', 'both_any_fail')],
                         *[fmt(v, 100, 2) for v in r['same_category'].values()]])
    table(lines, ['K', 'Group', 'Source only %', 'Target only %', 'Both any %', 'Both geometry %', 'Both score %', 'Both FP %', 'Both duplicate %'], rows)
    lines.extend(['## Clean geometry changes without conditioning on loss', '',
                  'Strict numerical change, not a practical-effect threshold. Improvements and worsening can coexist across geometry proxies for one image.', ''])
    table(lines, ['Detector', 'Recall changed %', 'Mean best IoU changed %', 'Any geometry worsens %', 'Any geometry improves %'],
          [[det, *[fmt(v, 100, 2) for v in r.values()]] for det, r in clean.items()])
    lines.extend(['TP distributions (pooled over matched detections; descriptive only) are in `tp_score_distributions.csv`; '+
                  'image-weighted paired TP score means and CIs are in `proxy_statistics.csv`. Changing TP membership prevents a calibration claim.', ''])
    (out/'tables.md').write_text('\n'.join(lines), encoding='utf-8')
    # Focused integrity checks on the frozen receipt join and quadrant partition.
    assert len(proxies) == receipt['proxy_rows'] == 19600
    assert len(paired) == receipt['paired_rows'] == 22400
    assert len({(r['detector'], r['case'], r['variant'], r['k'], r['image_id']) for r in proxies}) == len(proxies)
    assert len({(r['detector'], r['case'], r['contrast'], r['k'], r['image_id']) for r in paired}) == len(paired)
    quadrants = ('loss_good_proxy_good', 'loss_good_proxy_bad', 'loss_bad_proxy_good', 'loss_bad_proxy_bad', 'ties')
    checked = 0
    for groups in a['contrasts'].values():
        for r in groups.values():
            for proxy in r['proxies'].values():
                if proxy['mean_delta']['estimate'] is not None:
                    assert abs(sum(proxy[k]['estimate'] for k in quadrants)-1) < 1e-12
                    checked += 1
    audit = {'proxy_rows': len(proxies), 'paired_rows': len(paired), 'unique_keys': True,
             'quadrant_partitions_checked': checked, 'crowd_image_count': len({r['image_id'] for r in proxies if r['n_crowd_gt']}),
             'no_gt_images': receipt['no_gt_images'], 'sha256': {n: hashlib.sha256((out/n).read_bytes()).hexdigest()
                 for n in ('analysis.json', 'proxies.jsonl', 'paired.jsonl', 'annotations_t007.json')}}
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
