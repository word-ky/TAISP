"""Independent arithmetic review of T012 official metrics and paired dose receipts."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = ROOT/'research_log/remote_runs/20260912-212537-taisp-t012-coco1000-half-dose/artifacts/study'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
a, metrics, receipt = (read(S/n) for n in ('analysis.json', 'metrics.json', 'report_receipt.json'))
for name, digest in receipt['hashes'].items():
    if name != 'samples.jsonl':  # Raw archive has a separate SHA delivery receipt.
        assert hashlib.sha256((S/name).read_bytes()).hexdigest() == digest
with (S/'AP_tables.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
with (S/'paired_dose.csv').open(encoding='utf-8', newline='') as f:
    paired = list(csv.DictReader(f))
H, F = 'det_pseudo_half_dose', 'det_pseudo_clip_radius'
checks = 0
for r in rows:
    group, detector, case, variant = (r[k] for k in ('group', 'detector', 'case', 'variant'))
    values = metrics[group][f'{detector}_{case}_{variant}']
    for m in ('AP', 'AP50', 'AP75'):
        assert float(r[m]) == 100*values[m]
        for ref in ('no_adapt', F):
            expected = 100*(values[m]-metrics[group][f'{detector}_{case}_{ref}'][m])
            assert float(r[m+'_delta_'+ref]) == expected
            checks += 1
for group, detectors in a['contrasts'].items():
    for d, variants in detectors.items():
        for v, refs in variants.items():
            selected = [r for r in rows if r['group'] == group and r['detector'] == d and r['variant'] == v and r['case'] != 'clean_s0']
            for ref, result in refs.items():
                values = [float(r['AP_delta_'+ref]) for r in selected]
                assert math.isclose(math.fsum(values)/6, result['macro_corruption_AP_delta'], abs_tol=1e-12)
                assert sum(x > 0 for x in values) == result['positive_corruption_count']
controls = read(ROOT/'research_log/remote_runs/20260912-195353-taisp-t011-coco1000-offline/artifacts/study/analysis.json')
random_names = [r['config'] for r in controls['configurations'] if r['role'] == 'random']
for d, groups in a['random_controls'].items():
    for group, r in groups.items():
        values = np.array([controls['summaries'][n][group][d]['macro_AP_delta_no_adapt'] for n in random_names])
        c = a['contrasts'][group][d][H]['no_adapt']['macro_corruption_AP_delta']
        assert r['candidate'] == c
        assert r['random']['median'] == float(np.median(values))
        assert math.isclose(r['strict_empirical_percentile'], 100*sum(values < c)/200, abs_tol=1e-12)
        assert r['one_sided_tail'] == (1+sum(values >= c))/201
    assert a['positive_blocks'][d] == sum(r['candidate'] > 0 for g, r in groups.items() if g != 'aggregate')
    assert a['blocks_above_random_median'][d] == sum(r['candidate'] > r['random']['median'] for g, r in groups.items() if g != 'aggregate')
clean = [p for p in paired if p['case'] == 'clean_s0']
half_mean = math.fsum(float(p['half_phi3']) for p in clean)/len(clean)
full_mean = math.fsum(float(p['full_phi3']) for p in clean)/len(clean)
assert math.isclose(half_mean, a['safety']['aggregate']['clean_s0']['phi_norm_3']['mean'], abs_tol=1e-12)
assert math.isclose(full_mean, a['safety']['aggregate']['clean_s0']['full_phi3']['mean'], abs_tol=1e-12)
for p in paired:
    den = float(p['full_phi3'])
    assert p['same_original_support'] == 'True'
    if den > 0:
        assert math.isclose(float(p['phi3_ratio']), float(p['half_phi3'])/den, abs_tol=1e-12)
criteria = {
    'both_target_macros_positive': all(a['random_controls'][d]['aggregate']['candidate'] > 0 for d in ('target', 'ssd')),
    'both_targets_four_positive_blocks': all(a['positive_blocks'][d] >= 4 for d in ('target', 'ssd')),
    'both_targets_above_control_medians': all(a['random_controls'][d]['aggregate']['candidate'] > a['random_controls'][d]['aggregate']['random']['median'] and a['blocks_above_random_median'][d] >= 4 for d in ('target', 'ssd')),
    'all_clean_AP_within_bound': all(a['contrasts']['aggregate'][d][H]['no_adapt']['clean_AP_delta'] >= -.1 for d in ('source', 'target', 'ssd')),
    'clean_phi_at_most_65_percent': half_mean <= .65*full_mean,
}
assert criteria == a['criteria'] and all(criteria.values()) == a['half_dose_supported']
result = dict(AP_rows=len(rows), paired_metric_deltas=checks, paired_episodes=len(paired),
    all_macros_signs_and_random_comparisons_verified=True, clean_mean_phi_ratio=half_mean/full_mean,
    criteria=criteria, supported=all(criteria.values()), raw_receipt_verification='Remote report verifies all 7000 supports, step algebra and raw hashes; local raw archive delivery recorded separately.')
(S/'arithmetic_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
