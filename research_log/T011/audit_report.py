"""Reproduce the completed T011 tables/statistics directly from saved COCO panels."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'research_log/remote_runs/20260912-195353-taisp-t011-coco1000-offline/artifacts/study'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
analysis = read(STUDY/'analysis.json')
complete = read(STUDY/'completion.json')
receipt = read(STUDY/'report_receipt.json')
for name, expected in receipt['hashes'].items():
    assert hashlib.sha256((STUDY/name).read_bytes()).hexdigest() == expected
panels = {}
for item in complete['panels']:
    path = STUDY/'panels'/item['panel']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    p = read(path)
    for name, config in p['configs'].items():
        key = name, p['detector'], p['case']
        assert key not in panels
        panels[key] = config['evaluations']
with (STUDY/'AP_tables.csv').open(encoding='utf-8', newline='') as f:
    ap = list(csv.DictReader(f))
checks = 0
for row in ap:
    name, detector, case, group = (row[k] for k in ('config', 'detector', 'case', 'group'))
    v = panels[name, detector, case][group]
    for metric in ('AP', 'AP50', 'AP75'):
        assert float(row[metric]) == 100*v[metric]
        for ref in ('no_adapt', 'full_hybrid'):
            expected = 100*(v[metric]-panels[ref, detector, case][group][metric])
            assert float(row[metric+'_delta_'+ref]) == expected
            checks += 1
names = list(analysis['summaries'])
groups = list(analysis['summaries'][names[0]])
cases = sorted({k[2] for k in panels if k[2] != 'clean_s0'})
detectors = ('source', 'target', 'ssd')
for name in names:
    for group in groups:
        for detector in detectors:
            summary = analysis['summaries'][name][group][detector]
            for ref in ('no_adapt', 'full_hybrid'):
                for metric in ('AP', 'AP50', 'AP75'):
                    expected = math.fsum(100*(panels[name, detector, c][group][metric]-panels[ref, detector, c][group][metric]) for c in cases)/6
                    assert math.isclose(summary['macro_'+metric+'_delta_'+ref], expected, abs_tol=1e-12)
                positive = sum(panels[name, detector, c][group]['AP'] > panels[ref, detector, c][group]['AP'] for c in cases)
                assert summary['positive_conditions_'+ref] == positive
random_names = [r['config'] for r in analysis['configurations'] if r['role'] == 'random']
candidate = analysis['candidate']
def verify_distribution(reported, values):
    values = np.array(values)
    c = reported['candidate']
    for key, vals in [('random', values), ('candidate_minus_random', c-values)]:
        expected = dict(count=len(vals), mean=float(np.mean(vals)), median=float(np.median(vals)), p05=float(np.quantile(vals, .05)), p95=float(np.quantile(vals, .95)))
        for field, v in expected.items():
            assert math.isclose(reported[key][field], v, abs_tol=1e-12)
    assert reported['random_equal_candidate'] == sum(values == c)
    assert reported['random_greater_or_equal_candidate'] == sum(values >= c)
    assert math.isclose(reported['strict_empirical_percentile'], 100*sum(values < c)/200, abs_tol=1e-12)
    assert reported['one_sided_tail'] == (1+sum(values >= c))/201
for detector in detectors:
    for metric, dist in analysis['aggregate_distributions'][detector].items():
        verify_distribution(dist, [analysis['summaries'][n]['aggregate'][detector][metric] for n in random_names])
    for group, dist in analysis['block_distributions'][detector].items():
        verify_distribution(dist, [analysis['summaries'][n][group][detector]['macro_AP_delta_no_adapt'] for n in random_names])
    count = sum(r['candidate'] > r['random']['median'] for r in analysis['block_distributions'][detector].values())
    assert count == analysis['candidate_beats_block_random_median'][detector]
passed = 0
for row in analysis['configurations']:
    per_group = analysis['summaries'][row['config']]
    valid = row['clean_coverage'] <= .5
    valid &= all(per_group['aggregate'][d]['clean_AP_delta_no_adapt'] >= -.10 for d in detectors)
    valid &= all(per_group['aggregate'][d]['macro_AP_delta_no_adapt'] > 0 for d in ('target', 'ssd'))
    valid &= all(sum(per_group[g][d]['macro_AP_delta_no_adapt'] > 0 for g in groups if g != 'aggregate') >= 4 for d in ('target', 'ssd'))
    assert row['R012_rule_met'] == valid
    passed += valid and row['role'] == 'random'
assert passed == analysis['random_R012_pass_count']
criteria = {
    'positive_both_targets': all(analysis['summaries'][candidate]['aggregate'][d]['macro_AP_delta_no_adapt'] > 0 for d in ('target', 'ssd')),
    'above_p95_both_targets': all(analysis['aggregate_distributions'][d]['macro_AP_delta_no_adapt']['candidate'] > analysis['aggregate_distributions'][d]['macro_AP_delta_no_adapt']['random']['p95'] for d in ('target', 'ssd')),
    'four_blocks_both_targets': all(analysis['candidate_beats_block_random_median'][d] >= 4 for d in ('target', 'ssd')),
    'clean_bound_all_detectors': all(analysis['summaries'][candidate]['aggregate'][d]['clean_AP_delta_no_adapt'] >= -.10 for d in detectors),
}
assert all(criteria.values()) == analysis['selection_information_beyond_thinning']
result = dict(AP_rows=len(ap), paired_metric_deltas=checks, panels=len(complete['panels']), configurations=len(names), random_controls=len(random_names), random_R012_pass_count=passed, criteria=criteria, selection_information_beyond_thinning=all(criteria.values()), hashes_verified=True, all_macro_and_distribution_checks_passed=True)
(STUDY/'arithmetic_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
