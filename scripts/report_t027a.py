"""Generate T027-A report from immutable CUDA receipts; standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'research_log/T027A'
    runs = {'candidates': '20260914-073543-taisp-t027a-candidates48',
            'reference': '20260914-074105-taisp-t027a-reference48'}
    dirs = {k: root / f'research_log/remote_runs/{v}/artifacts/{k}' for k, v in runs.items()}
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in dirs.values():
        for f, h in read(folder / 'sha256.json').items():
            assert sha(folder / f) == h
    cs = read(dirs['candidates'] / 'records.json')
    rs = read(dirs['reference'] / 'records.json')
    s = read(dirs['reference'] / 'summary.json')
    table = []
    for c, r in zip(cs, rs, strict=True):
        assert (c['image_id'], c['case']) == (r['image_id'], r['case'])
        for key, g in [('S_orig', c['original']['gradient']), ('S_flip', c['flipped']['gradient']), ('S_cons', c['g_cons'])]:
            score = math.fsum(a*b for a, b in zip(r['task_gradient'], g)) / (math.sqrt(math.fsum(v*v for v in g)) + 1e-12)
            assert score == r[key]
        table.append({k: r[k] for k in ['episode_index', 'image_id', 'case', 'block', 'S_orig', 'S_cons', 'S_flip', 'Delta_cons', 'agreement', 'group', 'both_nonzero', 'abstain', 'integrity_passed']} | {
            'norm_original': c['norm_original'], 'norm_flip': c['norm_flip'],
            'original_supports': c['original']['support_count'], 'flip_supports': c['flipped']['support_count']})
    with (out / 'per_episode.tsv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), delimiter='\t'); w.writeheader(); w.writerows(table)
    (out / 'summary.json').write_bytes((dirs['reference'] / 'summary.json').read_bytes())
    plan = read(root / 'research_log/T027A_plan.json')
    pins = plan['protected_modules_LF'] | plan['prior_study_LF']
    checks = {p: hashlib.sha256((root/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h for p, h in pins.items()}
    remote = read(out / 'remote_code_checks.json')
    assert all(checks.values()) and all(remote.values())
    integrity = {'local_code_checks': checks, 'remote_code_checks': remote,
        'all48_integrity': all(r['integrity_passed'] for r in rs),
        'all48_candidate_isolation': all(c['isolation'] for c in cs),
        'max_candidate_relative_l2': max(c[v]['parity']['relative_l2'] for c in cs for v in ['original', 'flipped']),
        'max_reference_relative_l2': max(r['reference_checks']['parity']['relative_l2_error'] for r in rs),
        'max_partition_error': max(r['reference_checks']['partition']['max_absolute_error'] for r in rs),
        'empty_original_supports': sum(c['original']['support_count'] == 0 for c in cs),
        'empty_flip_supports': sum(c['flipped']['support_count'] == 0 for c in cs),
        'minimum_view_gradient_norm': min(c[k] for c in cs for k in ['norm_original', 'norm_flip'])}
    (out/'integrity.json').write_text(json.dumps(integrity, indent=2)+'\n', encoding='utf-8')
    a, b = s['overall']['distributions'], s['corrupt']['distributions']
    rel = s['reliability']
    rows = []
    for label, metric, field, required in [('S_cons positive', 'S_cons', 'positive', '>=30/48; >=15/24'), ('Delta_cons positive', 'Delta_cons', 'positive', '>=30/48; >=15/24'), ('Median Delta_cons', 'Delta_cons', 'median', '>0 both')]:
        rows.append(f'|{label}|{a[metric][field]}|{b[metric][field]}|{required}|')
    for label, key, required in [('Spearman(agreement,S_orig)', 'spearman', '>=0.30 both'), ('High minus low positive fraction', 'positive_fraction_difference', '>=0.20 both'), ('High minus low median S_orig', 'median_difference', '>0 both')]:
        rows.append(f"|{label}|{rel['overall'][key]}|{rel['corrupt'][key]}|{required}|")
    blocks = {k: v['distributions']['Delta_cons']['median'] for k, v in s['blocks'].items()}
    report = f'''# T027-A — NEEDS_REVIEW; E1 FAIL, E2 FAIL

R042 executed. Close the exact horizontal-flip gradient consensus/reliability family under the frozen rules. No AP, K-step, runtime integration, threshold/view/weight sweep or next-stage nomination.

## Protocol and ordering

Plan/cohort 887ad8a; candidate implementation and successful preflight c713818. All48 candidate records AND overall median agreement were committed/pushed in 0b1435b BEFORE original-task reference code ed444b6 and reference execution. Candidate process is GT-free; annotations enter only separate post-lock task reference. Cohort eligibility uses inherited GT-valid/readable-image criteria before outcomes, without outcome-based selection.

24 fresh train2017 images, seed20260927, four6imageblocks;24clean and8each gamma_s2/contrast_s2/color_cast_s2. Excluded2687 previous source/memory/audit images plus5000val IDs. Full image/JPEG/exclusion hashes are retained. No source memory, CLIP, feature/logit consistency or box geometry objective is used.

Preflight tested unchanged ISP flip commutation at identity and three fixed nonzero states on two synthetic shapes, including exact0/1 pixels, before teacher outcomes. All8 cases passed atol2e-7/rtol1e-6; max absolute error5.960464477539063e-08. No ISP repair.

Each raw original/flip view independently generates unchanged FasterRCNN pseudo supports: score>=.50, stable descending top20. Supports are frozen per view with no matching/filter/transfer. Unchanged det_pseudo gradients use the same global8D ISP coordinates, without remapping. Normalize each nonzero view gradient; symmetrically normalize their sum. eps1e-12; either zero view gives agreement=-1 and zero consensus, degenerate sum also abstains. No CLIP magnitude or confidence reweighting.

Accepted common8-column float32 ISP JVP with float64 reductions; unchanged original-view native four-loss task reference at seed20260913. S_orig and S_cons are dot(task_gradient,direction)/(norm(direction)+eps); Delta_cons=S_cons-S_orig. S_flip is diagnostic only. No flipped-task oracle.

## Frozen E1 and E2

|Metric|Overall48|Corrupt24|Required|
|---|---:|---:|---|
''' + '\n'.join(rows) + f'''

Positive Delta block medians: {sum(v>0 for v in blocks.values())}/4 (required>=3); exact medians {blocks}.
E1 conjunction FAIL. Overall/corrupted mean Delta={a['Delta_cons']['mean']}/{b['Delta_cons']['mean']}; positive means do not override the failed count and corrupted-median conditions.

Candidate-only overall median agreement pinned at {s['pinned_median_agreement']}; high>=this value, low<this same value for both analyses. No recomputed corrupted threshold. Overall high/low n={rel['overall']['high']['n']}/{rel['overall']['low']['n']}, positive S_orig={rel['overall']['high']['positive']}/{rel['overall']['low']['positive']}; corrupt high/low n={rel['corrupt']['high']['n']}/{rel['corrupt']['low']['n']}, positives={rel['corrupt']['high']['positive']}/{rel['corrupt']['low']['positive']}. Both-view nonzero coverage48/48 and24/24; zero consensus/abstentions0. E2 conjunction FAIL. Spearman uses average tied ranks; undefined correlation or empty groups fail without alternate cuts.

Full overall/clean/corrupt/family/block counts, fractions, means, medians and linear quantiles[0,.25,.5,.75,1] for all utilities/agreement, E1/E2 booleans, high/low distributions and coverage are in T027A/summary.json. per_episode.tsv contains every episode. No episodes dropped.

## Integrity and execution

All48 candidate isolation and reference integrity checks passed. Max candidate reverse/JVP relative L2={integrity['max_candidate_relative_l2']}; reference={integrity['max_reference_relative_l2']}; partition error={integrity['max_partition_error']}. Empty original/flip supports={integrity['empty_original_supports']}/{integrity['empty_flip_supports']}; minimum view gradient norm={integrity['minimum_view_gradient_norm']}.32 protected/prior files unchanged locally;36 protected/prior/new code pins match remote. Raw receipt hashes verified after archive SHA verification.

Source frozen/eval, parameter grads None; source state before/after73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. All model-bearing runs on NVIDIA RTX A6000 cuda:0, torch2.4.0+cu121. Candidate {runs['candidates']} completed14.41063988499809s; reference {runs['reference']} completed18.390366662992164s; both exit0. Tests: baseline13pass3.33s, candidate17pass3.85s, finalfocused20pass3.98s; full227pass11skip4warnings10.17s. Existing NVML/protobuf warnings retained. CPU used only for manifests and report aggregation.

Raw archive SHA256: candidate04b35c87896ef0591b2c93cd99b002bdebeab59c8d82257f0a74ea9efddff450; reference899d8e47111fdf1eaa587acd8b98ecb02ad1000fb5a440701158da896934caca.

Stop NEEDS_REVIEW. Future fresh cohorts use T027A_train_cohort.json as additional_source:2687prior+24new=2711source IDs plus5000val exclusion. R042 accepted the earlier T025 numeric correction; historical data and research-owned coordination files remain unchanged.
'''
    (root/'research_log/T027A_report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
