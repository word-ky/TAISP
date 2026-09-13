"""Reconstruct T024-A tables and report from pinned GPU receipts, stdlib only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'research_log/T024A'
    crun = '20260914-041922-taisp-t024a-candidates32'
    rrun = '20260914-042406-taisp-t024a-reference32'
    cp = root/f'research_log/remote_runs/{crun}/artifacts/candidates'
    rp = root/f'research_log/remote_runs/{rrun}/artifacts/reference'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in (cp, rp):
        for name, digest in read(folder/'sha256.json').items():
            assert sha(folder/name) == digest
    candidates, rows, summary = read(cp/'records.json'), read(rp/'records.json'), read(rp/'summary.json')
    plan = read(root/'research_log/T024A_plan.json')
    protected = {p: hashlib.sha256((root/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h
                 for p, h in plan['protected_modules_LF'].items()}
    assert all(protected.values()) and all(read(out/'remote_code_checks.json').values())
    checks = {'local_protected': protected, 'remote_code_checks': read(out/'remote_code_checks.json'),
              'candidate_receipts_commit': '9be51ee', 'reference_code_commit': '638b763',
              'candidate_files_verified': len(read(cp/'sha256.json')),
              'reference_files_verified': len(read(rp/'sha256.json')),
              'empty_supports': sum(r['support_count'] == 0 for r in rows),
              'reference_integrity_passed': all(r['integrity_passed'] for r in rows),
              'max_reference_global_reverse_relative_l2': max(r['reference_checks'][k]['parity']['relative_l2_error']
                   for r in rows for k in ('task', 'pseudo')),
              'max_partition_absolute_error': max(r['reference_checks'][k]['partition']['max_absolute_error']
                   for r in rows for k in ('task', 'pseudo'))}
    # Explicitly reconstruct horizontal round trips with original JPEG width.
    widths = {r['image_id']: r['width'] for r in read(root/'research_log/T024A_train_cohort.json')['images']}
    checks['max_flip_roundtrip_pixel_error'] = max(abs(x-z) for c in candidates for p in c['pairs']
        for x,z in zip(p['original_box'], [widths[c['image_id']]-p['flipped_box'][2], p['flipped_box'][1],
            widths[c['image_id']]-p['flipped_box'][0], p['flipped_box'][3]]))
    table = []
    for c,r in zip(candidates, rows, strict=True):
        assert (c['image_id'],c['case']) == (r['image_id'],r['case'])
        for name,m in r['metrics'].items():
            t,g = r['reference_gradients']['task']['object'],c['gradients'][name]['object']
            norm = math.sqrt(math.fsum(v*v for v in g))
            sc = math.fsum(a*b for a,b in zip(t,g))/(norm+1e-12)
            assert sc == m['S_c']
            per = c['objectives'][name]['per_object']
            table.append({'episode':r['episode_index'],'image_id':r['image_id'],'case':r['case'],'block':r['block'],
                'candidate':name,'supports':r['support_count'],'mask_area':r['mask']['area_fraction'],
                'loss':c['objectives'][name]['loss'],'per_object_min':min(per) if per else None,
                'per_object_median':statistics.median(per) if per else None,'per_object_max':max(per) if per else None,
                **{k:v for k,v in m.items() if k != 'norms'},**{f'norm_{k}':v for k,v in m['norms'].items()}})
    with (out/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');writer.writeheader();writer.writerows(table)
    (out/'integrity.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    (out/'summary.json').write_bytes((rp/'summary.json').read_bytes())
    lines=['# T024-A — NEEDS_REVIEW; both candidates FAIL', '',
        'R039/29bd004. This analysis-only audit closes the fixed ROI flip-equivariance objective family. '
        'No candidate is nominated. No AP, FCOS/SSD, adaptation steps, training or deployment change was run.', '',
        '## Frozen protocol and ordering', '',
        'Plan/cohort commit `3a7e764`; candidate code `bf78bc5`; complete candidate receipts '
        '`9be51ee` committed and pushed before reference code/run `638b763`. Candidate interpreter '
        'loaded no oracle/reference module. Only the separate cohort-preparation process used annotations '
        'for inherited noncrowd valid-box eligibility; candidate input contained image metadata and conditions only.', '',
        '16 fresh train2017 images, seed20260924, excluding1436 prior source/development/debug IDs '
        'and5000 val IDs (6436 total). Ordered JPEG hashes and full exclusion ledger are committed. '
        'Four blocks of4images;32episodes=16clean+6gamma_s2+5contrast_s2+5color_cast_s2.', '',
        'Original FasterRCNN supports only: unchanged score>=.50, stable descending top20. '
        'The binary union mask is unchanged. Background identity; object8D state is only an analysis '
        'variable at identity. No flip teacher or matching. ROI pair index preserves original support order. '
        'Feature point:1024D box_head output before box_predictor. Feature loss is mean1-cosine; '
        'logit loss is mean symmetric JS atT1. No confidence reweighting or extra objective.', '',
        'Accepted common8-column ISP JVP, float32 ISP and independent float64 global/object/background '
        'reductions onCUDA; exact helper equality tested against A1. Same helper/ISP/mask for candidates '
        'and reference. Source oracle is unchanged native four-loss sum with seed20260913; '
        'current pseudo reference is unchanged det_pseudo loss. eps1e-12. Nonempty candidate '
        'gradient norm<=1e-12 was prospectively declared a blocker; none occurred.', '',
        '## Frozen primary results', '',
        '|Metric|Feature equivariance|Logit equivariance|Required|',
        '|---|---:|---:|---:|']
    a,b=[summary['candidates'][n] for n in ('roi_feat_eq','roi_logit_eq')]
    for label,group,key,required in [('S_c positive overall','overall','S_c_positive','>=20/32'),
        ('S_c positive corrupt','corrupt','S_c_positive','>=10/16'),
        ('Delta_global positive overall','overall','Delta_global_positive','>=20/32'),
        ('Delta_global positive corrupt','corrupt','Delta_global_positive','>=10/16'),
        ('Median Delta_global overall','overall','median_Delta_global','>0'),
        ('Median Delta_global corrupt','corrupt','median_Delta_global','>0')]:
        lines.append(f'|{label}|{a[group][key]}|{b[group][key]}|{required}|')
    lines.append(f"|Positive block medians|{sum(x['median_Delta_global']>0 for x in a['blocks'].values())}/4|{sum(x['median_Delta_global']>0 for x in b['blocks'].values())}/4|>=3/4|")
    lines.extend(['|Frozen conjunction|FAIL|FAIL|all required|','','S_c is dot(t_o,g)/(|g|+eps); '
        'Delta_global subtracts dot(t_s,p_s)/(|p_s|+eps). Positive clean or diagnostic Delta_obj '
        'does not substitute for the frozen conjunction. Both corrupted median improvements are negative.', '',
        '## Integrity and validation', '',
        f"All32 reference records passed reconstruction/parity and isolation. Max global reverse/JVP relativeL2 "
        f"{checks['max_reference_global_reverse_relative_l2']}; max partition error {checks['max_partition_absolute_error']}. "
        f"Max reconstructed float64 flip-roundtrip error {checks['max_flip_roundtrip_pixel_error']} pixels. "
        f"Empty supports:{checks['empty_supports']}; zero/near-zero candidate gradients:0 for both objectives.", '',
        'Source state hash before/after both runs:73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. '
        'Frozen/eval/parameter-grad-None passed; ISP state unchanged. All23 protected module LF hashes unchanged '
        'locally and remotely; all4 new source/test pins match remote release.', '',
        'Baseline13tests passed3.04s; candidate increment18passed3.77s; final focused21passed3.76s. '
        'Full regression204passed,10skipped,4warnings in10.04s. Existing NVML initialization and '
        'protobuf deprecation warnings retained; CUDA computation succeeded. No driver changes.', '',
        f"Candidate run `{crun}`: {read(cp/'completion.json')['seconds']}s. "
        f"Reference run `{rrun}`: {read(rp/'completion.json')['seconds']}s. "
        f"Both exit0, GPU {read(cp/'environment.json')['gpu']}, CUDA{read(cp/'environment.json')['cuda']}. "
        'CPU used only cohort/file/report aggregation.', '',
        '## Durable artifacts and next action', '',
        'Per-episode utilities/cosines/norms/mask/support/per-object statistics: `T024A/per_episode.tsv`. '
        'Exact summaries by condition and block: `T024A/summary.json`. All32 individual candidate '
        'and32 reference records, original/flipped boxes, supports, mask rectangles/hashes, loss vectors, '
        'JVP columns, 8D gradients and environments remain in the two remote_runs directories. '
        'Candidate SHA pins, commit ordering, code pins, complete cohorts/exclusions and integrity receipts are retained.', '',
        'Stop NEEDS_REVIEW. Close fixed ROI flip-equivariance family under R039. No temperature/layer/support '
        'threshold/loss-blend/K/LR/mask/corruption rescue; no T024-B or AP. Await an explicit new research decision.', ''])
    (root/'research_log/T024A_report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
