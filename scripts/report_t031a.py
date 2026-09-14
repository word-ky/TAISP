"""Render R049 frozen train/map/holdout receipts; no model calls or fitting."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_log/T031A'
RUNS = ROOT / 'research_log/remote_runs'
CAND = RUNS / '20260914-165905-taisp-t031a-candidates480/artifacts/candidate'
TRAIN = RUNS / '20260914-170909-taisp-t031a-train360-map/artifacts/train'
HOLD = RUNS / '20260914-171544-taisp-t031a-holdout120/artifacts/holdout'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    counts = {}
    for name, folder in [('candidate', CAND), ('train', TRAIN), ('holdout', HOLD)]:
        manifest = read(folder / 'sha256.json')
        for f, h in manifest.items(): assert sha(folder / f) == h, f
        counts[name] = len(manifest)
    cs, tr, hr = [read(p / 'records.json') for p in [CAND, TRAIN, HOLD]]
    assert (len(cs), len(tr), len(hr)) == (480, 360, 120)
    by_id = {(r['image_id'], r['case']): r for r in cs}
    assert all(r['g_hard'] == by_id[r['image_id'], r['case']]['g_hard'] for r in tr + hr)
    assert all(r['integrity_passed'] and r['rng']['restored'] for r in cs + tr + hr)
    assert not {r['image_id'] for r in tr} & {r['image_id'] for r in hr}
    ml = read(ROOT / 'research_log/T031A_map_lock.json')
    assert ml['files'] == read(TRAIN / 'sha256.json')
    r, s, pf = read(TRAIN / 'R.json'), read(HOLD / 'summary.json'), read(HOLD / 'preflight.json')
    assert pf['map_lock']['map_commit'] == ml['map_commit']
    assert pf['map_lock']['map_sha256'] == sha(TRAIN / 'R.json')
    checks = {f: hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h
              for f, h in read(ROOT / 'research_log/T031A_plan.json')['protected_and_prior_LF'].items()}
    assert all(checks.values())
    integrity = dict(verified_raw_files=counts, all960_integrity_rng=True, train_holdout_disjoint=True,
                     candidate_vectors_unchanged=True, protected_prior=checks,
                     map_commit=ml['map_commit'], map_sha256=sha(TRAIN/'R.json'),
                     candidate_records_sha256=sha(CAND/'records.json'),
                     train_records_sha256=sha(TRAIN/'records.json'),
                     holdout_records_sha256=sha(HOLD/'records.json'),
                     norm_preservation_max_abs=max(x['norm_preservation_error'] for x in hr),
                     holdout_preflight=pf, AP_calls=0, candidate_recomputed=False)
    (OUT/'integrity.json').write_text(json.dumps(integrity, indent=2)+'\n', encoding='utf-8')
    for f in ['summary.json', 'preflight.json', 'completion.json']:
        (OUT/f).write_bytes((HOLD/f).read_bytes())
    fields = ['episode_index','image_id','case','block','S_hard','S_cal','Delta','norm_hard','norm_cal',
              'cos_hard_task','cos_cal_task','norm_preservation_error','integrity_passed']
    with (OUT/'per_episode.tsv').open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, delimiter='\t'); w.writeheader()
        w.writerows({k:x[k] for k in fields} for x in hr)
    groups={'overall':s['overall'],'clean':s['clean'],'corrupt':s['corrupt'],
            **s['conditions'],**{'block'+k:v for k,v in s['blocks'].items()}}
    table=[]; coords=[]
    for name, g in groups.items():
        d=g['distributions']
        table.append(f"|{name}|{g['n']}|{d['S_hard']['positive']}|{d['S_cal']['positive']}|{d['Delta']['positive']}|{d['Delta']['median']:.12g}|{d['Delta']['mean']:.12g}|{d['cos_hard_task']['median']}|{d['cos_cal_task']['median']}|")
        for key in ['g_hard','g_cal']:
            c=g['coordinates'][key]
            coords.append(f"|{name}|{key}|{c['max_energy_coordinate']}|{c['max_energy_share']:.12g}|")
    verdict='PASS' if s['passed'] else 'FAIL'
    gates='\n'.join(f'|{k}|{v}|' for k,v in s['gates'].items())
    report=f'''# T031-A / R049 — scientific {verdict}; NEEDS_REVIEW

The one global source-trained O(8) map {'passes the frozen holdout capacity gate; review is required before runtime evaluation' if s['passed'] else 'fails the frozen holdout conjunction. This exact global orthogonal-transport family is closed without rescue tuning'}. This is only a first-order source gradient capacity audit. AP, K-step adaptation, cross-detector evaluation and deployment changes: zero.

## Frozen stages and ordering

Research c76b2e2; cohort/split/plan b019a89; candidate code 620076fe2d5530898ae4e69a9c2ed390c764994b; reference/fit code 14b5c68 committed before any task gradients. All480 candidate results committed d997b6152d1aef4875498f55179664748c17b183; candidate-lock descriptor 7569ed4.

Train360 references and R committed/pushed **6ee34b06813918a70c2a2022ed40985bbb65452b** before holdout launch; map-lock descriptor committed/pushed **bc8ee6df75bb1781ecba0be27a891ce76438b4e0**. The holdout process verifies this descriptor and every train/candidate file before importing oracle/common_jacobian or loading its annotation partition. Saved preflight: {json.dumps(pf['map_lock'])}.

240 fresh COCO train2017 images, seed20261001, excluding3011 prior/reserved IDs plus5000val and1400evaluation IDs. First180 images are train360episodes; remaining60 are holdout120episodes. Each contributes clean plus one frozen severity2 corruption; train60/family and holdout20/family, holdout four15-image blocks with5/family. All image pairs and hashes frozen before outcomes. Cohort SHA256 {sha(ROOT/'research_log/T031A_train_cohort.json')}.

Preparation parsed the source annotation JSON only for outcome-free eligibility and to write separate train/holdout annotation files. Candidate generation used a manifest with no annotation paths and no oracle imports. Train reference loaded only the180-image train annotation subset. Holdout task gradients and metrics were first computed after the R commit; no holdout-based fitting, selection or retuning.

Candidate run20260914-165905-taisp-t031a-candidates480: 480 episodes, {read(CAND/'completion.json')['seconds']}s. Train run20260914-170909-taisp-t031a-train360-map:360 episodes, {read(TRAIN/'completion.json')['seconds']}s, source7569ed4. Holdout run20260914-171544-taisp-t031a-holdout120:120 episodes, {read(HOLD/'completion.json')['seconds']}s, sourcebc8ee6d. Train and holdout use identical reference release20260914-170250-taisp-t031a-reference. Exact commands, start/finish timestamps and exit status are preserved in each raw run.

All model-bearing calculations ran on A6000 CUDA0, frozen Faster R-CNN and unchanged identity ISP/common JVP, source-reference seed20260913. Only the single8x8 float64 SVD and summary algebra use CPU. No detector parameter updates; source state SHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf.

## Map and input locks

Candidate records SHA256 {sha(CAND/'records.json')}; candidate manifest {sha(CAND/'sha256.json')}; candidate descriptor {sha(ROOT/'research_log/T031A_candidate_lock.json')}.

Train records SHA256 {sha(TRAIN/'records.json')}; train manifest {sha(TRAIN/'sha256.json')}; R SHA256 **{sha(TRAIN/'R.json')}**. Holdout records SHA256 {sha(HOLD/'records.json')}; holdout manifest {sha(HOLD/'sha256.json')}.

Exactly one full SVD of C=sum unit(t)unit(h)^T, R=U@Vt; full O(8), no determinant correction, bias, weights, ridge or normalization after transport. Train360episodes/180images; zero unit hard/task={r['zero_unit_hard']}/{r['zero_unit_task']}. Determinant={r['determinant']}; orthogonality maxabs={r['orthogonality_max_abs']}; singular values={r['singular_values']}. Smallest singular value is near zero; reported as a diagnostic, with no refit or determinant selection. Full C and R in T031A/R.json. Holdout maximum norm-preservation absolute error={integrity['norm_preservation_max_abs']}.

## Frozen gate results

Thresholds: S_cal positive >=80/120 overall and45/60corrupt; Delta positive >=72/120overall and36/60corrupt; mean and median Delta strictly positive overall/corrupt; >=3/4 positive block medians; clean median>=0; all integrity checks.

|Gate|Pass|
|---|---|
{gates}

|Group|N|S_hard>0|S_cal>0|Delta>0|Median Delta|Mean Delta|Median cos(h,t)|Median cos(cal,t)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All exact-zero episodes remain counted. Holdout zero hard/cal={s['overall']['zero_hard']}/{s['overall']['zero_cal']}. Full means, standard deviations, mean absolute values, RMS and energy shares for each of8coordinates, each condition and each block are in summary.json. Coordinate index is zero-based; energy share=sum g_j^2/sum_all_coordinates g^2.

|Group|Gradient|Max-energy coordinate|Max share|
|---|---|---:|---:|
{chr(10).join(coords)}

## Integrity, tests and artifacts

All480 candidate and all480 reference integrity/RNG/state/parity receipts passed; all candidate vectors consumed unchanged. Raw file SHA checks: {counts}. Train/holdout IDs disjoint; R committed before holdout; one fit only. All{len(checks)} protected/prior files unchanged;58 candidate and61 reference remote hash checks passed. Existing detector/ISP/tta/deployment modules unchanged. No outcome-driven implementation edits.

Baseline9passed; candidate-focused6passed; reference/Procrustes-focused5passed; full271passed,11skipped in29.78s on A6000 environment. Known NVML/protobuf warnings unchanged. Focused tests cover exact column convention, O(8) reflection, one SVD, eps/zero handling, original hard candidate reuse, RNG/zero receipts, untouched import boundary and frozen mean/median/count gates. No new tests after outcomes because frozen code was unchanged.

Added analysis modules orthogonal_candidates.py, orthogonal_reference.py, orthogonal_transport.py and focused tests; report renderer scripts/report_t031a.py; frozen cohort/plan/code pins, input/map locks and raw runs under research_log. Complete per-episode scores: T031A/per_episode.tsv; distributions/coordinate diagnostics: T031A/summary.json; matrix: T031A/R.json; integrity: T031A/integrity.json. No labels in candidate files. Separate annotation subsets remain under remote project research_log/T031A with hashes in the cohort. No raw receipts deleted.

Transfer incident: first candidate SCP download stalled; terminated only that matching transfer process, existing workflow fallback completed. Archive and all483 raw file hashes verified; no experiment rerun.

Final disposition: {s['disposition']}. Stop NEEDS_REVIEW; no AP/K-step, runtime edits or alternative map family authorized. Future fresh-cohort exclusion should include T031A_train_cohort.json (3251 cumulative prior/reserved IDs).
'''
    (ROOT/'research_log/T031A_report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'verdict':verdict,'gates':s['gates'],'report_sha256':sha(ROOT/'research_log/T031A_report.md')}))

if __name__=='__main__': main()
