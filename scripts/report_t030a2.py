"""Summarize R047 frozen numerical receipts and candidate lock; no model imports."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_log/T030A2'
DIAG=ROOT/'research_log/remote_runs/20260914-151141-taisp-t030a2-attribution8x5/artifacts/diagnostic'
CAND=ROOT/'research_log/remote_runs/20260914-151622-taisp-t030a2-candidates120/artifacts/candidate'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)


def main():
    index=read(DIAG/'tensor_storage_index.json');tensors={Path(v['remote_path']).name:v['sha256'] for v in index['tensors']}
    for f,h in read(DIAG/'sha256.json').items():assert (tensors[f] if f.endswith('.pt') else sha(DIAG/f))==h
    for f,h in read(CAND/'sha256.json').items():assert sha(CAND/f)==h
    s=read(DIAG/'summary.json');dc=read(DIAG/'completion.json');cc=read(CAND/'completion.json');records=read(CAND/'records.json')
    assert s['passed'] and len(records)==120 and cc['episodes']==120
    episodes=[read(DIAG/f'episode_{i:02d}.json') for i in range(8)]
    rows=[]
    for e in episodes:
        for r in e['repetitions']:
            rows.append({'episode':e['episode_index'],'image_id':e['image_id'],'case':e['case'],'repetition':r['repetition'],
                'chain_relative':r['chain_parity']['relative_l2'],'chain_cosine':r['chain_parity']['cosine'],
                'chain_passed':r['chain_parity']['passed'],'scalar_self_noise':r['scalar_self_noise'],
                'scalar_min_cosine':r['scalar_min_cosine'],
                **{f'{k}_relative':r['image_space'][k]['relative_l2'] for k in ['multi','sep','rev']},
                'integrity_passed':r['integrity_passed'],'old_additivity_passed':r['old_additivity']['passed'],
                'cotangent_sha256':r['cotangent_sha256']})
    tsv(OUT/'per_repetition.tsv',rows)
    tsv(OUT/'candidate_per_episode.tsv',[{'episode':r['episode_index'],'image_id':r['image_id'],'case':r['case'],
        'support_count':r['support_count'],'native_loss':r['objectives']['native']['loss'],
        'native_norm':r['objectives']['native']['norm'],'native_chain_rel':r['objectives']['native']['parity']['relative_l2'],
        'native_chain_cos':r['objectives']['native']['parity']['cosine'],'component_sum_diagnostic':r['component_sum']['passed'],
        'multi_relative':r['multi_native_relative_l2'],'rng_restored':r['rng']['restored']} for r in records])
    checks=read(ROOT/'research_log/T030A2_plan.json')['protected_and_prior_LF']
    changes=['taisp/analysis/pseudo_native_candidates.py','tests/test_pseudo_native.py']
    preserved={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in checks.items() if f not in changes}
    corrected=read(ROOT/'research_log/T030A2_corrected_code_pins.json')
    corrected_ok={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in corrected.items()}
    assert all(preserved.values()) and all(corrected_ok.values())
    integrity={'protected_prior_unchanged':preserved,'conditional_correction_pins':corrected_ok,
        'diagnostic_remote_checks':read(OUT/'remote_code_checks.json'),'corrected_remote_checks':read(OUT/'corrected_remote_code_checks.json'),
        'candidate_all120_integrity':all(r['target_match'] and r['support_match'] and r['isolation'] and r['rng']['restored'] and all(v['parity']['passed'] for v in r['objectives'].values()) for r in records),
        'candidate_component_additivity_diagnostic_pass_count':sum(r['component_sum']['passed'] for r in records),
        'source_state_final_same':dc['source_hash_after']==cc['source_hash_after']=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf',
        'candidate_records_sha256':sha(CAND/'records.json'),'candidate_manifest_sha256':sha(CAND/'sha256.json'),
        'diagnostic_manifest_sha256':sha(DIAG/'sha256.json'),'GT_loaded':False,'AP_calls':0}
    assert integrity['candidate_all120_integrity'] and integrity['source_state_final_same']
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    lines=[]
    for r in s['episodes']:
        m=r['formulation_medians']
        lines.append(f"|{r['image_id']} / {r['case']}|{r['native']['max_pairwise_relative_l2']:.9g}|{r['native']['min_pairwise_cosine']:.12g}|{r['max_chain_relative']:.9g}|{r['scalar_self_noise_median']:.9g}|{m['multi']:.9g}|{m['sep']:.9g}|{m['rev']:.9g}|{r['noise_explained']}|")
    report=f'''# T030-A2 / R047 — numerical PASS; 120 candidates locked; NEEDS_REVIEW

All four precommitted numerical gates pass on the fresh next four T030 image pairs. The conditionally authorized analysis correction and full 120-episode GT-free candidate regeneration are complete. Stop for research review. There is no source-reference alignment, AP, K-step adaptation or scientific utility conclusion. R045/R046 remain BLOCKED under their original rules; no retrospective relabeling.

## Provenance and reuse

Research b083ffd; pre-outcome plan/manifest1d4b158; diagnostic codee5ed293; diagnostic result lock8c5991e; corrected runner59efa11. Audit cohort IDs27717,74938,347235,413056 retain originalT030 order/corruptions and JPEG hashes. Original60-image cohortSHA3ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752. Eight-episode manifestSHA{sha(OUT/'diagnostic_images.json')}.

Unchanged frozen Faster R-CNN weightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384 and stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. A6000CUDA0, torch2.4.0+cu121, native samplingseed20260930/setupseed20260913, unchanged nondeterministic CUDA environment. No CuBLAS setting/probe change. Existing native_task_signal, DetectorNativeLoss pseudo-target support, roi_equivariance setup/images/common_jvp reused. No annotation JSON opened by diagnostic/candidate process.

Audit run20260914-151141-taisp-t030a2-attribution8x5, release20260914-150932-taisp-t030a2-attribution, completed40repetitions in{dc['seconds']}s. Exact command in saved run.sh: python -m taisp.analysis.native_chain_attribution --manifest research_log/T030A2/diagnostic_images.json --output "$AUTODL_ARTIFACTS_DIR/diagnostic", TAISP_SOURCE_REVISION=e5ed293.

Support is generated once per episode, score>=.50/stabletop20, identical detached boxes/classes for five fresh native forward graphs. Each retained graph executes sum_a, sum_b, sum_c, one-engine multi-output, canonical4 and reversed4 component VJPs in that exact order. The exact detached sum_a cotangent contracts both the existing common8-column ISP JVP and the corresponding original ISP graph VJP. No second detector call enters the chain comparison. Native candidate is sum_a's projected vector, never a mean or selected diagnostic. All relative discrepancies use norm(a-b)/max(norm(a),norm(b),1e-12).

## Frozen gates

|Gate|Observed|Verdict|
|---|---|---|
|1: all8 candidate cosine>=.99999 and pairwise rel<=.002|mincos{min(r['native']['min_pairwise_cosine'] for r in s['episodes'])}; maxrel{max(r['native']['max_pairwise_relative_l2'] for r in s['episodes'])}|PASS|
|2: every same-cotangent ISP chain rel<=1e-5, cos>=.999999, finite/nonzero|40/40; maxrel{max(r['chain_relative'] for r in rows)}; mincos{min(r['chain_cosine'] for r in rows)}|PASS|
|3: all8 finite scalar self-noise, median<=.002, withinrep cos>=.99999|maxepisode median{max(r['scalar_self_noise_median'] for r in s['episodes'])}; mincos{min(r['scalar_min_cosine'] for r in rows)}|PASS|
|4: >=7/8 allthree medians<=2*noise+1e-5; individual<=.002; integrity|8/8; maxindividual{max(r['max_formulation_relative'] for r in s['episodes'])}; all40integrity|PASS|

|Image / case|candidate maxrel|candidate mincos|chain maxrel|scalar median noise|multi median|separate median|reverse median|noise explained|
|---|---:|---:|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(lines)+f'''

Fullprecision all40 comparisons inper_repetition.tsv; all8D vectors, image cotangent SHA/norm/dtype, scalar losses, pseudo-target/support and RNG/state hashes in raw per-repetition/episodeJSON. Six full cotangents per repetition retained in40server .pt files, verified against rawsha256.json; tensor_storage_index.json locates all raw tensors. Local D disk constraint means tensors stay expanded onserver; small complete JSON/TSV/hash receipts are local andGitHub. No tensor deletion or new image selection occurred.

These measurements separate the ISP chain from detector reverse-sweep variation: same-cotangent ISP errors are below1e-6, while repeated scalar reverse sweeps on one detector graph show roughly4e-4–5e-4 image-gradient noise. Formulation discrepancies fit the frozen measured-noise bounds. This supports the bounded numerical attribution; it does not identify a specific CUDA operator or establish task utility.

## Conditional correction and 120-record lock

After diagnostic PASS was committed, only pseudo_native_candidates.py and its tests were corrected: keep authoritative scalar-sum reverse; keep four component/additivity receipts; remove old component-additivity from stop assertion; log one-engine multi-output diagnostic; keep same-cotangent ISP parity for every gradient. The original candidate runner had no fresh-detector direct-phi equality assertion, so none needed removal. No native objective, weights, support, seed, cohort, ISP or deployment change.

Run20260914-151622-taisp-t030a2-candidates120, release20260914-151439-taisp-t030a2-candidates, source59efa11, fromrecord0 originalresearch_log/T030A/candidate_images.json. Exact saved command: python -m taisp.analysis.pseudo_native_candidates --manifest research_log/T030A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidate". Completed120episodes in{cc['seconds']}s. All120 support/target/RNG/freeze and all7same-cotangent gradient parity checks pass. Component-additivity diagnostic passes{integrity['candidate_component_additivity_diagnostic_pass_count']}/120; these diagnostics do not alter the candidate or stop it.

Candidate recordsSHA256 **{integrity['candidate_records_sha256']}**; candidate rawmanifestSHA256 **{integrity['candidate_manifest_sha256']}**. All120freshrecords preserved individually plus records.json. No failedR045record reused. Everyrecord includes supports, boxes/classes, allfour native losses, authoritative gradient, hard gradient, multi/component diagnostics and isolation; environment.json/run.sh provide code/model/data provenance.

## Tests and integrity

Baseline5passed5.15s (initial invocation referenced a nonexistent test filename and ranzero tests, corrected before code edits). Auditfocused15passed4warnings10.45s/full258passed11skipped4warnings22.76s. Correctedfocused16passed3warnings10.37s/full259passed11skipped4warnings22.60s. Existing NVML/protobuf warnings retained. Tests cover sharedgraph reuse, chain parity with exactcotangent, symmetric relative/zero semantics, exactGate1–4 boundaries, native scalar identity, non-gating decomposition logging, frozenstate/RNG restoration and no-reference import boundary. Model-bearing tests and both real runs onCUDA; local CPU report/hash work only.

{len(preserved)}protected/prior files unchanged; diagnostic49and corrected50remote hashes match. Both runs finaldetectorstate identical, no reference/oracle/CLIP import, GT_loadedfalse, APcalls0. Oldreports/receipts preserved. Task statusNEEDS_REVIEW: authorized numerical audit andcandidate lock complete; stopbeforeannotations/reference/AP/K-step. Futurefreshcohort exclusions remainT030A_train_cohort.json (3011reserved/prior IDs plus5000val).
'''
    (ROOT/'research_log/T030A2_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
