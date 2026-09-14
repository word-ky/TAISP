"""R048 report from immutable candidate/reference records; stdlib only."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_log/T030A3'
RAW=ROOT/'research_log/remote_runs/20260914-160040-taisp-t030a3-reference120/artifacts/reference'
CAND=ROOT/'research_log/remote_runs/20260914-151622-taisp-t030a2-candidates120/artifacts/candidate'
KEYS=('loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for f,h in read(RAW/'sha256.json').items():assert sha(RAW/f)==h,f
    lock=read(ROOT/'research_log/T030A3_candidate_lock.json')
    for f,h in lock['files'].items():assert sha(CAND/f)==h,f
    rows=read(RAW/'records.json');candidates=read(CAND/'records.json');s=read(RAW/'summary.json');completion=read(RAW/'completion.json')
    assert len(rows)==len(candidates)==120
    assert all((r['image_id'],r['case'],r['block'])==(c['image_id'],c['case'],c['block']) and r['candidate_hard_gradient']==c['objectives']['hard']['gradient'] and r['candidate_native_gradient']==c['objectives']['native']['gradient'] for r,c in zip(rows,candidates))
    preserved=read(ROOT/'research_log/T030A3_plan.json')['protected_and_prior_LF']
    checks={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in preserved.items()}
    remote=read(OUT/'remote_code_checks.json');assert all(checks.values()) and all(remote.values())
    integrity={'protected_prior':checks,'remote':remote,'all120_integrity':all(r['integrity_passed'] for r in rows),
        'all120_rng_restored':all(r['rng']['restored'] for r in rows),'all120_state_unchanged':all(r['state_before']==r['state_after']==completion['source_hash_after'] for r in rows),
        'candidate_vectors_exactly_unchanged':True,'candidate_hashes_unchanged':True,
        'reference_records_sha256':sha(RAW/'records.json'),'reference_manifest_sha256':sha(RAW/'sha256.json'),
        'candidate_records_sha256':sha(CAND/'records.json'),'preflight_sha256':sha(RAW/'preflight.json'),
        'reference_code_unchanged':sha(ROOT/'taisp/analysis/pseudo_native_reference.py'),
        'AP_calls':0,'candidate_recomputed':False}
    assert integrity['all120_integrity'] and integrity['all120_rng_restored'] and integrity['all120_state_unchanged']
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    for f in ['summary.json','preflight.json','completion.json']:(OUT/f).write_bytes((RAW/f).read_bytes())
    table=[]
    for r in rows:
        table.append({k:r[k] for k in ['episode_index','image_id','case','block','S_hard','S_native','Delta','norm_hard','norm_native','norm_ratio','gradient_cosine','integrity_passed']+[f'{prefix}_{k}' for k in KEYS for prefix in ['S','norm','cos_task']]})
    with (OUT/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');w.writeheader();w.writerows(table)
    groups={'overall':s['overall'],'clean':s['clean'],'corrupt':s['corrupt'],**s['conditions'],**{'block'+k:v for k,v in s['blocks'].items()}}
    group_lines=[];component_lines=[]
    for name,g in groups.items():
        d=g['distributions']
        group_lines.append(f"|{name}|{g['n']}|{d['S_hard']['positive']}|{d['S_native']['positive']}|{d['Delta']['positive']}|{d['Delta']['median']:.12g}|{d['Delta']['mean']:.12g}|{d['gradient_cosine']['median']}|{d['norm_ratio']['median']}|")
        for k in KEYS:component_lines.append(f"|{name}|{k}|{d['S_'+k]['positive']}|{d['S_'+k]['median']}|{d['cos_task_'+k]['median']}|{d['norm_'+k]['median']}|")
    a,b,c=[s[k]['distributions'] for k in ['overall','corrupt','clean']]
    blocks=[v['distributions']['Delta']['median'] for v in s['blocks'].values()]
    failure=[k for k,v in s['gates'].items() if not v]
    verdict='PASS' if s['passed'] else 'FAIL'
    report=f'''# T030-A3 / R048 — scientific {verdict}; NEEDS_REVIEW

The exact locked unweighted four-loss pseudo-native objective {'passes the original local source-gradient alignment gate; review before any further stage' if s['passed'] else 'fails the original R045 conjunction and is closed without rescue tuning'}. Failed gate flags: {failure}. This is a local source first-order gradient analysis; no AP, K-step or cross-detector result is claimed. R045/R046 retain their original BLOCKED numerical verdicts, while R047 retains numerical PASS.

## Locked inputs, implementation and execution

R048 research860f4d5 reviewed f9ca28e5ab56375c06e63c97700b97574acba775. Input-lock descriptor and original gates committed ea67513 before annotations; reference code/tests committed d480de9 before the model-bearing outcome. Candidate runner59efa11b99648c74d952918b3de3bebfce6df64e, stored run20260914-151622-taisp-t030a2-candidates120. Candidate recordsSHA256 {integrity['candidate_records_sha256']}; candidate rawmanifestSHA256 e5b019de2034f1697c5e630538869bf487674ca9cd06b10a3ed489bbdf6deac7. Original cohortSHA2563ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752.

Preflight verified all123candidate file hashes, 120individual/aggregate record equality and order, original cohort/corruption/JPEG manifests, reviewed/candidate commits, saved provenance, frozenstate, support/target equality andsupport hashes, finite8D gradients, saved RNG restoration and all7same-cotangent parity receipts. The preflight receipt is written before oracle/common_jacobian imports and before annotation load; oracle modules absent at that point. Candidate_all120_integrity true. The saved component-additivity diagnostic passes0/120 and is deliberately not a gate under R047/R048. No candidate/support/pseudo-target regeneration occurred; report independently compares allstored candidate vectors byte-equivalent as JSON values.

Promoted the preserved T030 reference draft into taisp.analysis.pseudo_native_reference; kept score/describe algebra and R045 thresholds, deferred oracle imports until lock verification, removed only the stale component-additivity prerequisite, and recorded per-reference RNG/state and exact consumed candidate vectors. All{len(checks)}protected/prior files unchanged, including candidate/deployment code and original drafts; all{len(remote)}remote hashes matched. No outcome-driven code change.

Reference run20260914-160040-taisp-t030a3-reference120, release20260914-155837-taisp-t030a3-reference, TAISP_SOURCE_REVISION=d480de9. A6000CUDA0, torch2.4.0+cu121, unchanged source-reference seed20260913 and native four-loss task definition. Exact command in rawrun.sh: python -m taisp.analysis.pseudo_native_reference --cohort research_log/T030A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-151622-taisp-t030a2-candidates120/artifacts/candidate --candidate-lock research_log/T030A3_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference".

Completed120episodes in{completion['seconds']}s. Frozen Faster R-CNN stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf; weightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384.

Annotation SHA256: {read(ROOT/'research_log/T030A_train_cohort.json')['annotation_sha256']}. Labels were used only to construct the unchanged reference task gradient after candidate lock verification. S=<task,g>/(norm(g)+1e-12); Delta=S_native-S_hard. Exact-zero candidates retained. No deterministic/CuBLAS/kernel/seed/dtype/model change.

## Original R045 scientific gate

|Criterion|Observed|Verdict|
|---|---|---|
|1: S_native>0 >=80/120|{a['S_native']['positive']}/120|{s['gates']['S_native_overall']}|
|2: S_native>0 >=45/60 corrupted|{b['S_native']['positive']}/60|{s['gates']['S_native_corrupt']}|
|3: Delta>0 >=68/120|{a['Delta']['positive']}/120|{s['gates']['Delta_overall']}|
|4: Delta>0 >=35/60 corrupted|{b['Delta']['positive']}/60|{s['gates']['Delta_corrupt']}|
|5: median Delta>0 overall and corrupted|{a['Delta']['median']}; {b['Delta']['median']}|{s['gates']['median_overall'] and s['gates']['median_corrupt']}|
|6: >=3/4 positive block medians|{sum(v>0 for v in blocks)}/4; {blocks}|{s['gates']['positive_blocks']}|
|7: clean median Delta>=0|{c['Delta']['median']}|{s['gates']['median_clean']}|
|8: all integrity|120/120; state/RNG/JVP/lock pass|{s['gates']['integrity']}|

## Overall, condition and block summaries

Counts are strictly positive, with zeros included in denominators. Cosines/norm ratios with zero denominators remain undefined rather than filtered from score distributions. Full means, medians, quantiles, zero and undefined counts are in summary.json.

|Group|N|S_hard positive|S_native positive|Delta positive|median Delta|mean Delta|median hard/native cosine|median native/hard norm ratio|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(group_lines)+'''

## Component diagnostics only

These component gradients are the stored diagnostic vectors from the immutable candidate lock. No component was recomputed, selected, weighted or promoted to a candidate.

|Group|Component|positive task score|median task score|median cosine to task|median norm|
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(component_lines)+f'''

## Tests, receipts and decision

Baseline11passed3warnings7.87s; focused12passed3warnings7.77s; full264passed11skipped4warnings25.03s, before annotated outcomes. Tests cover original gate conjunction/clean criterion/zero preservation, exact score epsilon, failed diagnostic versus mandatory RNG/nonfinite checks, and invalid lock stopping before oracle import. Existing NVML/protobuf warnings retained. Deployment155750SSH timeout preceded upload/modelrun; workflowstatus showed no job, deployment155837 succeeded. Only one annotated reference run was launched.

All120 references have frozen/eval/gradNone and restored RNG/state, accepted same-cotangent JVP/direct ISP parity. Candidate hashes still match the pre-GT lock after execution. Reference recordsSHA256 {integrity['reference_records_sha256']}; rawmanifestSHA256 {integrity['reference_manifest_sha256']}. Raw120individual receipts plus records.json, summary, preflight, environment, logs, per_episode.tsv and manifest retained local/server/GitHub. Model-bearing compute onA6000CUDA; offline report/hash work onCPU.

Disposition: {s['disposition']}. Stop NEEDS_REVIEW, no activejob. No AP/K-step/FCOS/SSD, no component/support/seed/K-LR/CLIP rescue. Futurefreshcohort exclusions remainT030A_train_cohort.json(3011reserved/prior IDs plus5000val).
'''
    (ROOT/'research_log/T030A3_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
