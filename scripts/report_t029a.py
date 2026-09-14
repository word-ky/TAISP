"""Render T029-A report from preserved candidate and reference receipts."""
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];out=root/'research_log/T029A'
    croot=root/'research_log/remote_runs/20260914-113829-taisp-t029a-candidates120/artifacts/candidates'
    rroot=root/'research_log/remote_runs/20260914-114328-taisp-t029a-reference120/artifacts/reference'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in [croot,rroot]:
        for f,h in read(folder/'sha256.json').items():assert sha(folder/f)==h
    cs,rs=read(croot/'records.json'),read(rroot/'records.json');s=read(rroot/'summary.json')
    table=[]
    for c,r in zip(cs,rs,strict=True):
        assert (c['image_id'],c['case'])==(r['image_id'],r['case'])
        for name in ['hard','soft']:
            g=c['objectives'][name]['gradient']
            expected=math.fsum(a*b for a,b in zip(r['task_gradient'],g))/(math.sqrt(math.fsum(v*v for v in g))+1e-12)
            assert expected==r['S_'+name]
        table.append({k:r[k] for k in ['episode_index','image_id','case','block','S_hard','S_soft','Delta','norm_hard','norm_soft','norm_ratio','gradient_cosine','integrity_passed']} |
            {'support_count':c['support_count'],'hard_loss':c['objectives']['hard']['loss'],'soft_loss':c['objectives']['soft']['loss']})
    with (out/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');w.writeheader();w.writerows(table)
    (out/'summary.json').write_bytes((rroot/'summary.json').read_bytes())
    plan=read(root/'research_log/T029A_plan.json')
    checks={f:hashlib.sha256((root/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in plan['protected_and_prior_LF'].items()}
    remote=read(out/'remote_code_checks.json');assert all(checks.values()) and all(remote.values())
    integrity={'local_protected_and_prior':checks,'remote':remote,
        'all120_reference_integrity':all(r['integrity_passed'] for r in rs),
        'all120_candidate_isolation_support_target':all(c['isolation'] and c['support_match'] and c['target_valid'] for c in cs),
        'max_hard_parity_relative_l2':max(c['objectives']['hard']['parity']['relative_l2'] for c in cs),
        'max_soft_parity_relative_l2':max(c['objectives']['soft']['parity']['relative_l2'] for c in cs),
        'max_reference_parity_relative_l2':max(r['reference_checks']['parity']['relative_l2_error'] for r in rs),
        'max_partition_error':max(r['reference_checks']['partition']['max_absolute_error'] for r in rs),
        'empty_supports':sum(c['support_count']==0 for c in cs),
        'cohort_sha256':sha(root/'research_log/T029A_train_cohort.json'),
        'candidate_records_sha256':sha(croot/'records.json'),'reference_records_sha256':sha(rroot/'records.json')}
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    a,b,clean=[s[k]['distributions'] for k in ['overall','corrupt','clean']]
    blocks={k:v['distributions']['Delta']['median'] for k,v in s['blocks'].items()}
    report=f'''# T029-A — NEEDS_REVIEW; exact power-2 soft pseudo FAIL

R044/023f58b executed. Close the exact alpha=2 soft-sharpened target family under the frozen conjunction. No runtime replacement, CLIP norm transfer, K-step/AP, FCOS/SSD, target/validation evaluation or tuning.

## Cohort and outcome ordering

Plan80319b1 precommitted60fresh train2017 images, seed20260929, excluding2891cumulative source/memory/audit IDs throughT028 and5000val IDs. Existing GT-valid noncrowd/readable-image eligibility is used only during outcome-free cohort preparation. Selection sorts sha256('20260929:'+ID);60clean plus20each gamma_s2/contrast_s2/color_cast_s2. Four15imageblocks each contain5percorruption, with pairs kept together. No outcome-based replacement. Cohort SHA256 {integrity['cohort_sha256']}.

Candidate codea2bc921; all120candidate receipts committed/pushed4ec7df4 BEFORE reference code8ef7b3e and annotation load. Candidate receives only image manifest/JPEGs; no oracle/reference/source-meta/CLIP/memory imports transitively. Separate reference verifies123candidate files and cohortSHA before opening annotation JSON. Actual source revisions a2bc921 and8ef7b3e are present in the respective raw run metadata; no placeholder or provenance repair.

## Literal objective and gradients

One original-view teacher, unchanged score>=.50/stable descendingtop20 selection. Hard baseline is unmodified DetectorNativeLoss(det_pseudo). The new analysis-only soft objective directly shares hard.boxes and hard.weights (originalscores/scores.sum), so support ordering, boxes and confidence weights are identical. Raw support indices/boxes/labels/scores/hash retained.

Full91-way original-image fixed-ROI logits z0; q=softmax(2*log_softmax(z0)), detached, includingbackground. Loss is sum_i w_i[-sum_c q_ic logsoftmax(z(phi))_ic], natural logs. Empty support returns enhanced.sum()*0 for both. No foreground truncation, extra confidence weighting, new box loss, flip, memory, CLIP, spatial state or auxiliary loss.

Both identity image-cotangents contract with the same accepted8-column global ISP Jacobian (float32 ISP/model, float64 reductions); direct reverse/JVP parity retained. Post-lock source reference is unchanged original-view four-native-loss sum at seed20260913. S_hard/soft=dot(t_s,g)/(norm(g)+1e-12); Delta=S_soft-S_hard. Exactzero gradients are retained. Gradient cosine and soft/hard norm ratio are null only when the denominator is zero, with undefined counts explicit.

## Frozen gate

|Metric|Overall120|Corrupt60|Required|
|---|---:|---:|---|
|S_soft>0|{a['S_soft']['positive']}|{b['S_soft']['positive']}|>=80 / >=45|
|Delta>0|{a['Delta']['positive']}|{b['Delta']['positive']}|>=68 / >=35|
|Median Delta|{a['Delta']['median']}|{b['Delta']['median']}|>0 both|
|Mean Delta (diagnostic)|{a['Delta']['mean']}|{b['Delta']['mean']}|not a gate|

Positive block medians {sum(v>0 for v in blocks.values())}/4 (required>=3); exact values {blocks}. Clean medianDelta={clean['Delta']['median']} (required>=0). Integrity passes; full conjunction FAIL. Positive delta counts/medians do not rescue insufficient positive task utility and block consistency. Hard positive utility counts={a['S_hard']['positive']}/120overall,{b['S_hard']['positive']}/60corrupted.

All overall/clean/corrupt/family/block counts, fractions, means, medians and linear quantiles[0,.25,.5,.75,1] are retained for S_hard/S_soft/Delta/norms/ratio/cosine in summary.json; per_episode.tsv retains all120episodes. Overall/corrupted median hard-softcosine={a['gradient_cosine']['median']}/{b['gradient_cosine']['median']}; median normratio={a['norm_ratio']['median']}/{b['norm_ratio']['median']}. Family heterogeneity is diagnostic only; no bootstrap or new significance criterion.

## Integrity, tests and execution

All120candidate support/target/isolation and reference checks pass. Zero hard/soft={s['overall']['zero_hard']}/{s['overall']['zero_soft']}; empty supports={integrity['empty_supports']}. Max hard/soft/reference reverse-JVP relativeL2={integrity['max_hard_parity_relative_l2']}/{integrity['max_soft_parity_relative_l2']}/{integrity['max_reference_parity_relative_l2']}; partition maxerror={integrity['max_partition_error']}. {len(checks)}protected/prior files unchanged locally, all{len(remote)}protected/prior/new hashes match remote release.

Detector remains frozen/eval/gradNone, state73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf before/after both jobs. ISP unchanged/identity/gradNone outside differentiated temporary state. Model work on A6000 CUDA0, torch2.4.0+cu121; CPU only manifests and report aggregation.

Tests: baseline9pass2.92s; candidatefocused10pass5.62s; finalfocused12pass5.84s; full243pass11skip4warnings15.43s. Initial test expectation used literal1.4 instead of the existing float32 scores.sum(), producing an assertion mismatch; corrected only that test expectation, then passed. No objective or tolerance change. ExistingNVML/protobuf warnings retained. One SCP download timedout and succeeded via existing workflow legacySCP retry; no experiment restart.

Candidate run20260914-113829-taisp-t029a-candidates120, release20260914-113707-taisp-t029a-candidate-tested, {read(croot/'completion.json')['seconds']}s. Reference20260914-114328-taisp-t029a-reference120, release20260914-114121-taisp-t029a-reference, {read(rroot/'completion.json')['seconds']}s. Both exit0. Candidate archiveSHA be209bdaf44a5c429c5b8b871a68e86dad5c3344bc0bcd5784dab776e5288800; reference b60f287c79fc10fe53144b955062b495ad97a1fab171cdc9a476de2da484ea31, verified before extraction plus per-fileSHA checks.

Candidate recordsSHA {integrity['candidate_records_sha256']}; reference recordsSHA {integrity['reference_records_sha256']}. Original logits/fullq, supports/weights, hard/soft losses/gradients, source references and all environment/integrity receipts preserved.

Stop NEEDS_REVIEW. No alpha/temperature/classmass/foreground/weights/support/K/LR/hard+soft rescue or T029-B without explicit new research task. Future fresh cohorts use T029A_train_cohort.json as additional_source:2891prior+60new=2951source IDs, plus5000val exclusion.
'''
    (root/'research_log/T029A_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
