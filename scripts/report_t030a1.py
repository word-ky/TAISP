"""T030-A1 numerical attribution report from frozen zero-GT receipts."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];out=root/'research_log/T030A1'
    raw=root/'research_log/remote_runs/20260914-133102-taisp-t030a1-attribution8x5/artifacts/diagnostic'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    tensors={Path(v['archive_member']).name:v['sha256'] for v in read(out/'tensor_storage_index.json')['tensors']}
    for f,h in read(raw/'sha256.json').items():assert (tensors[f] if f.endswith('.pt') else sha(raw/f))==h
    s=read(raw/'summary.json');probe=read(raw/'deterministic_probe.json');completion=read(raw/'completion.json')
    eps=[read(raw/f'episode_{i:02d}.json') for i in range(8)]
    reps=[r for e in eps for r in e['repetitions']]
    table=[]
    for e in eps:
        for r in e['repetitions']:
            table.append({'episode':e['episode_index'],'image_id':e['image_id'],'case':e['case'],'repetition':r['repetition'],
                **{f'{space}_{k}_{metric}':r[space][k][metric] for space in ['projected','image_space'] for k in ['multi','sep','sep_rev'] for metric in ['relative_l2','cosine']},
                'direct_phi_rel':r['direct_phi']['relative_l2'],'direct_phi_cosine':r['direct_phi']['cosine'],
                'reverse_order_relative':r['reverse_order_relative'],'image_reverse_order_relative':r['image_reverse_order_relative'],
                'old_additivity_pass':r['old_additivity']['passed'],'old_max_error_over_bound':r['old_additivity']['max_error_over_bound'],
                'rng_restored':r['rng']['restored'],'isolation':r['isolation'],'target_unchanged':r['target_unchanged'],
                'cotangent_file':r['cotangent_file'],'cotangent_sha256':r['cotangent_sha256']})
    with (out/'per_repetition.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');w.writeheader();w.writerows(table)
    (out/'summary.json').write_bytes((raw/'summary.json').read_bytes())
    plan=read(root/'research_log/T030A1_plan.json')
    checks={f:hashlib.sha256((root/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in plan['protected_and_prior_LF'].items()}
    remote=read(out/'remote_code_checks.json');assert all(checks.values()) and all(remote.values())
    integrity={'local_protected_prior':checks,'remote':remote,'all40_integrity':all(r['integrity_passed'] for r in reps),
        'all40_state_hashes':all(r['state_before']==r['state_after']==completion['source_hash_after'] for r in reps),
        'direct_phi_relative_pass_repetitions':sum(r['direct_phi']['relative_l2']<=1e-5 for r in reps),
        'direct_phi_cosine_pass_repetitions':sum(r['direct_phi']['cosine']>=.999999 for r in reps),
        'old_additivity_pass_repetitions':sum(r['old_additivity']['passed'] for r in reps),
        'all_probe_restored':probe['restored'] and probe['frozen_state_unchanged'] and probe['rng_restored'],
        'full_raw_manifest_sha256':sha(raw/'sha256.json')}
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    rows=s['episodes'];lines=[]
    for r in rows:
        m=r['median_projected_relative_l2']
        lines.append(f"|{r['episode_index']} / {r['image_id']} / {r['case']}|{r['native']['max_pairwise_relative_l2']:.9g}|{r['native']['min_pairwise_cosine']:.12g}|{r['direct_phi_max_relative_l2']:.9g}|{m['multi']:.9g}|{m['sep']:.9g}|{m['sep_rev']:.9g}|{r['multi_no_worse']}|")
    image_lines=[]
    for r in rows:
        m=r['median_image_relative_l2']
        image_lines.append(f"|{r['episode_index']}|{m['multi']:.9g}|{m['sep']:.9g}|{m['sep_rev']:.9g}|{r['hard']['max_pairwise_relative_l2']:.9g}|{r['hard']['min_pairwise_cosine']:.12g}|")
    report=f'''# T030-A1 — BLOCKED; replacement integrity gate failed

R046/27fed9c completed the bounded8episode x5repetition numerical audit. Do not correct the T030-A stop assertion, rerun120candidates, loadannotations or claim scientific utility. Conditional continuation is NOT authorized because direct-phi parity and multi-output comparison fail. OriginalT030-A failure remains unchanged.

## Frozen scope and execution

Plan/manifest9ce9b5e before outcomes; diagnostic codea2844cd; unchanged firstfourT030images304815,131976,507312,517967 with the original clean/assignedcorruption pairs, no newselection. OriginalcohortSHA3ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752; diagnosticmanifestSHA{sha(out/'diagnostic_images.json')}. Native samplingseed20260930; source/environmentsetup20260913; score>=.50/stabletop20, one detachedsupport generation perepisode shared by all5reps.

Run20260914-133102-taisp-t030a1-attribution8x5, release20260914-132745-taisp-t030a1-attribution, actualTAISP_SOURCE_REVISION=a2844cd, A6000CUDA0/torch2.4.0+cu121; all40reps completed in{completion['seconds']}s, exit0 with BLOCKED scientific-process status. Initiallauncher20260914-132943 leftmeta/run.sh only afterSSHfailure; status showedno tmux/process/train.log before retry. Bothlauncherreceipts retained; no repeated outcome run.

Eachrep uses a freshnativeforward graph: scalar-sumVJP; one engine call onfouroutputs/unitgradoutputs; fourcanonical reversecalls; fourreversed reversecalls. Separate imagecotangents are summed in float64; allfourcontract against the sameaccepted8columnfloat32ISPJVP/float64reductions. Then a freshfixedseednativeforward differentiates directly through ISP tophi, plus one freshunchangedhardgradient control. No cotangent shared acrossreps and no averagedcandidate gradient.

Allfourimagecotangents saved perrep in40torchfiles (sum/multi float32, separate/reverse float64); raw8Dvectors, canonical/reversecomponentvectors, losses, target/support hashes, before/afterRNG hashes and modelstate hashes retained. Oldpercoordadditivity check remains diagnostic only; passed{integrity['old_additivity_pass_repetitions']}/40 here and original failedrun was not relabeled.

## Exact replacement gate

|Criterion|Observed|Decision|
|---|---|---|
|Native repeat cosine>=.99999 on8/8|min={min(r['native']['min_pairwise_cosine'] for r in rows)}|PASS|
|Native maxpairrel<=.002 on8/8|max={max(r['native']['max_pairwise_relative_l2'] for r in rows)}|PASS|
|Freshdirectphi rel<=1e-5 andcos>=.999999 on8/8|relativepasses {integrity['direct_phi_relative_pass_repetitions']}/40 reps; cosinepasses {integrity['direct_phi_cosine_pass_repetitions']}/40|FAIL|
|Allsupport/RNG/state/import/finite checks|all40pass|PASS|
|Medianmulti no worse than bothseparate orders on>=7/8|{sum(r['multi_no_worse'] for r in rows)}/8|FAIL|
|Eachmedianmulti<=.002|max={max(r['median_projected_relative_l2']['multi'] for r in rows)}|PASS|

First failed replacement item is direct-phi/common-JVP parity (item3); everyepisode's worst fresh-forward relativeL2 exceeds1e-5. Item5 also fails. No new tolerance, seed, weighting or criterion was introduced after outcomes.

## 8-D comparison, all episodes

Each discrepancy is relative to g_sum; episode columns are medianover5except repeatability extrema and maximumdirecterror. Pairwise dispersion is maximum over allordered distinctpairs, denominator max(norm(first),1e-12), as precommitted.

|Episode / image / condition|d_native|min cosine native|max directphi rel|median multi|median separate|median reverse|multi no worse|
|---|---:|---:|---:|---:|---:|---:|---|
'''+'\n'.join(lines)+'''

## Image-cotangent localization and hard control

Image discrepancies already occur before ISP contraction. These are median relativeL2 to c_sum; hardcontrol statistics use the samefivefreshreps.

|Episode|image multi|image separate|image reverse|d_hard|min cosine hard|
|---|---:|---:|---:|---:|---:|
'''+'\n'.join(image_lines)+f'''

Reversing separate-backward order changes the projectedvector bymorethan1e-5relative tosum in{sum(r['reverse_order_material_reps'] for r in rows)}/40reps (predeclared diagnostic threshold, not a gate). Fullimage-space order differences and all40perrep comparisons inper_repetition.tsv; fullprecisionepisode metrics insummary.json. Hardcontrol variability is nonuniform and is not used to rescue nativefailure.

## Deterministic probe and interpretation limits

Exactlyone episode0 attempt afteritsfive ordinaryreps with deterministic_algorithms=True failed withRuntimeError: CuBLAS requires a process-start CUBLAS_WORKSPACE_CONFIG for deterministic execution. Fullerror retained indeterministic_probe.json. OriginalFalse setting and warn_only=False restored immediately; RNG andsourcehashunchanged. Noenvironmentvariable, driver, kernel, seed or deployment setting changed. Probe is non-gating.

The receipts support upstream image-gradient variability across reverse sweeps and independently generated nativeforward graphs. The scalar-sum candidate met the separately frozen0.2%repeatability threshold, but did not meet the muchtighter fresh-forward directphi equality bound, and multi-output was not consistently better in8D. Therefore this audit does not establish that removing only the oldcomponentadditivity stop condition is justified. It does not isolate a single responsibleCUDAoperator or establish source-task utility.

## Integrity, tests and handoff

All40state_before/state_after hashes equal73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf; detectorfrozen/eval/gradNone, ISPstateidentity, target/RNGrestore andtransitiveimportchecks pass. All{len(checks)}protected/prior files unchanged locally and{len(remote)}remote hashes match. No annotation JSON orreference taskgradient; APcalls0. Numericalgate failure does not negate these separate isolation checks.

Focused16pass7.94s;full254pass11skip4warnings19.80s. ExistingNVML/protobuf warnings retained. AllmodelcomputeA6000CUDA; CPU onlyserialization/manifest/report math. SHA-verified rawarchive and everyfile verified, including40cotangent tensors retained inside the local project archive and expanded server run directory. GitHub contains all JSON/TSV/log/code receipts and the tensor storage/hash index; the 40large .pt tensors remain in the local archive and server project directory rather than Git objects. report/TSV/manifest are reproduciblefromthese frozen outputs.

Stop BLOCKED. No full120candidate rerun or correctedassertioncommit because R046conditions failed. Preserve T03060reservedcohort and originalfailedrecord. Awaitexplicitresearchdecision; no objective/support/component/seed/bounds/averaging/deterministic-mode/GT/AP change. Futurefreshcohort exclusions remainT030A_train_cohort.json(3011reserved/prior IDs plus5000val).
'''
    (root/'research_log/T030A1_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
