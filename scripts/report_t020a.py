"""Render complete R032 receipts; standard library only, no model/AP calls."""
import argparse
import hashlib
import json
import math
import statistics as st
import subprocess
from pathlib import Path

from report_t018a import CASES, table

METHODS = ('no_adapt','current_ours','grad_transport_ours')


def render(project, run, smoke):
    raw=project/'research_log/remote_runs'/run; root=raw/'artifacts/study'; fitroot=raw/'artifacts/source_fit'
    read=lambda p:json.loads(p.read_text())
    metrics,summary,diagnostics,completion=[read(root/f'{s}.json') for s in ('metrics','summary','diagnostics','completion')]
    collection,fits,alignment=[read(fitroot/f'{s}.json') for s in ('completion','fits','alignment')]
    assert completion['status']=='completed' and completion['images']==200 and completion['adaptive_samples']==2800 and completion['evaluations']==105
    assert collection['status']=='completed' and collection['pairs']==1400 and collection['folds']==4
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    pairs=[json.loads(s) for s in (fitroot/'gradient_pairs.jsonl').read_text().splitlines()]
    assert len(rows)==2800 and len(pairs)==1400
    assert all(all(r['isolation'].values()) for r in rows+pairs)
    paired={(r['image_id'],r['case']):r for r in pairs}
    current={(r['image_id'],r['case']):r for r in rows if r['method']=='current_ours'}
    candidates=[r for r in rows if r['method']=='grad_transport_ours']
    norm_errors=[]; source_runtime_deltas=[]
    for r in candidates:
        f=fits[r['transport_fold']]; key=(r['image_id'],r['case'])
        assert len(f['train_image_ids'])==150 and len(f['heldout_image_ids'])==50 and len(f['train_indices'])==1050
        assert set(f['train_image_ids']).isdisjoint(f['heldout_image_ids'])
        assert r['image_id'] in f['heldout_image_ids'] and all(pairs[i]['block']!=r['block'] for i in f['train_indices'])
        assert r['support']==current[key]['support']==paired[key]['support']
        assert len(r['diagnostics'])==4 and sum(d['update_applied'] for d in r['diagnostics'])==3
        assert r['diagnostics'][0]['phi']==[0.]*8
        if r['support_count']==0: assert r['phi3_norm']==0
        for d in r['diagnostics']:
            pn=math.sqrt(sum(v*v for v in d['detector_gradient'])); qn=math.sqrt(sum(v*v for v in d['transported_gradient']))
            assert math.isclose(pn,qn,rel_tol=1e-5,abs_tol=1e-8)
            norm_errors.append(abs(qn-pn)/(pn+1e-12))
        source_runtime_deltas.append(math.sqrt(sum((a-b)**2 for a,b in zip(paired[key]['pseudo_gradient'],current[key]['diagnostics'][0]['detector_gradient']))))
    pins=read(project/'research_log/T020A_method_pins.json'); checks={}
    for key in ('protected_modules','authorized_modified_or_new_modules_sha256_LF'):
        for path,expected in pins[key].items():
            actual=hashlib.sha256(subprocess.check_output(['git','show','0546b05:'+path],cwd=project)).hexdigest()
            checks[path]=actual==expected
    assert all(checks.values())
    audit={'all_episode_isolation':True,'image_fold_isolation':True,'paired_supports':len(candidates),
        'norm_checks':len(norm_errors),'max_relative_norm_error':max(norm_errors),
        'source_vs_runtime_identity_pseudo_max_absolute_l2_diagnostic':max(source_runtime_deltas),
        'code_hash_checks':checks,'zero_pseudo_pairs':collection['zero_pseudo'],'zero_task_pairs':collection['zero_task'],
        'empty_adaptive_episodes':sum(r['support_count']==0 for r in rows),
        'frozen_model_hashes':read(root/'isolation.json'),'collection_isolation':collection['isolation']}
    out=project/'research_log/T020A';out.mkdir(exist_ok=True)
    (out/'receipt_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    tables='# T020-A complete held-out tables\n\nAP and deltas use the 0–100 scale. Each image uses only its held-out fold matrix.\n\n'
    tables+=table('Official COCO metrics',['Group','Condition','Method','AP','AP50','AP75'],
        [[g,c,m]+[100*v[f'{c}_{m}'][k] for k in ('AP','AP50','AP75')] for g,v in metrics.items() for c in CASES for m in METHODS])
    tables+=table('Macro AP/AP50/AP75',['Group','Method','AP','AP50','AP75'],
        [[g,m]+[100*st.mean(v[f'{c}_{m}'][k] for c in CASES[:-1]) for k in ('AP','AP50','AP75')] for g,v in metrics.items() for m in METHODS])
    tables+=table('Condition deltas',['Group','Condition','Reference','AP','AP50','AP75'],
        [[g,c,ref]+[100*(v[f'{c}_grad_transport_ours'][k]-v[f'{c}_{ref}'][k]) for k in ('AP','AP50','AP75')]
         for g,v in metrics.items() for c in CASES for ref in METHODS[:2]])
    blocks=table('Held-out block deltas',['Block','AP delta current','AP delta raw','AP50 delta current','AP75 delta current'],
        [[g,v['candidate_minus_current'],v['candidate_minus_no_adapt'],v['additional_macro_deltas']['AP50'],v['additional_macro_deltas']['AP75']]
         for g,v in summary['groups'].items() if g!='aggregate'])
    tables+=blocks
    diagtable=table('Adaptation diagnostics',['Method/group','N','Mean phi','Median phi','Update fraction','Mean support','Median support','Mean adapt s','Median adapt s','Mean teacher-inclusive s','Peak GiB'],
        [[g,d['n'],d['phi3_norm']['mean'],d['phi3_norm']['median'],d['update_fraction'],d['support_count']['mean'],d['support_count']['median'],d['adapt_seconds']['mean'],d['adapt_seconds']['median'],d['deploy_seconds']['mean'],d['peak_allocated_bytes']['max']/1024**3] for g,d in diagnostics.items()])
    tables+=diagtable
    aligntable=table('Held-out identity gradient alignment',['Group','N','Defined','Undefined zero','Raw mean','Transport mean','Raw median','Transport median','Improved fraction defined'],
        [[g,d['episodes'],d['defined'],d['undefined_zero'],d['raw_cosine']['mean'],d['transported_cosine']['mean'],d['raw_cosine']['median'],d['transported_cosine']['median'],d['improved_fraction_defined']] for g,d in alignment['groups'].items()])
    tables+=aligntable
    tables+=table('Orthogonal fit receipts',['Fold','Train images','Train pairs','Zero pseudo','Zero task','Determinant','Max orthogonality error','Squared fit residual'],
        [[k,len(f['train_image_ids']),len(f['train_indices']),f['zero_pseudo_rows'],f['zero_task_rows'],f['determinant'],f['orthogonality_max_abs'],f['fit_squared_residual']] for k,f in fits.items()])
    tables+=table('Singular values',['Fold']+[str(i) for i in range(8)],[[k]+f['singular_values'] for k,f in fits.items()])
    tables+=table('Fold matrix similarity',['Fold A','Fold B','Frobenius distance','Frobenius cosine'],
        [[r[k] for k in ('fold_a','fold_b','frobenius_distance','frobenius_cosine')] for r in alignment['matrix_similarities']])
    for k,f in fits.items():tables+=table(f'Q fold {k} (row convention)',['Row']+[str(i) for i in range(8)],[[i]+r for i,r in enumerate(f['Q'])])
    (out/'complete_tables.md').write_text(tables,encoding='utf-8')
    files=[]
    for folder in (raw,project/'research_log/remote_runs'/smoke):
        for p in sorted(x for x in folder.rglob('*') if x.is_file()):
            b=p.read_bytes();files.append(dict(path=p.relative_to(project).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    manifest=dict(run=run,smoke_run=smoke,file_count=len(files),total_bytes=sum(f['bytes'] for f in files),files=files)
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    a=summary['groups']['aggregate'];gate=summary['gate'];smokeroot=project/'research_log/remote_runs'/smoke/'artifacts'
    report=f'''# T020-A — {gate['decision']}; NEEDS_REVIEW

R032/2654048. Plan/cohort precommit766ae89; experiment code0546b05.
Formal `{run}`; smoke `{smoke}`; release20260913-202732-taisp-t020a-crossfit.
200 new train2017 images, four image-level 50-image folds, exclude836prior-source/all5000val;
cohortSHA{pins['cohort_sha256']}, selectionseed20260921, model/oracle seed20260912.
For each held-out fold Q=U Vh from normalized training P^T T (150images/1050pairs).
All1400rawpairs/fourmatrices/singularvalues/trainindices retained. No determinant correction,
optimizer, ridge, hyperparameter search, per-condition matrix or post-outcome exclusions.
Source labels enter only the separate collection process. Held-out gt is diagnostic only
for its own fold. Runtime receives only matrix/provenance and the existing label-free
pseudo loss; gd @ Q then unchanged CLIP norm transfer. K3/LR.1/identity/score.5top20,
frozen source/CLIP/ISP operators/prompts/corruptions unchanged. Public current Ours is
unchanged by default; identity-Q equivalence tested. No target/val/cross-detector work.

## Exact performance result

Corruption macro AP: {json.dumps(a['macro_AP'])}.
Candidate minus current: **{a['candidate_minus_current']:+.9f} AP**; minus raw **{a['candidate_minus_no_adapt']:+.9f} AP**.
Positive blocks **{gate['positive_blocks']}/4**; positive conditions **{a['positive_conditions']}/6**.
Clean delta current **{a['clean_delta']:+.9f} AP**, clean delta raw **{a['clean_delta_vs_raw']:+.9f} AP**.
Macro AP50/AP75 deltas current: {json.dumps(a['additional_macro_deltas'])}.
Frozen decision passed: **{gate['passed']}**. Alignment is diagnostic and does not override AP criteria.

'''
    report+=table('R032 frozen criteria',['Criterion','Passed'],[[k,v] for k,v in gate['flags'].items()])+blocks+aligntable+diagtable
    report+=f'''## Validation, timing and artifacts

Baseline10passed2skipped1.79s; increment1 12passed1skipped1.69s.
Focused: {(smokeroot/'focused_tests.txt').read_text().strip()}
Full regression: {(smokeroot/'full_tests.txt').read_text().strip()}
Two-image smoke:14sourcepairs,twoleave-one-image-outmaps,28adaptiveK3episodes,0AP.
Formal:1400sourcepairs,2800adaptiveepisodes,21predictionfiles,105officialevaluations.
GPU collection {collection['collection_seconds']:.6f}s; CUDA float64 fit {collection['fit_seconds']:.6f}s;
CUDA float32 runtime collection/officialCPUevaluation {completion['elapsed_seconds']:.6f}s.
All source/runtime isolation and code pins pass. {len(norm_errors)}runtime norm checks pass,
max relative norm error {max(norm_errors):.12g}; paired teacher supports match all1400episodes.
Zero pseudo/task pairs: {collection['zero_pseudo']}/{collection['zero_task']}; retained in fitting
because EPS normalization is defined. Zero-vector alignment is null and reported explicitly.
Empty adaptive episodes {audit['empty_adaptive_episodes']}; exact identity updates verified.
Source-vs-runtime identity pseudo maximum absolute L2 diagnostic {max(source_runtime_deltas):.12g}.
Adaptation latency includes terminal gradient diagnostics (four evaluations,threeupdates),
excludes source fitting, final prediction, state verification and AP. Teacher-inclusive adds
shared teacher setup. Peak memory includes frozen-state verification copies.

{len(files)}rawfiles, {manifest['total_bytes']}bytes; SHA256 manifest inT020A/artifact_manifest.json.
All AP/AP50/AP75 tables,clean/block/condition deltas,Q/singularvalues/matrixsimilarity and
diagnostics: T020A/complete_tables.md. Isolation/norm/hash audit:T020A/receipt_audit.json.
No tuning or extra cohort was run. StopNEEDS_REVIEW. A pass permits only a developmental
candidate; a failure closes the fixed global linear gradient-transport branch underR032.
'''
    (project/'research_log/T020A_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(gate=gate,aggregate=a,audit=audit,raw_files=len(files),raw_bytes=manifest['total_bytes']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--run',required=True);p.add_argument('--smoke',required=True)
    a=p.parse_args();render(a.project,a.run,a.smoke)
