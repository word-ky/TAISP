"""R033 complete receipt report; offline standard-library aggregation only."""
import argparse
import hashlib
import json
import math
import statistics as st
import subprocess
from pathlib import Path

from report_t018a import CASES, table

METHODS=('no_adapt','current_ours','flip_consensus_ours')


def audit_supports(rows):
    current={(r['image_id'],r['case']):r for r in rows if r['method']=='current_ours'}
    candidates=[r for r in rows if r['method']=='flip_consensus_ours']
    retained=0
    for r in rows:
        assert all(r['isolation'].values())
        assert len(r['diagnostics'])==4 and sum(d['update_applied'] for d in r['diagnostics'])==3
        assert r['diagnostics'][0]['phi']==[0.]*8 and r['native_components'] is None
        assert all('transported_gradient' not in d for d in r['diagnostics'])
        if not r['support_count']:assert r['phi3_norm']==0
    for r in candidates:
        c=r['consensus']; a=c['original_eligible'];b=c['flip_eligible'];oi=c['original_indices'];fi=c['flip_indices']
        assert c['retained_pairs']==c['matches'][:20] and c['retained_count']==len(c['retained_pairs'])==r['support_count']
        assert c['matched_count']==len(c['matches']) and c['original_eligible_count']==len(oi)
        assert c['flip_eligible_count']==len(fi) and all(s>=.5 for s in a['scores']+b['scores'])
        assert len({m['original_index'] for m in c['matches']})==len(c['matches'])
        assert len({m['flip_index'] for m in c['matches']})==len(c['matches'])
        assert c['matches']==sorted(c['matches'],key=lambda m:(-m['geometric_confidence'],m['original_index'],m['flip_index']))
        for pair in c['matches']:
            i,j=oi.index(pair['original_index']),fi.index(pair['flip_index'])
            assert a['labels'][i]==b['labels'][j] and pair['iou']>=.6
            assert pair['geometric_confidence']==math.sqrt(a['scores'][i]*b['scores'][j])
        indices=[oi.index(pair['original_index']) for pair in c['retained_pairs']]
        assert r['support']=={k:[v[i] for i in indices] for k,v in a.items()}
        baseline=current[r['image_id'],r['case']]
        base_indices=sorted((i for i in range(len(oi)) if a['labels'][i]>0),key=lambda i:(-a['scores'][i],oi[i]))[:20]
        assert baseline['support']=={k:[v[i] for i in base_indices] for k,v in a.items()}
        assert c['current_top20_count']==baseline['support_count']
        assert c['retention_fraction']==(len(indices)/len(oi) if oi else None)
        assert all(d['support_count']==len(indices) for d in r['diagnostics'])
        assert math.isclose(r['deploy_seconds'],r['adapt_seconds']+r['teacher_seconds']+r['consensus_setup_seconds'],rel_tol=1e-12)
        retained+=len(indices)
    return dict(adaptive_episodes=len(rows),consensus_episodes=len(candidates),retained_supports_verified=retained,
                all_support_matching_receipt_checks=True,all_original_scores_boxes_classes_preserved=True,
                all_baseline_supports_unchanged=True,all_episode_isolation=True,
                empty_candidate_episodes=sum(r['support_count']==0 for r in candidates),
                empty_current_episodes=sum(r['support_count']==0 for r in current.values()))


def render(project,run,smoke):
    raw=project/'research_log/remote_runs'/run;root=raw/'artifacts/study'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    metrics,summary,diagnostics,supports,completion=[read(root/f'{k}.json') for k in ('metrics','summary','diagnostics','support_diagnostics','completion')]
    assert completion['status']=='completed' and completion['images']==200 and completion['adaptive_samples']==2800
    assert completion['teacher_forwards']==2800 and completion['evaluations']==105
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    audit=audit_supports(rows)
    pins=read(project/'research_log/T021A_method_pins.json');hashes={}
    for group in ('protected_modules','authorized_modified_or_new_modules_sha256_LF'):
        for f,h in pins[group].items():
            hashes[f]=hashlib.sha256(subprocess.check_output(['git','show','daa79e5:'+f],cwd=project)).hexdigest()==h
    assert all(hashes.values());audit['code_hash_checks']=hashes
    audit['frozen_model_hashes']=read(root/'isolation.json')
    out=project/'research_log/T021A';out.mkdir(exist_ok=True)
    (out/'receipt_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    tables='# T021-A complete fixed-protocol tables\n\nAP and deltas use the0-100scale; all blocks/conditions/methods retained.\n\n'
    tables+=table('Official AP/AP50/AP75',['Group','Condition','Method','AP','AP50','AP75'],
        [[g,c,m]+[100*v[f'{c}_{m}'][k] for k in ('AP','AP50','AP75')] for g,v in metrics.items() for c in CASES for m in METHODS])
    tables+=table('Corruption macro metrics',['Group','Method','AP','AP50','AP75'],
        [[g,m]+[100*st.mean(v[f'{c}_{m}'][k] for c in CASES[:-1]) for k in ('AP','AP50','AP75')] for g,v in metrics.items() for m in METHODS])
    tables+=table('Condition deltas',['Group','Condition','Reference','AP','AP50','AP75'],
        [[g,c,ref]+[100*(v[f'{c}_flip_consensus_ours'][k]-v[f'{c}_{ref}'][k]) for k in ('AP','AP50','AP75')]
         for g,v in metrics.items() for c in CASES for ref in METHODS[:2]])
    blocktable=table('Fixed block macro deltas',['Block','AP-current','AP-raw','AP50-current','AP75-current'],
        [[g,v['candidate_minus_current'],v['candidate_minus_no_adapt'],v['additional_macro_deltas']['AP50'],v['additional_macro_deltas']['AP75']]
         for g,v in summary['groups'].items() if g!='aggregate'])
    tables+=blocktable
    supporttable=table('Consensus support retention',['Group','N','Mean original eligible','Mean current top20','Mean matched','Mean retained top20','Mean retention defined','Median retention defined','Undefined retention','Zero count','Zero fraction','Extra flip+match s'],
        [[g,d['episodes'],d['original_eligible_count']['mean'],d['current_top20_count']['mean'],d['matched_count']['mean'],d['retained_count']['mean'],d['mean_retention_fraction_defined'],d['median_retention_fraction_defined'],d['undefined_retention_count'],d['zero_consensus_count'],d['zero_consensus_fraction'],d['consensus_setup_seconds']['mean']] for g,d in supports.items()])
    tables+=supporttable
    diagtable=table('Adaptation diagnostics',['Method/group','N','Mean phi','Median phi','Update fraction','Mean support','Median support','Mean adapt s','Median adapt s','Mean teacher-inclusive s','Peak GiB'],
        [[g,d['n'],d['phi3_norm']['mean'],d['phi3_norm']['median'],d['update_fraction'],d['support_count']['mean'],d['support_count']['median'],d['adapt_seconds']['mean'],d['adapt_seconds']['median'],d['deploy_seconds']['mean'],d['peak_allocated_bytes']['max']/1024**3] for g,d in diagnostics.items()])
    tables+=diagtable
    step_rows=[]
    for m in METHODS[1:]:
        for group in ('clean','corrupted'):
            chosen=[r for r in rows if r['method']==m and (r['case']=='clean_s0')==(group=='clean')]
            for step in range(4):
                ds=[r['diagnostics'][step] for r in chosen]
                step_rows.append([m,group,step,st.mean(math.sqrt(sum(x*x for x in d['phi'])) for d in ds)]+
                                 [st.mean(d[k] for d in ds) for k in ('total','detector_gradient_norm','clip_gradient_norm','gradient_norm')])
    tables+=table('Step diagnostics',['Method','Group','Step','Mean phi norm','Pseudo loss','Pseudo grad norm','CLIP grad norm','Update norm'],step_rows)
    tables+='Retention denominator is the original score>=.5 count before top20. Current top20 count is separately shown; consensus matching precedes its own top20. Original confidence weights are retained.\n'
    (out/'complete_tables.md').write_text(tables,encoding='utf-8')
    files=[]
    for folder in (raw,project/'research_log/remote_runs'/smoke):
        for p in sorted(x for x in folder.rglob('*') if x.is_file()):
            b=p.read_bytes();files.append(dict(path=p.relative_to(project).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    manifest=dict(run=run,smoke_run=smoke,file_count=len(files),total_bytes=sum(f['bytes'] for f in files),files=files)
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    a=summary['groups']['aggregate'];gate=summary['gate'];sr=project/'research_log/remote_runs'/smoke/'artifacts'
    test_result=lambda name:next(l for l in reversed((sr/name).read_text(encoding='utf-8').splitlines()) if ' passed' in l)
    report=f'''# T021-A - {gate['decision']}; NEEDS_REVIEW

R033/a5bae26; pre-outcome plan0207885; experimental code daa79e5.
Formal `{run}`; smoke `{smoke}`; exact commands/environment in raw receipts.
200newtrain2017 images/four50blocks,1036prior-source/debug IDs/all5000val excluded.
CohortSHA{pins['cohort_sha256']}; selectionseed20260922, unchanged modelseed20260912.
No selection by predictions/retention. No source labels or fitting enter adaptation.

Frozen rule: original and horizontal flip teacher,score>=.50 each view,sameclass IoU>=.60,
one-to-one greedy geometric confidence descending, original/flip index ties,top20after
matching. Original boxes/classes/**scores** retained; geometric mean is ranking only.
No box/score averaging or rematching. Empty consensus means exact identity/no fallback.
Unchanged current-Ours pseudo loss,global8D ISP/identity,K3/LR.1,CLIP magnitude rule,
frozen weights/prompts/teacher/corruptions. Final inference is a single enhanced view.

Corruption macro AP: {json.dumps(a['macro_AP'])}.
Candidate-current **{a['candidate_minus_current']:+.9f} AP**, candidate-raw **{a['candidate_minus_no_adapt']:+.9f} AP**.
Positive blocks **{gate['positive_blocks']}/4**, positive conditions **{a['positive_conditions']}/6**.
Clean-current **{a['clean_delta']:+.9f} AP**, clean-raw **{a['clean_delta_vs_raw']:+.9f} AP**.
Macro AP50/AP75 deltas versus current: {json.dumps(a['additional_macro_deltas'])}.
Frozen decision passed: **{gate['passed']}**. Retention/AP50/AP75 never replace AP gates.

'''
    report+=table('Frozen criteria',['Criterion','Passed'],[[k,v] for k,v in gate['flags'].items()])+blocktable+supporttable+diagtable
    report+=f'''## Validation and artifacts

Baseline12passed2skipped1.77s; increment1 11passed2skipped2.42s.
Focused: {test_result('focused_tests.txt')}. Full regression: {test_result('full_tests.txt')}.
Two-image CUDA smoke:28teacherforwards/28adaptiveK3episodes/zeroAP; all support checks pass.
Formal:2800teacherforwards/2800adaptiveepisodes/21predictions/105officialCOCOevaluations,
{completion['elapsed_seconds']:.6f}s, A6000 CUDA models/gradients/adaptation; CPU official AP.
All{audit['consensus_episodes']}consensus receipt checks and{audit['retained_supports_verified']}
retained supports preserve original boxes/classes/scores, ranking/ties/one-to-one/IoU.
All baseline supports reproduce original top20. Frozen state/hash/gradNone/ISP/support
checks pass. Empty candidate/current episodes:{audit['empty_candidate_episodes']}/{audit['empty_current_episodes']}.
All step histories retained. No native loss components or transported gradients appear.

Adaptation latency includes four diagnostic evaluations/threeupdates, excludes final
prediction/state/AP. Candidate teacher-inclusive includes original plus extra flip/matching;
current includes original teacher only. Extra overhead is separately reported above.
Peak memory includes verification copies. Support retention divides retained top20 by
original eligible pre-top20 count; a zero denominator is null and explicitly counted.

{len(files)}rawfiles,{manifest['total_bytes']}bytes,all SHA256 recorded inT021A/artifact_manifest.json.
Full AP/AP50/AP75/condition/block/step/support tables:T021A/complete_tables.md.
Receipt audit:T021A/receipt_audit.json. No thresholds/augmentation/topk/fallback tuning,
extra cohort,FCOS/SSD/val,spatial/meta/predictor/gate/dose or closed-branch experiments.
Stop NEEDS_REVIEW: pass means developmental candidate only; fail closes the fixed
horizontal-flip support filter under R033. Current Ours remains authoritative pending review.
'''
    (project/'research_log/T021A_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(gate=gate,aggregate=a,audit=audit,raw_files=len(files),raw_bytes=manifest['total_bytes']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--run',required=True);p.add_argument('--smoke',required=True)
    a=p.parse_args();render(a.project,a.run,a.smoke)
