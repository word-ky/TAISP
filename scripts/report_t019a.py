"""Render completed R031 fixed-family receipts without model/evaluator calls."""
import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

from report_t018a import CASES, table

CANDIDATES=('native_cls','native_conf','native_roi','native_conf_roi')
METHODS=('no_adapt','current_ours',*CANDIDATES)
ACTIVE={
    'native_cls':('loss_classifier',),
    'native_conf':('loss_classifier','loss_objectness'),
    'native_roi':('loss_classifier','loss_box_reg'),
    'native_conf_roi':('loss_classifier','loss_objectness','loss_box_reg'),
}


def render(project,run,smoke):
    raw=project/'research_log'/'remote_runs'/run;root=raw/'artifacts'/'study'
    read=lambda name:json.loads((root/name).read_text())
    metrics,summary,diagnostics,completion=map(read,('metrics.json','summary.json','diagnostics.json','completion.json'))
    assert completion['status']=='completed' and not completion['smoke']
    assert completion['images']==200 and completion['adaptive_samples']==7000 and completion['evaluations']==210
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    assert len(rows)==7000 and all(all(r['isolation'].values()) for r in rows)
    out=project/'research_log'/'T019A';out.mkdir(exist_ok=True)
    all_tables='# T019-A complete tables\n\nAP/AP50/AP75 and deltas are reported on the0–100scale. All fixed candidates/conditions/blocks are retained.\n\n'
    all_tables+=table('Official COCO metrics', ['Group','Condition','Method','AP','AP50','AP75'],
                      [[g,c,m]+[100*v[f'{c}_{m}'][k] for k in ('AP','AP50','AP75')]
                       for g,v in metrics.items() for c in CASES for m in METHODS])
    all_tables+=table('Six-corruption macro metrics', ['Group','Method','AP','AP50','AP75'],
                      [[g,m]+[100*st.mean(v[f'{c}_{m}'][k] for c in CASES[:-1]) for k in ('AP','AP50','AP75')]
                       for g,v in metrics.items() for m in METHODS])
    all_tables+=table('All condition deltas against both baselines', ['Group','Condition','Candidate','Reference','AP delta','AP50 delta','AP75 delta'],
                      [[g,c,m,ref]+[100*(v[f'{c}_{m}'][k]-v[f'{c}_{ref}'][k]) for k in ('AP','AP50','AP75')]
                       for g,v in metrics.items() for c in CASES for m in CANDIDATES for ref in METHODS[:2]])
    macro_deltas=[];overview=[];blocks=[];flags=[]
    for m,item in summary['candidates'].items():
        a=item['groups']['aggregate']
        overview.append([m,a['macro_AP'][m],a['candidate_minus_current'],a['candidate_minus_no_adapt'],
                         a['additional_macro_deltas']['AP50'],a['additional_macro_deltas']['AP75'],
                         item['positive_blocks'],a['positive_conditions'],a['clean_delta'],a['clean_delta_vs_raw'],item['eligible']])
        for g,v in item['groups'].items():
            for ref in METHODS[:2]:
                macro_deltas.append([g,m,ref,v['macro_AP'][m]-v['macro_AP'][ref]]+
                                   [v['additional_macro_metrics'][k][m]-v['additional_macro_metrics'][k][ref] for k in ('AP50','AP75')])
            if g!='aggregate':blocks.append([m,g,v['candidate_minus_current'],v['additional_macro_deltas']['AP50'],v['additional_macro_deltas']['AP75']])
        flags.extend([m,k,v] for k,v in item['flags'].items())
    all_tables+=table('Macro deltas against both baselines', ['Group','Candidate','Reference','AP','AP50','AP75'],macro_deltas)
    all_tables+=table('Frozen eligibility checks', ['Candidate','Criterion','Passed'],flags)
    diag=[]
    for g,d in diagnostics.items():
        diag.append([g,d['n'],d['phi3_norm']['mean'],d['phi3_norm']['median'],d['update_fraction'],
                     d['support_count']['mean'],d['support_count']['median'],d['empty_support_fraction'],
                     d['adapt_seconds']['mean'],d['adapt_seconds']['median'],d['deploy_seconds']['mean'],d['peak_allocated_bytes']['max']/1024**3])
    diag_table=table('Adaptation diagnostics', ['Method/group','N','Mean phi','Median phi','Update fraction','Mean support',
                     'Median support','Empty fraction','Mean adapt s','Median adapt s','Mean adapt+teacher s','Peak GiB'],diag)
    all_tables+=diag_table
    losses=[]
    for m in CANDIDATES:
        for group in ('clean','corrupted'):
            selected=[r for r in rows if r['method']==m and (r['case']=='clean_s0')==(group=='clean')]
            for step in range(4):
                values=[st.mean(r['native_components'][step][k] for r in selected) if k in ACTIVE[m] else 'inactive'
                        for k in ('loss_classifier','loss_objectness','loss_box_reg')]
                losses.append([m,group,step]+values+[st.mean(r['diagnostics'][step][k] for r in selected)
                              for k in ('total','detector_gradient_norm','clip_gradient_norm','gradient_norm')])
    all_tables+=table('Mean active losses and gradient norms by diagnostic step',
                      ['Candidate','Group','Step','Classifier','Objectness','ROI box reg','Total','Native grad norm','CLIP grad norm','Update norm'],losses)
    all_tables+='Inactive means absent from the optimized objective, not a measured zero. RPN box regression is absent from every candidate. Native forward still computes the usual outputs; only active keys contribute to the returned loss/gradient.\n\n'
    all_tables+='Latency includes four diagnostic loss/gradient evaluations with exactly three updates. Teacher-inclusive adds the shared original teacher forward, excluding final prediction/state verification/AP. Peak memory includes a frozen state copy used for verification.\n'
    (out/'complete_tables.md').write_text(all_tables,encoding='utf-8')
    files=[]
    for folder in (raw,project/'research_log'/'remote_runs'/smoke):
        for p in sorted(x for x in folder.rglob('*') if x.is_file()):
            b=p.read_bytes();files.append({'path':p.relative_to(project).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    manifest={'run':run,'smoke_run':smoke,'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),'files':files}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    gate=summary['gate'];selected=gate['selected']
    verdict=f'{selected} selected for later confirmation' if selected else 'No candidate eligible; close this fixed component family'
    reference=summary['candidates'][CANDIDATES[0]]['groups']['aggregate']['macro_AP']
    report=f'''# T019-A — {verdict}; NEEDS_REVIEW

Eligible candidates: {gate['eligible_candidates']}. Selected: **{selected}**.
Near-best candidates within0.01AP: {gate['within_point01_of_best']}.
Selection uses only frozen six-corruption macro AP, then macro AP75 within the0.01AP tie range,
then number of positive blocks. Any selected method is a developmental candidate, not confirmed.
Baseline corruption macro AP: no-adapt {reference['no_adapt']:.9f}, current Ours {reference['current_ours']:.9f}.

## Protocol and provenance

R031/pointerd60e72c; pre-outcome plan/config/cohort/method hashes b49c432;
experiment codeee98de0; CUDA smoke receipt d047312.
Release20260913-182322-taisp-t019a-components; formal `{run}`; smoke `{smoke}`.
200newtrain2017images,fourprecommitted50-imageblocks,636prior-source/all5000val excluded.
CohortSHA534b17343ccba995beb9d112d552eefd7f1b2116479c2132dd99265b1d6735b4.
Same readable/non-crowd positive-box/area eligibility as T018-A/B;13,162eligible after exclusions.
Hash selection seed20260920; model/native sampling seed20260912. Annotation eligibility is used
only for cohort preparation. Adaptation sees JPEGs and detached teacher predictions; official
GT annotations enter evaluation only after all prediction files are saved.

Only named native-loss subset selection/configuration and study reporting were added.
Ten protected modules remain unchanged: existing current Ours, adaptation, source detector,
CLIP, teacher filtering, oracle, ISP and corruptions. Default/full native behavior remains
backward compatible; the failed full-native formulation was not evaluated as a fifth candidate.
All candidates use detachedscore>=.5/top20 teacher boxes/int labels, unit active coefficients,
global8D identity phi,K3/LR.1/EPS1e-12 and current-phi CLIP-norm transfer with direction discarded.
Detector and CLIP remain frozen; only functional episodic ISP state changes.

'''
    report+=table('Exact fixed family', ['Candidate','Active keys'],[[m,', '.join(v)] for m,v in ACTIVE.items()])
    report+=f'''## Validation and execution

- Unchanged baseline12passed2skipped1.57s; focused24passed2skipped1.82s.
- Full162passed10skipped6.84s. Optional tests remain skipped; these are not162real-model tests.
- Two-imageCUDAK3smoke:70adaptiveepisodes,56candidateepisodes,224exactfloat32active-sum checks,
  allstate/K3checks passed,0AP,exit0. Exact receipt inT019A_smoke_audit.json.
- Formal200images,1400teacherforwards,7000adaptiveepisodes,42predictionfiles,210officialevaluations.
  Collection/evaluation elapsed{completion['elapsed_seconds']:.6f}s. Exact start/finish/exit in train.log.
- A6000cuda:0,float32,threads1. Source/CLIP before-after hashes and all per-episode isolation checks pass.
  CUDA model work completed despite the existing NVML warning. Exact environment/commands are retained.
- Empty-support semantics are covered by focused tests and per-episode checks. Actual empty fractions appear below.
  Four diagnostic evaluations apply three updates. Latency includes terminal diagnostics and excludes final
  prediction/state verification/evaluation; teacher-inclusive adds the original detector teacher forward.

'''
    report+=table('Aggregate candidate outcomes', ['Candidate','Macro AP','Delta current','Delta raw','AP50 delta current','AP75 delta current',
                  'Positive blocks/4','Positive conditions/6','Clean delta current','Clean delta raw','Eligible'],overview)
    report+=table('Aggregate AP across all conditions', ['Condition',*METHODS],
                  [[c]+[100*metrics['aggregate'][f'{c}_{m}']['AP'] for m in METHODS] for c in CASES])
    report+=table('All four block deltas', ['Candidate','Block','AP delta current','AP50 delta current','AP75 delta current'],blocks)
    report+=table('All frozen eligibility checks', ['Candidate','Criterion','Passed'],flags)
    if not selected:
        common_failures=[k for k in summary['candidates'][CANDIDATES[0]]['flags']
                         if all(not v['flags'][k] for v in summary['candidates'].values())]
        report+='Every candidate fails: '+', '.join(common_failures)+'. No tie-breaker is invoked because none is eligible. Close this fixed component-subset branch without weights/thresholds or another cohort search.\n\n'
    report+=diag_table
    report+='All condition/block AP/AP50/AP75 values and deltas against both baselines, active per-step losses and gradient norms are in [complete_tables.md](T019A/complete_tables.md); per-episode raw traces retain all supports/phi/gradients/counts/timings. No negative candidate, condition or block is omitted.\n\n'
    report+='This is a four-candidate source-development study in the availability-defined training population. No candidate is independently confirmed here, and no cross-detector/generalization claim follows. Component comparisons do not prove a causal account of teacher geometry noise. T018-B remains a negative full-native confirmation; T017 attribution remains blocked.\n\n'
    report+=f"All{len(files)}raw files ({manifest['total_bytes']:,}bytes), including smoke, are hashed in [artifact_manifest.json](T019A/artifact_manifest.json).\n\n"
    report+='StopNEEDS_REVIEW. No coefficient/threshold/K/LR/prompt tuning, new cohort, FCOS/SSD/val2017, spatial ISP, T017 forensics, gate/dose/meta-training or deployment replacement occurred. Follow-on work requires a new explicit research task.\n'
    (project/'research_log'/'T019A_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'gate':gate,'outcomes':overview,'raw_files':len(files),'raw_bytes':manifest['total_bytes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--run',required=True)
    p.add_argument('--smoke',required=True)
    a=p.parse_args();render(a.project.resolve(),a.run,a.smoke)
