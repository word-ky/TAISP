"""Render completed R030 source500 receipts; no model or evaluator calls."""
import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

from report_t018a import CASES, KEYS, METHODS, table


def render(project, run, smoke):
    raw=project/'research_log'/'remote_runs'/run
    root=raw/'artifacts'/'study'
    read=lambda name:json.loads((root/name).read_text())
    metrics,summary,diagnostics,completion=map(read,('metrics.json','summary.json','diagnostics.json','completion.json'))
    assert completion['status']=='completed' and not completion['smoke']
    assert completion['images']==500 and completion['adaptive_samples']==7000 and completion['evaluations']==126
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    assert len(rows)==7000 and all(all(r['isolation'].values()) for r in rows)
    out=project/'research_log'/'T018B';out.mkdir(exist_ok=True)
    complete='# T018-B complete results\n\nAll AP metrics/deltas are in 0–100 AP points. AP50/AP75 are diagnostics only.\n\n'
    complete+=table('Official COCO metrics', ['Group','Condition','Method','AP','AP50','AP75'],
                    [[g,c,m]+[100*v[f'{c}_{m}'][k] for k in ('AP','AP50','AP75')]
                     for g,v in metrics.items() for c in CASES for m in METHODS])
    macro=[[g,m]+[100*st.mean(v[f'{c}_{m}'][k] for c in CASES[:-1]) for k in ('AP','AP50','AP75')]
           for g,v in metrics.items() for m in METHODS]
    complete+=table('Six-corruption macro metrics', ['Group','Method','AP','AP50','AP75'],macro)
    complete+=table('Candidate-minus-current metrics', ['Group','Condition','AP delta','AP50 delta','AP75 delta'],
                    [[g,c]+[100*(v[f'{c}_nativePT_ours'][k]-v[f'{c}_current_ours'][k]) for k in ('AP','AP50','AP75')]
                     for g,v in metrics.items() for c in CASES])
    complete+=table('Macro and clean deltas', ['Group','Macro AP','Macro AP50','Macro AP75','Clean AP'],
                    [[g,v['candidate_minus_current'],v['additional_macro_deltas']['AP50'],
                      v['additional_macro_deltas']['AP75'],v['clean_delta']] for g,v in summary['groups'].items()])
    complete+=table('Adaptation diagnostics', ['Method/group','N','Mean phi norm','Median phi norm','Update fraction',
                    'Mean supports','Median supports','Empty fraction','Mean adapt s','Median adapt s','Mean adapt+teacher s','Peak GiB'],
                    [[g,d['n'],d['phi3_norm']['mean'],d['phi3_norm']['median'],d['update_fraction'],
                      d['support_count']['mean'],d['support_count']['median'],d['empty_support_fraction'],
                      d['adapt_seconds']['mean'],d['adapt_seconds']['median'],d['deploy_seconds']['mean'],
                      d['peak_allocated_bytes']['max']/1024**3] for g,d in diagnostics.items()])
    loss_rows=[];pairs=[]
    for group in ('clean','corrupted'):
        selected=[r for r in rows if r['method']=='nativePT_ours' and (r['case']=='clean_s0')==(group=='clean')]
        for step in range(4):
            loss_rows.append([group,step,len(selected)]+[st.mean(r['native_components'][step][k] for r in selected) for k in KEYS]+
                             [st.mean(r['diagnostics'][step]['total'] for r in selected)])
        d=[r['paired_prediction_count_delta'] for r in selected]
        pairs.append([group,len(d),st.mean(d),st.median(d),sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)])
    complete+=table('Mean native losses at each diagnostic step', ['Group','Step','N',*KEYS,'Total'],loss_rows)
    complete+=table('Paired prediction count deltas', ['Group','N','Mean','Median','Positive','Zero','Negative'],pairs)
    complete+='Latency includes steps0–3 diagnostics with exactly three updates. Teacher-inclusive latency adds the shared original-image detector forward but excludes final enhanced-image prediction, state verification and official evaluation. Peak memory includes the frozen source state copy used for verification.\n'
    (out/'complete_tables.md').write_text(complete,encoding='utf-8')
    files=[]
    for folder in (raw,project/'research_log'/'remote_runs'/smoke):
        for path in sorted(p for p in folder.rglob('*') if p.is_file()):
            b=path.read_bytes();files.append({'path':path.relative_to(project).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    manifest={'run':run,'smoke_run':smoke,'file_count':len(files),'total_bytes':sum(r['bytes'] for r in files),'files':files}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    a=summary['groups']['aggregate'];gate=summary['gate']
    verdict='SOURCE-CONFIRMED' if gate['passed'] else 'NOT SOURCE-CONFIRMED'
    report=f'''# T018-B — {verdict}; NEEDS_REVIEW

The fixed nativePT candidate achieves six-corruption macro AP {a['macro_AP']['nativePT_ours']:.9f},
versus current Ours {a['macro_AP']['current_ours']:.9f} and no-adapt {a['macro_AP']['no_adapt']:.9f}.
Native-current is {a['candidate_minus_current']:+.9f} AP; native-no-adapt is {a['candidate_minus_no_adapt']:+.9f} AP.
Positive blocks: {gate['positive_blocks']}/5. Positive corruption conditions: {a['positive_conditions']}/6.
Clean AP delta: {a['clean_delta']:+.9f}. R030 rule passed: **{gate['passed']}**.
Macro AP50 delta: {a['additional_macro_deltas']['AP50']:+.9f}; macro AP75 delta: {a['additional_macro_deltas']['AP75']:+.9f}.
AP75 nonnegative diagnostic: {summary['AP75_nonnegative_diagnostic']}; it is not a confirmation gate.

## Protocol and provenance

R030 / pointer7902e3a. Pre-outcome plan/config/cohort958d898; experiment code d8ff14f.
Release20260913-161116-taisp-t018b-source500; formal `{run}`; smoke `{smoke}`.
Cohort SHA256 e7a771126ae2fee9844dde456648b50f4c01ebcdc404318c21f9a33260da7b17.
500 new train2017 images, five precommitted consecutive100-image blocks, zero overlap with
136 prior-source IDs or all5000val IDs. Same disclosed readable/non-crowd positive-box/area eligibility
as T018-A: 13,662 available eligible images after exclusions. Selection uses hash seed20260919.

NativePseudoTargetLoss, adapt_clip_radius, current Ours, source loading/teacher selection,
CLIP, oracle, ISP and corruptions are unchanged from accepted T018-A; 11 protected modules
are pinned in T018B_method_pins.json. The runner only extends cohort count and result summary.
Same score>=.5/top20 detached teacher boxes/int labels, unit four-native-loss sum, global8D
identity phi, K3/LR.1/EPS1e-12, CLIP norm transfer and seeded native sampling20260912.
No detector/CLIP optimizer or parameter update. Ground-truth eligibility metadata is used in
cohort preparation; adaptation receives JPEGs and detached teacher targets only. Official
annotations are loaded after all prediction files are saved, for evaluation only.

## Tests and execution

- Unchanged baseline10passed2skipped1.74s; focused12passed2skipped1.77s.
- Full regression150passed10skipped7.06s. Optional skips retained; these are not150real-model tests.
- CUDAK3 smoke:2images,28adaptiveepisodes,14teacherforwards,all isolation checks passed,0AP,exit0.
- Formal:500images,3500teacherforwards,7000adaptiveepisodes,21predictionfiles,126official evaluations.
  Collection/evaluation elapsed {completion['elapsed_seconds']:.6f}s. Exact timestamps/exit status in train.log.
- A6000cuda:0,float32 models/ISP,oneCPUthread; source/CLIP frozen with before-after state hashes and
  per-episode source/ISP/support isolation checks. The run completed with all isolation checks passing.
  Existing NVML warning did not prevent CUDA execution. Environment/package/hash/command receipts retained.
- Empty-support identity behavior is covered by existing tests; actual empty fraction is reported in tables.
  Adaptation includes four diagnostic evaluations with exactly three updates; its latency excludes
  final prediction, state verification and evaluation. Teacher-inclusive adds original teacher forward.

'''
    report+=table('Aggregate results', ['Condition','No-adapt AP','Current AP','Native AP','AP delta','AP50 delta','AP75 delta'],
                  [[c]+[100*metrics['aggregate'][f'{c}_{m}']['AP'] for m in METHODS]+
                   [100*(metrics['aggregate'][f'{c}_nativePT_ours'][k]-metrics['aggregate'][f'{c}_current_ours'][k]) for k in ('AP','AP50','AP75')] for c in CASES])
    report+=table('Fixed block macro AP deltas', ['Block','Delta','AP50 delta','AP75 delta'],
                  [[g,v['candidate_minus_current'],v['additional_macro_deltas']['AP50'],v['additional_macro_deltas']['AP75']]
                   for g,v in summary['groups'].items() if g!='aggregate'])
    report+=table('All six R030 criteria', ['Criterion','Passed'],list(gate['flags'].items()))
    report+='## Diagnostic summary and limitations\n\n'
    for g,d in diagnostics.items():
        report+=f"- {g}: phi norm mean/median {d['phi3_norm']['mean']:.9g}/{d['phi3_norm']['median']:.9g}; update fraction {d['update_fraction']:.9g}; support mean/median {d['support_count']['mean']:.9g}/{d['support_count']['median']:.9g}; adapt mean/median {d['adapt_seconds']['mean']:.9g}/{d['adapt_seconds']['median']:.9g}s.\n"
    report+='\nAll conditions, blocks, AP/AP50/AP75 metrics, native component histories and paired counts are retained in [complete_tables.md](T018B/complete_tables.md). The full per-step records remain in samples.jsonl. Negative results are not omitted.\n\n'
    report+='This is a disjoint confirmation on the same source detector and availability-defined train population. Source-detector self-consistency remains an alternative explanation; no cross-detector/generalization conclusion is supported. Clean phi/update diagnostics do not by themselves establish selective adaptation. T017 attribution remains blocked and untouched.\n\n'
    report+=f"All{len(files)}raw files ({manifest['total_bytes']:,}bytes), including smoke, are hashed in [artifact_manifest.json](T018B/artifact_manifest.json). Exact settings, JPEG/cohort hashes and commands are retained.\n\n"
    report+='Stop NEEDS_REVIEW. No follow-on FCOS/SSD, COCO-val, hyperparameter/component search, spatial ISP, T017 forensics, meta-training or deployment replacement was executed. A new explicit research task is required for the next experiment.\n'
    (project/'research_log'/'T018B_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'verdict':verdict,'aggregate':a,'gate':gate,'files':len(files),'bytes':manifest['total_bytes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--run',required=True)
    p.add_argument('--smoke',required=True)
    args=p.parse_args()
    render(args.project.resolve(),args.run,args.smoke)
