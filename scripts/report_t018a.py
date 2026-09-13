"""Render the completed fixed T018-A receipts with standard-library arithmetic."""
import argparse
import hashlib
import json
import statistics as st
from pathlib import Path


METHODS = ('no_adapt', 'current_ours', 'nativePT_ours')
KEYS = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
CASES = ('gamma_s1', 'gamma_s2', 'contrast_s1', 'contrast_s2', 'color_cast_s1', 'color_cast_s2', 'clean_s0')


def table(title, headers, rows):
    def cell(v):
        return f'{v:.9g}' if isinstance(v, float) else str(v)
    return '\n'.join(['## '+title, '', '| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(cell, row))+' |' for row in rows])+ '\n\n'


def render(project, run_id, smoke_id):
    raw=project/'research_log'/'remote_runs'/run_id
    smoke=project/'research_log'/'remote_runs'/smoke_id
    root=raw/'artifacts'/'study'
    read=lambda name: json.loads((root/name).read_text())
    metrics,summary,diagnostics,completion=map(read,('metrics.json','summary.json','diagnostics.json','completion.json'))
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    assert completion['status']=='completed' and not completion['smoke']
    assert completion['images']==100 and len(rows)==1400 and completion['evaluations']==105
    assert all(all(r['isolation'].values()) for r in rows)
    assert all(all(v==0 for v in r['diagnostics'][0]['phi']) and len(r['diagnostics'])==4 and
               sum(d['update_applied'] for d in r['diagnostics'])==3 for r in rows)
    out=project/'research_log'/'T018A'
    out.mkdir(exist_ok=True)
    lines='# T018-A complete tables\n\nAP/AP50/AP75 use the 0–100 scale. All four precommitted blocks and all seven conditions are retained.\n\n'
    lines+=table('Official COCO AP', ['Group','Condition','Method','AP','AP50','AP75'],
                 [[group,c,m]+[100*values[f'{c}_{m}'][k] for k in ('AP','AP50','AP75')]
                  for group,values in metrics.items() for c in CASES for m in METHODS])
    macro_rows=[[group,m]+[100*st.mean(values[f'{c}_{m}'][k] for c in CASES[:-1]) for k in ('AP','AP50','AP75')]
                for group,values in metrics.items() for m in METHODS]
    lines+=table('Six-corruption macro metrics', ['Group','Method','AP','AP50','AP75'], macro_rows)
    lines+=table('Candidate-minus-current AP', ['Group','Macro delta','Positive conditions','Clean delta'],
                 [[g,v['candidate_minus_current'],v['positive_conditions'],v['clean_delta']] for g,v in summary['groups'].items()])
    drows=[]
    for name,d in diagnostics.items():
        drows.append([name,d['n'],d['phi3_norm']['mean'],d['phi3_norm']['median'],d['update_fraction'],
                      d['support_count']['mean'],d['support_count']['median'],d['empty_support_fraction'],
                      d['adapt_seconds']['mean'],d['adapt_seconds']['median'],d['deploy_seconds']['mean'],
                      d['prediction_count']['mean'],d['peak_allocated_bytes']['max']/1024**3])
    lines+=table('Adaptation diagnostics', ['Method/group','N','Mean phi norm','Median phi norm','Update fraction',
                 'Mean supports','Median supports','Empty fraction','Mean adapt s','Median adapt s',
                 'Mean adapt+teacher s','Mean predictions','Peak GiB'], drows)
    lines+='Timing includes steps 0–3 loss/gradient diagnostics with exactly three updates. The adapt+teacher field adds the shared original-image detector forward, but excludes final enhanced-image prediction, evaluation and state comparisons. CUDA peaks include the frozen source state copy used by instrumentation.\n\n'
    loss_rows=[]
    pair_rows=[]
    extra={}
    for group in ('clean','corrupted'):
        selected=[r for r in rows if r['method']=='nativePT_ours' and (r['case']=='clean_s0')==(group=='clean')]
        for step in range(4):
            loss_rows.append([group,step,len(selected)]+[st.mean(r['native_components'][step][k] for r in selected) for k in KEYS]+
                             [st.mean(r['diagnostics'][step]['total'] for r in selected)])
        delta=[r['paired_prediction_count_delta'] for r in selected]
        pair_rows.append([group,len(delta),st.mean(delta),st.median(delta),sum(x>0 for x in delta),sum(x==0 for x in delta),sum(x<0 for x in delta)])
        extra[group]={'paired_prediction_delta_mean':st.mean(delta),'paired_prediction_delta_median':st.median(delta),
                      'positive':sum(x>0 for x in delta),'zero':sum(x==0 for x in delta),'negative':sum(x<0 for x in delta)}
    lines+=table('Mean native pseudo-target components by diagnostic step', ['Group','Step','N',*KEYS,'Total'],loss_rows)
    lines+=table('Paired candidate-minus-current prediction counts', ['Group','N','Mean','Median','Positive','Zero','Negative'],pair_rows)
    lines+=table('Frozen R029 criteria', ['Criterion','Passed'],list(summary['gate']['flags'].items()))
    (out/'complete_tables.md').write_text(lines,encoding='utf-8')
    (out/'paired_prediction_counts.json').write_text(json.dumps(extra,indent=2)+'\n',encoding='utf-8')
    files=[]
    for folder in (raw,smoke):
        for path in sorted(p for p in folder.rglob('*') if p.is_file()):
            b=path.read_bytes()
            files.append({'path':path.relative_to(project).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    manifest={'run':run_id,'smoke_run':smoke_id,'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),'files':files}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    a=summary['groups']['aggregate'];gate=summary['gate']
    verdict=('PROMISING: all six developmental criteria pass; confirmation still requires research authorization.' if gate['passed'] else
             'INSUFFICIENT / WORSE: the fixed candidate fails the advancement rule; close this exact formulation without tuning.')
    report=f'''# T018-A — full native pseudo-target loss: NEEDS_REVIEW

**{verdict}**

The six-corruption macro AP is {a['macro_AP']['nativePT_ours']:.6f} for nativePT_ours,
{a['macro_AP']['current_ours']:.6f} for current_ours and {a['macro_AP']['no_adapt']:.6f} for no_adapt.
Candidate minus current is {a['candidate_minus_current']:+.6f} AP points;
candidate minus no-adapt is {a['candidate_minus_no_adapt']:+.6f}.
Positive candidate-minus-current macro deltas occur in {gate['positive_blocks']}/4 fixed blocks,
and {a['positive_conditions']}/6 corruption conditions improve. Clean AP delta is {a['clean_delta']:+.6f}.
These are developmental source-only results, with no independent confirmation or cross-detector claim.

## Provenance and implementation

- Research R029 / pointer 53f8662. Pre-outcome plan/cohort/config commit cd2ed29; implementation 7c43f1f.
- Release 20260913-153309-taisp-t018a-nativept. Formal run `{run_id}`; smoke `{smoke_id}`.
- Exactly 100 new COCO train2017 images, four precommitted blocks of 25, and seven unchanged conditions.
  Cohort SHA256 a01dfb1d40a6daceddccc1b7aa7f3f2e74871fd4a511d8d6c9acf8e50c2c111f.
  Zero overlap with 36 prior source IDs or all 5000 validation IDs. Selection and JPEG hashes are preserved.
- Original-condition frozen teacher predictions are computed once, filtered with existing score>=0.5/top20 ordering.
  Native targets contain detached boxes and integer labels only. All four native losses have unit weight.
- Existing global 8-D ISP and adapt_clip_radius are unchanged: identity initialization, K=3, LR=0.1,
  epsilon=1e-12 and current-phi CLIP gradient magnitude. CLIP direction is discarded.
  RPN/ROI sampling uses the precommitted seed 20260912 for each loss call. Four diagnostic evaluations apply three updates.
- Candidate code is isolated in analysis modules. Current Ours, teacher filtering, CLIP prompts/preprocessing,
  corruption definitions, detector and ISP bounds are unchanged. No deployed method replacement occurred.
- The adaptation loader sees image metadata/JPEGs only. Train annotations enter official COCO evaluation after
  every prediction file is written; annotation targets never enter adaptation.

## Validation and execution

- Baseline: 6 passed, 2 skipped in 1.40 s. Focused: 10 passed, 2 skipped in 1.70 s.
- Full regression: 148 passed, 10 skipped in 6.94 s. Optional skips are retained; these are not 148 real-model tests.
- CUDA smoke: two images, seven conditions, 28 adaptive episodes (14 native), K=3, four finite native
  component histories per episode, identity initialization and all frozen-state checks passed. No AP was computed.
- Formal: 100 images, 700 shared teacher forwards, 1400 adaptive episodes, 21 prediction files and
  105 official aggregate/block evaluations. Completion elapsed {completion['elapsed_seconds']:.6f} s; exit status is preserved in train.log.
- A6000 cuda:0, Python 3.12.12, torch 2.4.0+cu121, float32 model/ISP, one CPU thread.
  Existing NVML warning did not prevent CUDA execution. Package versions, weight/state hashes and exact commands
  are in environment.json, meta.json and run.sh. Model work used GPU; official COCO aggregation used CPU.
- Source state, requires_grad flags, eval modes and gradient absence are checked after every adaptive episode.
  Source and CLIP state hashes match before/after the whole study; every isolation check passes.
  ISP-owned phi remains zero with no accumulated gradient. Only the functional episodic phi is updated.
  Empty-support episodes retain exact zero phi and the existing identity-ISP output semantics.

'''
    report+=table('Aggregate official AP', ['Condition','No adapt','Current','NativePT','Native-current'],
                  [[c]+[100*metrics['aggregate'][f'{c}_{m}']['AP'] for m in METHODS]+
                   [100*(metrics['aggregate'][f'{c}_nativePT_ours']['AP']-metrics['aggregate'][f'{c}_current_ours']['AP'])] for c in CASES])
    report+=table('Fixed block corruption macro AP', ['Block','No adapt','Current','NativePT','Delta'],
                  [[g]+[v['macro_AP'][m] for m in METHODS]+[v['candidate_minus_current']]
                   for g,v in summary['groups'].items() if g!='aggregate'])
    report+=table('Frozen advancement rule', ['Criterion','Passed'],list(gate['flags'].items()))
    report+='## Diagnostics and limits\n\n'
    for name,d in diagnostics.items():
        report+=f"- {name}: phi norm mean/median {d['phi3_norm']['mean']:.9g}/{d['phi3_norm']['median']:.9g}; update fraction {d['update_fraction']:.6g}; support mean/median {d['support_count']['mean']:.6g}/{d['support_count']['median']:.6g}; adaptation mean/median {d['adapt_seconds']['mean']:.6g}/{d['adapt_seconds']['median']:.6g} s.\n"
    report+='''
Per-step component losses, paired prediction counts, AP/AP50/AP75 for all conditions/blocks and macro averages
are in [complete_tables.md](T018A/complete_tables.md). All negative conditions and blocks are retained.
Timing includes terminal diagnostics and excludes final enhanced-image inference/state verification/evaluation;
the adapt+teacher column adds the shared original teacher forward. No end-to-end throughput claim is made.
Peak allocated memory includes the frozen state copy used for verification.

The 100-image source set is developmental, not a publication or generalization claim. Diagnostics do not establish
a causal explanation of AP changes. No threshold, loss weight, step count, learning rate or cohort was tuned.
T017-A remains numerically blocked and unchanged; no component-attribution conclusion follows from T018-A.

## Artifacts and next action

'''
    report+=f"All {len(files)} raw files ({manifest['total_bytes']:,} bytes), including smoke and formal receipts, are hashed in [artifact_manifest.json](T018A/artifact_manifest.json). Raw root: `research_log/remote_runs/{run_id}/`.\n\n"
    report+='Stop for research review. No active experiment remains. Do not launch confirmatory, FCOS/SSD, COCO-val, spatial ISP, component-only variants, sweeps or deployment changes without a new queued task.\n'
    (project/'research_log'/'T018A_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'verdict':verdict,'gate':gate,'aggregate':a,'raw_files':len(files),'raw_bytes':manifest['total_bytes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--run',required=True)
    p.add_argument('--smoke',required=True)
    args=p.parse_args()
    render(args.project.resolve(),args.run,args.smoke)
