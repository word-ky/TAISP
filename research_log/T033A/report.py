"""R051 report from immutable receipts, stdlib only; no model/SVD calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_log/T033A';RUNS=ROOT/'research_log/remote_runs'
CAND=RUNS/'20260914-191144-taisp-t033a-candidates600/artifacts/candidate'
REP=RUNS/'20260914-192538-taisp-t033a-representation/artifacts/representation'
TRAIN=RUNS/'20260914-192812-taisp-t033a-train480-model/artifacts/train'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(directions_run,holdout_run):
    dirs=RUNS/directions_run/'artifacts/directions';hold=RUNS/holdout_run/'artifacts/holdout'
    manifests={}
    for name,folder in [('candidate',CAND),('representation',REP),('train',TRAIN),('directions',dirs),('holdout',hold)]:
        hashes=read(folder/'sha256.json')
        for f,h in hashes.items():assert sha(folder/f)==h,f
        manifests[name]={'files':len(hashes),'manifest_sha256':sha(folder/'sha256.json')}
    cs,tr,hr=[read(p/'records.json') for p in [CAND,TRAIN,hold]]
    assert (len(cs),len(tr),len(hr))==(600,480,120)
    bykey={(r['image_id'],r['case']):r for r in cs}
    assert all(r['g_hard']==bykey[r['image_id'],r['case']]['g_hard'] for r in tr+hr)
    assert all(r['integrity_passed'] and r['rng']['restored'] for r in cs+tr+hr)
    assert not {r['image_id'] for r in tr}&{r['image_id'] for r in hr}
    dr=read(dirs/'directions.json');assert len(dr)==120
    assert all((a['episode_index'],a['u_obj'])==(b['episode_index'],b['u_obj']) for a,b in zip(dr,hr))
    rep=read(REP/'representation.json');model=read(TRAIN/'model.json');s=read(hold/'summary.json')
    ml=read(ROOT/'research_log/T033A_model_lock.json');dl=read(ROOT/'research_log/T033A_directions_lock.json')
    rl=read(ROOT/'research_log/T033A_representation_lock.json');pf=read(hold/'preflight.json')
    assert ml['files']==read(TRAIN/'sha256.json') and dl['files']==read(dirs/'sha256.json')
    assert model['train_reference_sha256']==sha(TRAIN/'records.json')
    assert model['holdout_task_gradients_computed']==0 and rep['task_gradients_computed']==0
    assert pf['direction_lock']['directions_commit']==dl['directions_commit']
    assert pf['direction_lock']['model_commit']==ml['model_commit']
    pins=read(ROOT/'research_log/T033A_all_code_pins.json')
    checks={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in pins.items()}
    assert all(checks.values()) and all(read(OUT/'remote_code_checks.json').values())
    integrity={'raw_manifests':manifests,'all1200_integrity_rng':True,'candidate_vectors_unchanged':True,
        'train_holdout_disjoint':True,'corrected_directions_exactly_consumed':True,'code_pins':checks,
        'representation_commit':rl['representation_commit'],'model_commit':ml['model_commit'],
        'directions_commit':dl['directions_commit'],'preflight':pf,'AP_calls':0,
        'candidate_records_sha256':sha(CAND/'records.json'),'representation_sha256':sha(REP/'representation.json'),
        'states_sha256':sha(REP/'states.json'),'train_records_sha256':sha(TRAIN/'records.json'),
        'model_sha256':sha(TRAIN/'model.json'),'directions_sha256':sha(dirs/'directions.json'),
        'holdout_records_sha256':sha(hold/'records.json')}
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    for f in ['summary.json','preflight.json','completion.json']:(OUT/f).write_bytes((hold/f).read_bytes())
    (OUT/'model.json').write_bytes((TRAIN/'model.json').read_bytes());(OUT/'directions.json').write_bytes((dirs/'directions.json').read_bytes())
    fields=['episode_index','image_id','case','block','support_count','S_hard','S_obj','Delta','norm_hard','norm_obj',
            'cos_hard_task','cos_obj_task','abstain','integrity_passed']
    with (OUT/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows({k:r[k] for k in fields} for r in hr)
    groups={'overall':s['overall'],'clean':s['clean'],'corrupt':s['corrupt'],**s['conditions'],
            **{'block'+k:v for k,v in s['blocks'].items()},**{'support_'+k:v for k,v in s['support_subsets'].items()}}
    lines=[];coords=[]
    for name,g in groups.items():
        d=g['distributions'];o=d.get('S_obj',d.get('S_cal'));cos=d.get('cos_obj_task',d.get('cos_cal_task'))
        lines.append(f"|{name}|{g['n']}|{d['S_hard']['positive']}|{o['positive']}|{d['Delta']['positive']}|{d['Delta']['median']}|{d['Delta']['mean']}|{d['cos_hard_task']['median']}|{cos['median']}|")
        for key,c in g['coordinates'].items():coords.append(f"|{name}|{key}|{c['max_energy_coordinate']}|{c['max_energy_share']}|")
    variance=sum(v*v for v in rep['pca_singular_values'][:16])/sum(v*v for v in rep['pca_singular_values'])
    verdict='PASS' if s['passed'] else 'FAIL';gates='\n'.join(f'|{k}|{v}|' for k,v in s['gates'].items())
    report=f'''# T033-A / R051 — scientific {verdict}; NEEDS_REVIEW

The exact confidence-pooled own-ROI -> PCA16 -> affine tangent-correction family {'passes the frozen source holdout capacity gate; review before any runtime work' if s['passed'] else 'fails the frozen source holdout conjunction and is closed without same-holdout rescue'}. No AP, finite-step, cross-detector or runtime result is claimed. T031, T032, T013 and memory branches remain closed and were not reused as learned models.

## Frozen inputs and ordering

Research R051 1d17ff9 / latest ae70654. Plan/cohort commit **4126a98** preceded new model calls. Candidate code/pins **b5d4c6fd024cc45c9a1c417be34709cd2532d58a**; numerical code **c0d70fc**; stage/reference code **13ecdd9**, all fixed before actual PCA/source fitting. Candidate600 commit **25d74e9142e58224d2589ac00be18872292606c5**, descriptor **dff5355**, BEFORE representation fit. Train-only representation/all600states commit **{rl['representation_commit']}**, descriptor **fabdfda**, BEFORE source annotation/reference process. Train480/model commit **{ml['model_commit']}** BEFORE any holdout task gradient. All120 corrected directions commit **{dl['directions_commit']}** and pre-oracle descriptor BEFORE the holdout reference process. Complete lock receipt: {json.dumps(pf['direction_lock'])}.

Fresh300 COCOtrain2017images, selectionseed20261004; excludes3491 prior/reserved IDs including T032, plus5000val and1400evaluation IDs. Existing eligible validbbox/readableJPEG selection sha256(seed:ID),ID; cyclic three established severity2families100pairs each. First240images train480episodes,80pairs/family; last60holdout120episodes,20pairs/family; four15pairblocks5/family. Clean then assignedcorruption perimage. CohortSHA{sha(ROOT/'research_log/T033A_train_cohort.json')}; allimage/exclusion-set provenance hashes in cohort. Preparation parsed source annotations only for outcome-free eligibility and writing separate train/holdout annotation subsets. Candidate manifest has no annotation paths; candidate construction/representation/direction stage imports no oracle/reference/source-meta path. Actual train process opens only train annotations; holdout annotations only after committed corrected directions.

## Candidate and representation

Unchanged hard candidate calls existing orthogonal_candidates.candidate / original view_gradient, frozen score>=.50 stabletop20 confidence-weighted pseudo loss and 8-D identityISP. No support regeneration after source labels. The stored own-support boxes feed existing fixed_roi_representation at identityISP. Per-ROI1024-D float32feature is L2-normalized, detached-score weighted mean uses denominator sumscore+1e-12, pooled vector L2-normalized with eps1e-12. No predicted class identity, GT, family/case flags, memory, native losses or CLIP enter descriptor/state. PerROI rawfeature hashes, pooleddescriptor hash/vector/norm, exactsupports/hash,count,meanscore retained. Empty support=>zero1024descriptor/count0/score0. All600 candidates include{sum(r['support_count']==0 for r in cs)}empty supports and{sum(r['zero'] for r in cs)}zero hard gradients, all retained.

PCA fits centered480x1024train descriptors only, onefloat64CPUthinSVD; numericalrank={rep['pca_rank']}, threshold={rep['pca_rank_tolerance']}, top16 variance fraction={variance}. Largest-absolute-loading (earliest tie) signpositive; no dimension/rank sweep. State27=[PCA16, g/(normg+1e-12)8, log(normg+1e-12),count/20,meanscore]. Train-only populationmean/std; constantdims={rep['constant_dimensions']} withscale1/forcedzero convention precommitted. All600states saved with individual float64-vector hashes before source labels. Fullbasis/spectrum/signindices/state means/scales in representation.json. NumPy{rep['numpy_version']}.

## One affine tangent fit

Train480 source taskgradients use unchanged oracle at identityISP, reference seed20260913 and existing common-JVP/reverse parity. u_h=g/(normg+1e-12),u_t=t/(normt+1e-12), target=u_t-dot(u_t,u_h)u_h; exactzero hard/task=>zerotarget, keepallrows. Onefloat64SVDminimum-norm affine fit X=[1,h_std], no ridge/weights/optimizer/search. Rank={model['rank']}, ranktolerance={model['rank_tolerance']}; singular spectrum={model['singular_values']}; effectivecondition={model['condition_effective']}, fullcondition={model['condition_full']}; coefficientFrobenius={model['coefficient_frobenius']},Bnorm={model['B_frobenius']},bnorm={model['b_l2']},trainresidualSSE={model['training_squared_residual']}. Zero hard/task/target counts={model['zero_hard_count']}/{model['zero_task_count']}/{model['zero_target_count']}. All480train targets/predictions, b/B and fit diagnostics retained.

Application is fixed r_perp=r_hat-dot(r_hat,u_h)u_h;u_obj=(u_h+r_perp)/(norm(u_h+r_perp)+1e-12); exactzerohard=>zero/abstain. This exact eps-normalized formula is retained; projectiondot and finalnorm are diagnostics, not a new fitted correction or tolerance. All120 corrected directions computed and committed in a separate label-free process; reference consumes those exact vectors without recomputation. Maximum absolute projectiondot={max(abs(r['projection_dot']) for r in dr)}. Holdout abstentions={s['abstentions']}, counted nonpositive.

## Execution and integrity

Candidate run20260914-191144-taisp-t033a-candidates600:600episodes/{read(CAND/'completion.json')['seconds']}s, sourceb5d4c6f,release20260914-191003-taisp-t033a-candidates. Representation run20260914-192538-taisp-t033a-representation, sourcedff5355. Train run20260914-192812-taisp-t033a-train480-model:480episodes/{read(TRAIN/'completion.json')['seconds']}s, sourcefabdfda. Directions run{directions_run}. Holdout run{holdout_run}:120episodes/{read(hold/'completion.json')['seconds']}s. Postcandidate stages use identical frozen release20260914-192036-taisp-t033a-reference. Exact commands, commits, timestamps and exit statuses retained in raw run.sh/meta/train.log. Allmodel forwards/gradients onA6000CUDA0; CPUonlyPCA/affineSVD/summaries. Frozen detectorstate73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf andweight258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384 unchanged. No CUDA/RNG/seed/model/ISP setting changes.

All600candidate+480train+120holdout integrity/state/RNG/JVP checks pass; allhardvectors unchanged; train/holdoutdisjoint;alllocks verifiedbeforeannotations; savedu_obj consumedexactly. All{len(checks)}local/remote pinsmatch,including65protected/prior paths. Rawmanifests: {json.dumps(manifests)}. ArtifactSHA receipt: {json.dumps({k:v for k,v in integrity.items() if k.endswith('_sha256')})}.

## Exact R051 gate

S_hard=dot(task,hard)/(normhard+1e-12);S_obj=dot(task,u_obj);Delta=S_obj-S_hard. Required S_objpositive82/120overall45/60corrupt;Deltapositive72/120overall36/60corrupt;positive mean+medianoverall/corrupt;>=3/4positiveblockmedians;cleanmedian>=0;finite/zeroabstention/integrity. Summary S_cal/g_cal/cos_cal_task aliases refer only to S_obj/u_obj/object-corrected cosine. No oldmap comparator.

|Gate|Pass|
|---|---|
{gates}

|Group|N|S_hard>0|S_obj>0|Delta>0|MedianDelta|MeanDelta|Median cos(h,task)|Median cos(obj,task)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Outliers and empty/zero episodes remain in totals; undefined zero cosine is None, not an invented direction. Full distributions and per-episode TSV retained.

|Group|Gradient|Max-energy coordinate|Max share|
|---|---|---:|---:|
{chr(10).join(coords)}

Energy shares compare rawhard to unit corrected direction and are descriptive only, not matched-norm performance. Percoordinate mean/std/meanabs/RMS in summary. No subgroup-selected rescue.

## Validation, changes and recovery

Baseline5passed3warnings5.12s; candidate+hardfocused5passed3warnings7.64s; mathfocused3passed1.61s; stage+mathfocused5passed5.45s; full286passed11skipped4warnings38.81s, completed before actual representation/source outcomes. Tests cover normalized confidence pool, empty supports, class irrelevance, original hard reuse, PCA sign/rankcollapse/train-onlystandardization, one affineSVD/minimumnorm, tangent andzero formulas,82positive threshold, abstention accounting, meanoutlier andlazyoracle boundary. Real600+480+120 serves as end-to-end integration. No outcome-driven core code change or extra tests after frozen code.

New analysis object_state_candidates/math/stages/reference modules and focusedtests; project-local report/plan/cohort/codepins/locks/rawreceipts. Existing detector/ISP/tta/current-Ours/CLIP andpastanalysis protected. AP0/Kstep0/no runtime changes. Source-only fit, deployment-visible features, separateholdoutdirectioncommit all preserved.

Local Git auto-GC filled D disk before mathcommit; it exited with out-of-space, leaving two read-only incomplete temp packs. Normal deletion failed; automaticapprovalreview rejected force deletion with 'blocked by policy', so no alternate deletion was attempted. Available disk space subsequently sufficed for small commits; one-off gc.auto=0/maintenance.auto=false avoided recurrence without changing repo config. All experiment archives/logs preserved; no model rerun. Project-local progress and .autodl receipts preserve this recovery history.

Report research_log/T033A_report.md; representation/model/directions/summary/integrity/per_episode.tsv andallrawruns retained. Annotation subsets stay underremoteproject research_log/T033A withcohortSHA pins. Futurefreshadditional_source T033A_train_cohort.json preserves3791cumulativeprior/reserved plusval/evaluation.

Final **{s['disposition']}**, NEEDS_REVIEW. No pooling/PCA/class/CLIP/native/ridge/MLP/threshold rescue; no runtime/AP extension authorized.
'''
    (ROOT/'research_log/T033A_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'verdict':verdict,'gates':s['gates'],'report_sha256':sha(ROOT/'research_log/T033A_report.md')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directions-run',required=True);p.add_argument('--holdout-run',required=True)
    a=p.parse_args();main(a.directions_run,a.holdout_run)
