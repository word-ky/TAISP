"""R050 report from locked receipts; stdlib only, no fitting or model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_log/T032A'
RUNS=ROOT/'research_log/remote_runs'
CAND=RUNS/'20260914-181428-taisp-t032a-candidates480/artifacts/candidate'
TRAIN=RUNS/'20260914-182353-taisp-t032a-train360-maps/artifacts/train'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(holdout_run):
    hold=RUNS/holdout_run/'artifacts/holdout'
    manifests={}
    for name,folder in [('candidate',CAND),('train',TRAIN),('holdout',hold)]:
        m=read(folder/'sha256.json')
        for f,h in m.items():assert sha(folder/f)==h,f
        manifests[name]={'file_count':len(m),'records_sha256':sha(folder/'records.json'),
                         'manifest_sha256':sha(folder/'sha256.json')}
    cs,tr,hr=[read(p/'records.json') for p in [CAND,TRAIN,hold]]
    assert (len(cs),len(tr),len(hr))==(480,360,120)
    bykey={(r['image_id'],r['case']):r for r in cs}
    assert all(r['g_hard']==bykey[r['image_id'],r['case']]['g_hard'] for r in tr+hr)
    assert all(r['integrity_passed'] and r['rng']['restored'] for r in cs+tr+hr)
    assert not {r['image_id'] for r in tr}&{r['image_id'] for r in hr}
    router=read(OUT/'router.json');maps=read(TRAIN/'maps.json')
    assert [r['route'] for r in tr]==router['assignments']
    assert maps['router_sha256']==sha(OUT/'router.json') and maps['svd_calls']==2
    assert maps['holdout_task_gradients_computed']==0
    routed=read(hold/'holdout_routes_before_oracle.json')
    assert len(routed)==120 and all((a['episode_index'],a['route'],a['g_route'])==
        (b['episode_index'],b['route'],b['g_route']) for a,b in zip(routed,hr))
    ml=read(ROOT/'research_log/T032A_map_lock.json');pf=read(hold/'preflight.json')
    assert ml['files']==read(TRAIN/'sha256.json')
    assert pf['map_lock']['map_commit']==ml['map_commit']
    checks={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
        for f,h in read(ROOT/'research_log/T032A_all_code_pins.json').items()}
    assert all(checks.values()) and all(read(OUT/'remote_code_checks.json').values())
    integrity={'raw_manifests':manifests,'all960_integrity_rng':True,'unchanged_hard_vectors':True,
        'train_holdout_disjoint':True,'fixed_holdout_routes_match':True,'code_checks':checks,
        'map_commit':ml['map_commit'],'maps_sha256':sha(TRAIN/'maps.json'),
        'router_sha256':sha(OUT/'router.json'),'norm_preservation_max_abs':max(r['norm_preservation_error'] for r in hr),
        'preflight':pf,'AP_calls':0,'candidate_recomputed':False}
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    for f in ['summary.json','preflight.json','completion.json','holdout_routes_before_oracle.json']:
        (OUT/f).write_bytes((hold/f).read_bytes())
    (OUT/'maps.json').write_bytes((TRAIN/'maps.json').read_bytes())
    fields=['episode_index','image_id','case','block','route','S_hard','S_route','Delta','norm_hard','norm_cal',
            'cos_hard_task','cos_cal_task','norm_preservation_error','integrity_passed']
    with (OUT/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
        w.writerows({k:r[k] for k in fields} for r in hr)
    s=read(hold/'summary.json');verdict='PASS' if s['passed'] else 'FAIL'
    groups={'overall':s['overall'],'clean':s['clean'],'corrupt':s['corrupt'],**s['conditions'],
            **{'block'+k:v for k,v in s['blocks'].items()},**{'cluster'+k:v for k,v in s['clusters'].items()}}
    lines=[];coords=[]
    for name,g in groups.items():
        d=g['distributions'];cal=d.get('S_route',d.get('S_cal'))
        lines.append(f"|{name}|{g['n']}|{d['S_hard']['positive']}|{cal['positive']}|{d['Delta']['positive']}|{d['Delta']['median']}|{d['Delta']['mean']}|{d['cos_hard_task']['median']}|{d['cos_cal_task']['median']}|")
        for key,c in g['coordinates'].items():coords.append(f"|{name}|{key}|{c['max_energy_coordinate']}|{c['max_energy_share']}|")
    matrix_lines=[]
    for k,m in enumerate(maps['experts']):
        matrix_lines.append(f"Cluster{k}: train count{m['train_episode_count']}; zero unit hard/task{m['zero_unit_hard']}/{m['zero_unit_task']}; determinant{m['determinant']}; orthogonality maxabs{m['orthogonality_max_abs']}; singular spectrum{m['singular_values']}; SVD calls{m['svd_calls']}.")
    gate_lines='\n'.join(f'|{k}|{v}|' for k,v in s['gates'].items())
    report=f'''# T032-A / R050 — scientific {verdict}; NEEDS_REVIEW

The frozen spherical K=2 router plus two O(8) source experts {'passes the predeclared holdout capacity conjunction; stop for research review before runtime work' if s['passed'] else 'fails the predeclared holdout conjunction. Close this exact two-regime transport family without rescue tuning'}. This is only a first-order source gradient capacity audit. No AP, K-step adaptation or deployment performance is claimed. T031 remains closed; none of its records/maps/outcomes entered this fit.

## Frozen stages, provenance and label separation

Research R050 195bfc8 / latest8d45ec2. Fresh cohort/split/protocol **b0ce54b**; candidate code is unchanged R049 orthogonal_candidates.py. New router code/tests **d71bd34**; reference integration/tests **0d22b7d**, both committed before real router/source fits. Candidate480 commit **5ae8bb68087d9739b5f69c7524a9f4b75effc829**, descriptor **0401c8e** BEFORE router fitting. Router+assignments commit **844bc0d0a8dbaf2f24b11ab35cbddc50aad6cb98**, descriptor **2742f87**, BEFORE annotations/task gradients. Both source maps+train references commit **{ml['map_commit']}** BEFORE holdout task gradients. Map descriptor **57bf9b6** is committed before the holdout launch; its SHA is {sha(ROOT/'research_log/T032A_map_lock.json')}. Exact map-lock receipt: {json.dumps(pf['map_lock'])}.

240 completely fresh COCO train2017 images, selectionseed20261002, excluding3251 prior/reserved IDs (including T031) plus5000val and1400evaluation IDs. Existing prepare_t018a eligibility/order; cyclic frozen severity2 family assignment. First180 images train360episodes (60/family), last60 holdout120episodes (20/family); clean then assignedcorruption perimage; four holdout blocks15pairs/5perfamily. Cohort SHA{sha(ROOT/'research_log/T032A_train_cohort.json')}. Outcome-free preparation parsed annotations to screen valid boxes and write separate train/holdout subsets. Candidate manifest contains no annotation paths; no candidate oracle/reference import. Train process opened only train annotations after the router commit. Holdout process wrote ALL120 routes/transformed gradients before lazyoracle imports and task-gradient computation.

Router fit accepts only360 ordered train hard-gradient vectors. It receives no annotation, task outcome, image/family/case metadata or holdout vectors. For nonzero g, u=g/(norm+1e-12); exactzero routes0 and stays counted. First nonzero initializes centroid0; smallest cosine initializes centroid1, ties earliest. Maximum-cosine assignments tie0; centroids normalized arithmetic means; unchanged assignments or20iteration cap; no restarts or K search. Empty/zero-mean centroid collapse was predeclared before outcomes. Router iterations={router['iterations']}, converged={router['converged']}, initial indices={router['initial_indices']}, counts={router['counts']}, zero count={router['zero_count']}, counttrace={router['count_trace']}. Both>=72; no router-collapse stop. Router SHA{sha(OUT/'router.json')}, candidate-lock SHA{sha(ROOT/'research_log/T032A_candidate_lock.json')}.

Candidate run20260914-181428-taisp-t032a-candidates480, release20260914-181346-taisp-t032a-candidates, sourceb0ce54b,480episodes/{read(CAND/'completion.json')['seconds']}s. Router command and outputs recorded in project progress; fitted once in release20260914-181842-taisp-t032a-reference. Train run20260914-182353-taisp-t032a-train360-maps, source2742f87,360episodes/{read(TRAIN/'completion.json')['seconds']}s. Holdout run{holdout_run}, source57bf9b6,120episodes/{read(hold/'completion.json')['seconds']}s. Train/holdout use the same frozen release20260914-181842-taisp-t032a-reference; exact commands/timestamps/source revisions retained in raw runs. A6000CUDA0 for all model forward/backward/JVP; CPUfloat64 only router, two8x8SVD and summaries. Same frozen Faster R-CNN, 8-D identityISP, score>=.50 stabletop20 confidence-weighted hard objective and taskreferenceseed20260913. Source stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf unchanged.

## Source expert diagnostics

Exactly one Procrustes SVD per frozen train cluster, two total; unit normalize iff norm>1e-12 elsezero; C=sum u_t u_h.T; R=U@Vt, fullO(8), no determinant correction/bias/ridge/weights. Stored transformed gradients are not renormalized. Maps SHA{sha(TRAIN/'maps.json')}; full matrices/C/spectra in T032A/maps.json. Maximum holdout norm-preservation absolute error={integrity['norm_preservation_max_abs']}.

{chr(10).join(matrix_lines)}

## Frozen R050 gate

Required S_routepositive>=80/120overall,45/60corrupt; Deltapositive>=72/120overall,36/60corrupt; mean+medianDeltaoverall/corrupt>0; >=3/4positive block medians; cleanmedian>=0; bothholdoutroutes>=12; all integrity. Exactzero episodes retained in denominators. S_route=<task,g_route>/(norm(g_route)+1e-12), Delta=S_route-S_hard. Summary reuses R049 distribution field names S_cal/g_cal/cos_cal_task as aliases for S_route/g_route/routed cosine; no old T031 map is evaluated.

|Gate|Pass|
|---|---|
{gate_lines}

|Group|N|S_hard>0|S_route>0|Delta>0|MedianDelta|MeanDelta|Median cos(h,task)|Median cos(route,task)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Route counts by scope (diagnostics only; never routing inputs): {json.dumps(s['route_counts'])}.

## Coordinate concentration

Full8-coordinate mean/std/meanabs/RMS/energy share by scope/cluster are in summary.json. Energy share=sum g_j²/sum_all g²; indices zero-based. No diagnostic-driven refit.

|Group|Gradient|Max-energy coordinate|Max share|
|---|---|---:|---:|
{chr(10).join(coords)}

## Integrity, tests and artifacts

All480candidate+360train+120holdout integrity/RNG/state/parity receipts pass. Stored hard vectors unchanged; pair-preserving train/holdout disjoint; train assignments exactly router lock; holdout routes/transformed gradients exactly equal pre-oracle file. All{len(checks)} source/code pins match locally/remotely, including61protected/prior files. Candidate/router/source-model/deployment code not changed after outcomes. AP0; no K-step/runtime/FCOS/SSD. Raw hashes: {json.dumps(manifests)}.

Baseline7passed3warnings7.28s; routerfocused4passed3.09s; router+referencefocused7passed2warnings6.36s; full278passed11skipped4warnings33.75s on server. Tests cover deterministic initialization/ties/zeros/minimum cluster boundary, no detector/oracle import, original mean/count gates, >=12routecoverage and empty-cluster diagnostics. Initial test incorrectly assumed package import excludes torch; existing taisp.__init__ imports ISP/torch. Corrected only that test to enforce the actual detector/oracle boundary; no implementation or task constraint changed. Candidate manifest download initially failed due missing newlocaldirectory; recreated exact JSON from SHA-verified cohort, no model rerun. Train archive SCP stalled at229376bytes; stopped only matching transfer PID54660, existing legacy fallback completed; full archive SHA24cb957225aa68e438bd0a260e45775fae55160b17ae3102dfc4786f22ff6ad9 and all365file hashes verified before mapscommit. No training rerun.

New code: taisp/analysis/gradient_regime_router.py, gradient_regime_reference.py; tests/test_gradient_regime_router.py, test_gradient_regime_reference.py; this stdlib report renderer under research_log/T032A. Existing R049 verifier/Procrustes/summary algebra and exact task-gradient loop reused; donor same repository under existing license. New logic is restricted to router, lock ordering, two experts and cluster diagnostics. Detector/ISP/tta/current-Ours/CLIP/deployment unchanged.

Report research_log/T032A_report.md; T032A/router.json,maps.json,summary.json,per_episode.tsv,integrity.json; frozen descriptors and allrawJSON/logs under research_log/remote_runs. Annotation subsets remain under remote project research_log/T032A, hashes pinned in cohort. No raw artifacts deleted. Future fresh selection additional_source T032A_train_cohort.json preserves3491 cumulative prior/reserved IDs plusval/evaluation.

Final disposition **{s['disposition']}**. Stop NEEDS_REVIEW. No alternative clustering/K, learned/soft/family routing, map tuning, CLIP/native inputs or AP selection authorized.
'''
    (ROOT/'research_log/T032A_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'verdict':verdict,'gates':s['gates'],'report_sha256':sha(ROOT/'research_log/T032A_report.md')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--holdout-run',required=True);a=p.parse_args();main(a.holdout_run)
