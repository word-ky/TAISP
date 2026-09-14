"""Summarize immutable T028-A candidate/reference/fit receipts; no model calls."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];out=root/'research_log/T028A'
    croot=root/'research_log/remote_runs/20260914-082515-taisp-t028a-candidates360/artifacts/candidates'
    rroot=root/'research_log/remote_runs/20260914-083057-taisp-t028a-reference360/artifacts/reference'
    froot=rroot.parent/'fit'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in [croot,rroot,froot]:
        for f,h in read(folder/'sha256.json').items():assert sha(folder/f)==h
    cs,rs=read(croot/'records.json'),read(rroot/'records.json')
    s,e=read(froot/'summary.json'),read(froot/'estimator.json');hs=read(froot/'holdout_records.json')
    score={r['episode_index']:r['score'] for r in hs};table=[]
    for c,r in zip(cs,rs,strict=True):
        assert (c['image_id'],c['case'])==(r['image_id'],r['case'])
        table.append({k:r[k] for k in ['episode_index','image_id','case','partition','block','S_orig','y','integrity_passed']} |
            {'holdout_score':score.get(r['episode_index']),'norm_pseudo':c['norm_pseudo'],'norm_clip':c['norm_clip'],
             'support_count':c['pseudo']['support_count'],'pseudo_loss':c['pseudo']['loss'],'clip_loss':c['clip_loss']} |
            {f'feature_{i}':v for i,v in enumerate(c['features'])})
    with (out/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');w.writeheader();w.writerows(table)
    for f in ['summary.json','estimator.json']:(out/f).write_bytes((froot/f).read_bytes())
    plan=read(root/'research_log/T028A_plan.json')
    checks={f:hashlib.sha256((root/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in plan['protected_and_prior_LF'].items()}
    assert all(checks.values()) and all(read(out/'remote_code_checks.json').values())
    integrity={'local_protected_and_prior':checks,'remote':read(out/'remote_code_checks.json'),
        'all360_reference_integrity':all(r['integrity_passed'] for r in rs),
        'all360_candidate_isolation':all(c['isolation'] for c in cs),
        'max_pseudo_parity_relative_l2':max(c['pseudo']['parity']['relative_l2'] for c in cs),
        'max_clip_parity_relative_l2':max(c['clip_parity']['relative_l2'] for c in cs),
        'max_reference_parity_relative_l2':max(r['reference_checks']['parity']['relative_l2_error'] for r in rs),
        'max_partition_error':max(r['reference_checks']['partition']['max_absolute_error'] for r in rs),
        'zero_pseudo':sum(c['norm_pseudo']==0 for c in cs),'zero_clip':sum(c['norm_clip']==0 for c in cs),
        'cohort_sha256':sha(root/'research_log/T028A_train_cohort.json'),
        'candidate_records_sha256':sha(croot/'records.json'),'reference_records_sha256':sha(rroot/'records.json')}
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    a,b=s['overall'],s['corrupt'];null=s['null'];positiveblocks=sum(v['precision_gain'] is not None and v['precision_gain']>0 for v in s['blocks'].values())
    lines=[]
    for label,key,required in [('AUROC','AUROC','>=.70 / >=.65'),('Trusted coverage','trusted_coverage','[.25,.80] both'),('Precision gain','precision_gain','>=.10 both')]:
        lines.append(f'|{label}|{a[key]}|{b[key]}|{required}|')
    lines.append(f"|Trusted median utility|{a['trusted']['median']}|{b['trusted']['median']}|>0 both|")
    result='PASS' if s['passed'] else 'FAIL'
    report=f'''# T028-A — NEEDS_REVIEW; fixed affine gradient-state capacity {result}

R043/41cc666 executed. Decision: {s['disposition']}. This is a source-supervised capacity audit; no deployment/AP/K-step/FCOS/SSD result or authorization.

## Frozen cohort, model and label ordering

Plan/split d3dcde6 precommitted 180 fresh train2017 images selected by sha256('20260928:'+ID), excluding2711cumulative source/memory/audit IDs and5000val IDs. First120images/240episodes train, remaining60images/120episodes holdout. Cyclic gamma_s2/contrast_s2/color_cast_s2 assignment gives40/40/40train,20/20/20holdout; each of four15image holdoutblocks contains5perfamily. Every image contributes clean then assignedcorruption; all pairs stay in the same partition. Cohort eligibility uses inherited readable-image and valid-noncrowd-GT-box criteria before outcomes, with no outcome-based replacement. Cohort SHA256 {integrity['cohort_sha256']}.

Candidate code3614d6e, release20260914-082414-taisp-t028a-candidate. ALL360 label-free records committed/pushed08dd3fd before post-lock reference/fit codeb502d15 ran. Reference verifies363candidate files and cohortSHA BEFORE annotation loading. Candidate transitive import check and runtime module list exclude oracle/reference/source-meta/memory. Full supports and hashes retained.

Launcher metadata correction: candidate TAISP_SOURCE_REVISION was accidentally literal `placeholder`. Original environment/meta/run.sh remain intact; candidate_provenance.json records exact release/code hashes matching3614d6e, independently verified before reference. No model or outcome rerun was used to repair metadata. Reference has correct source revisionb502d15. Protected/prior/new remote code41pins match;35protected/prior files unchanged locally.

## Literal representation and fit

At global ISP identity, unchanged score>=.50/stabletop20 FasterRCNN fixed-ROI pseudo gradient g_p, and unchanged generic frozen CLIP displacement gradient g_c. Exactly21features: g_p/(norm+eps)[8],g_c/(norm+eps)[8],logbothnorms,cosine,pseudo_loss,CLIP_loss;eps1e-12. No flip/support scalars/GT/corruption hints/post-update data or alternative objective. Case/partition metadata only identifies records and fixed evaluation/null strata, never enters21D features. Generic CLIP displacement loss at identity is retained as computed, without forcing mathematical zeros. Observed360loss range [-0.00024456685059703887,0.00016188467270694673]; no exactzero loss and no zero-std train coordinate. The predeclared zero-std rule remains unchanged.

Only after complete feature lock, unchanged original native four-loss annotated task gradient t_s gives S_orig=dot(t_s,g_p)/(norm(g_p)+eps), y=+1 iffS_orig>0 else-1; exactzero remains negative, not discarded. Acceptedfloat32 common8columnISPJVP/float64 reductions and inheritedglobalreverseparity preserved.

Train240-only population mean/std(ddof0), exactzerostd maps coordinate0, interceptlast. One float64 thin-SVD pseudoinverse, tol=eps64*max(Z.shape)*smax; retain singularvalues>tol. No regularizer/model/thresholdsearch. Literal estimator saved before holdout scoring; trust r>0 only. Rank={e['rank']}/22; tol={e['tol']}; retained condition={e['effective_condition']}; reconstruction relativeerror={e['reconstruction_relative_error']}; zero-std coordinates={[i for i,v in enumerate(e['std']) if v==0]}.

## Holdout gate

|Metric|Overall120|Corrupted60|Required|
|---|---:|---:|---|
'''+'\n'.join(lines)+f'''

Positive precision-gain blocks {positiveblocks}/4, required>=3. Ungated prevalence overall/corrupted={a['prevalence']}/{b['prevalence']}; trusted positive fraction={a['trusted']['positive_fraction']}/{b['trusted']['positive_fraction']}. Clean trusted coverage={s['conditions']['clean_s0']['trusted_coverage']}. Gate conjunction {result}; full boolean ledger in summary.json. No diagnostic substitutes for a failed criterion. The high retained condition number and two near-dependent singular directions are recorded without changing the prescribed tolerance, deleting coordinates or fitting a regularized model.

## Pair-preserving null

128fixed PCG64(20260928) iterations, sorted family strata, permute complete(clean,corrupted) training-label pairs withinfamily, features/trainstandardization/SVD unchanged. Every permutation index, nullweight and120holdout scores retained. No secondseed. Average ranks for AUROC ties; undefined AUROC automatically fails. Null95th percentile uses linear interpolation; observed percentile=fraction(null<observed); correctedtail=(1+count(null>=observed))/129.

Overall observed/q95={null['overall']['observed']}/{null['overall']['q95']}; percentile={null['overall']['percentile_strict']}; correctedtail={null['overall']['corrected_one_sided_tail']}.
Corrupted observed/q95={null['corrupt']['observed']}/{null['corrupt']['q95']}; percentile={null['corrupt']['percentile_strict']}; correctedtail={null['corrupt']['corrected_one_sided_tail']}.

Summary retains overall,clean,corrupt,eachcorruption,fourblocks: prevalence,AUROC,Spearman,coverage,trusted/untrustedpositivefractions,precisiongain,utility mean/median/linearquantiles and nonzero-gradient counts. Every train/holdout episode and21Dfeature is in per_episode.tsv; train scores are intentionally blank because no training metric selects the model. Original360reference gradients and labels remain raw receipts.

## Integrity and execution

All360reference integrity and candidate isolation pass. Zero pseudo/CLIP gradients={integrity['zero_pseudo']}/{integrity['zero_clip']}, retained. Maximum relativeL2 pseudo/CLIP/reference parity={integrity['max_pseudo_parity_relative_l2']}/{integrity['max_clip_parity_relative_l2']}/{integrity['max_reference_parity_relative_l2']}; partition maxerror={integrity['max_partition_error']}.

FasterRCNN state73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf, CLIP state6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c unchanged, frozen/eval/parametergradNone. Same generic prompts/CLIP revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. All model computation on A6000 CUDA0, torch2.4.0+cu121; CPU only manifest/report and mandated NumPyfloat64SVD/nulls. No detector/CLIP optimization or test-label adaptation.

Baseline12pass2.82s;candidatefocused12pass5.33s;finalfocused17pass5.15s;full236pass11skip4warnings12.56s. ExistingNVML/protobuf warnings retained. Candidate20260914-082515-taisp-t028a-candidates360 {read(croot/'completion.json')['seconds']}s; reference+fit20260914-083057-taisp-t028a-reference360, reference {read(rroot/'completion.json')['seconds']}s. Both jobs exit0; rawarchives verified before extraction, all per-file digests verified.

Stop NEEDS_REVIEW. No retraining/extrafeatures/splitchange/moredata/threshold/model/holdoutrescue or T028-B without new research. Future fresh cohorts use T028A_train_cohort.json additional_source, cumulative2711+180=2891source IDs plus5000val exclusion. Exact estimator, source stats, all360candidates/references and128nulls preserved.
'''
    (root/'research_log/T028A_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
