# T016-A pre-outcome differential calibration plan

R027/1107946 accepts T015-A and closes the uncalibrated fixed-pseudo spatial branch.
This task uses only frozen arrays. Commit this plan, input hashes and permutation
schedule before any fit or new scientific summary. No model/ISP/CLIP/optimizer calls.

## Frozen data and reuse

T015-A authoritative records SHA256 ae01cea586d0585b510293460ce174e2cf65350cc3d2ba3bd364c24b7de55b12.
Pin T015 records/environment/summary and A1 records in T016A_input_manifest.json.
Use saved task/pseudo s,d vectors; A1 reference regions remain the provenance source.
Verified32originalorderedepisodes,16imagepairs,clean then corrupted perpair,4blocks.
Foldf holds indices8f..8f+7 (4pairs); training is sorted complement24episodes/12pairs.
No shuffle/filter/regroup. Task arrays enter training coefficients only via trainindices;
heldout taskvectors are used solely for metric evaluation.

Reuse linear_capacity.pair_rows for pair expansion and differential_subspace.cosine,
describe/write_json, plus spatial_action.EPS=1e-12. Adapt only the sample-size-specific
fold/schedule and exact requested diagonal fit/norm matching; no changes to old modules.
Baseline unchanged T015 release: differential_subspace+linear_capacity tests,
13passed1.50s. Existingfloat64 NumPyCPU runtime; GPU available for applicable regression
checks peruserpreference.16D arrays do not benefit from moving this audit toGPU.

## Fit, projection and numerical receipts

Each of4trainingfolds fits a=sum(dp*dt)/(sum(dp**2)+EPS), coordinatewise, nointercept.
No centering/ridge/clipping/selection/rankchange/optimizer. Save numerator, denominator,
all8coefficients and train/testindices. qdiff=concat(a*dp,-a*dp),
qdiff_nm=norm(pdiff)*qdiff/(norm(qdiff)+EPS),q=pshared+qdiff_nm.
Never renormalize a second time or alter EPS. Keep pshared exactly as saved.
Save all raw/calibrated vectors, cosines, signs, norms and shared/differential dots.
Zero-norm cosines are null with explicit counts, never favorable replacements.
Verify differential and full pseudo norm preservation using predeclared
abs(new-old)<=1e-10+1e-8*old, both observed and null evaluations. Record errors/bounds
and EPS-induced differential shrinkage norm(pdiff)*EPS/(norm(qdiff)+EPS). These
formulas are not exactly norm preserving with finite EPS; no claim of bitwise equality.
If this numerical check fails, preserve the receipt and report blocker, no retuning.

Heldout C_diff_cal=dot(tdiff,qdiff_nm)/(norm(q)+EPS),
D_cal=dot(t,q)/(norm(q)+EPS),Delta_D_cal=D_cal-savedD_spatial.
Rawcos/dot/Cdiff/D values retained; all32heldout outputs orderedoriginally.
Summaries overall/clean/corrupt/4blocks/originalcase labels; no new groups.
Coefficient stability: percoordinate values, signs, mean/std/min/max and signcounts;
no retrospective feature-selection threshold.

## Matched null committed before outcomes

Exactly256 permutations generated with NumPy default_rng(20260913). Iteration order:
permutation0..255, within each folds0..3; each draw rng.permutation(12).
Schedule committed as T016A_permutation_schedule.json. For eachfold permute only
training taskimagepairs relative to fixed pseudo rows, preserving clean/corruptpositions.
Holdout unchanged. Fit same4diagonalmaps andapplysame normmatch for eachnull.
Persist all1024nullcoefficients, mappings, heldoutpredictions/per-episode metrics
and poolednull outcomes. Total1028closedformfoldfits; zero optimizersteps.
Four nulldistributions: medianDeltaD, positiveDeltaDcount, positiveCdiff_calcount,
mediancalibratedcosdiff. Quantile95 uses np.quantile(...,.95,method='linear');
correctedtail=(1+count(null>=observed))/257; reporttiesandstrictpercentiles.
Null median cosine excludes onlyexplicit zero-norm nullvalues; validcounts retained.

## Literal R027 gate

Cdiff_cal positive>=20overall,>=10corrupted, overallmedian>0,>=3positiveblockmedians;
DeltaD_cal same20/10/median/3blocks rules. ObservedmedianDeltaD strictly>null95median
andpositiveDeltaDcount strictly>null95count. All10flags mustpass. No thresholdsweakened.
PASS onlysource-cohortcalibratability, thenfutureexplicitnewcohortreplication; FAILclose
diagonal/source-calibration rescue. Neither outcome authorizes deployment or newdata.

## Increment and completion

One analysis module, synthetic tests and report renderer. Tests: known diagonal/no
intercept coefficients, heldouttarget isolation, pairpermutation structure/determinism,
normpreservation inclzero/one-region cases, knownproductivity and exactgate/tailboundaries.
Focused thenfullregression thenoneactualarrayaudit. Persistreport/raw/hashmanifest,
appendCODEXmailbox,commit/push,mirrorA6000 andstopNEEDS_REVIEW.
No newcohort, regionalobjective/CLIP redesign, spatialfinite-step/AP, masks/regions,
source/meta-training,predictorredesign,FCOS/SSD,gate/dose tuning ordeploymentcodechange.
