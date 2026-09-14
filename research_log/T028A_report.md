# T028-A — NEEDS_REVIEW; fixed affine gradient-state capacity FAIL

R043/41cc666 executed. Decision: close_fixed_affine_gradient_state_representation. This is a source-supervised capacity audit; no deployment/AP/K-step/FCOS/SSD result or authorization.

## Frozen cohort, model and label ordering

Plan/split d3dcde6 precommitted 180 fresh train2017 images selected by sha256('20260928:'+ID), excluding2711cumulative source/memory/audit IDs and5000val IDs. First120images/240episodes train, remaining60images/120episodes holdout. Cyclic gamma_s2/contrast_s2/color_cast_s2 assignment gives40/40/40train,20/20/20holdout; each of four15image holdoutblocks contains5perfamily. Every image contributes clean then assignedcorruption; all pairs stay in the same partition. Cohort eligibility uses inherited readable-image and valid-noncrowd-GT-box criteria before outcomes, with no outcome-based replacement. Cohort SHA256 9eb17718bf5a02f645c082eb8a025af59c2c9261f62c110f908c65d81e429119.

Candidate code3614d6e, release20260914-082414-taisp-t028a-candidate. ALL360 label-free records committed/pushed08dd3fd before post-lock reference/fit codeb502d15 ran. Reference verifies363candidate files and cohortSHA BEFORE annotation loading. Candidate transitive import check and runtime module list exclude oracle/reference/source-meta/memory. Full supports and hashes retained.

Launcher metadata correction: candidate TAISP_SOURCE_REVISION was accidentally literal `placeholder`. Original environment/meta/run.sh remain intact; candidate_provenance.json records exact release/code hashes matching3614d6e, independently verified before reference. No model or outcome rerun was used to repair metadata. Reference has correct source revisionb502d15. Protected/prior/new remote code41pins match;35protected/prior files unchanged locally.

## Literal representation and fit

At global ISP identity, unchanged score>=.50/stabletop20 FasterRCNN fixed-ROI pseudo gradient g_p, and unchanged generic frozen CLIP displacement gradient g_c. Exactly21features: g_p/(norm+eps)[8],g_c/(norm+eps)[8],logbothnorms,cosine,pseudo_loss,CLIP_loss;eps1e-12. No flip/support scalars/GT/corruption hints/post-update data or alternative objective. Case/partition metadata only identifies records and fixed evaluation/null strata, never enters21D features. Generic CLIP displacement loss at identity is retained as computed, without forcing mathematical zeros. Observed360loss range [-0.00024456685059703887,0.00016188467270694673]; no exactzero loss and no zero-std train coordinate. The predeclared zero-std rule remains unchanged.

Only after complete feature lock, unchanged original native four-loss annotated task gradient t_s gives S_orig=dot(t_s,g_p)/(norm(g_p)+eps), y=+1 iffS_orig>0 else-1; exactzero remains negative, not discarded. Acceptedfloat32 common8columnISPJVP/float64 reductions and inheritedglobalreverseparity preserved.

Train240-only population mean/std(ddof0), exactzerostd maps coordinate0, interceptlast. One float64 thin-SVD pseudoinverse, tol=eps64*max(Z.shape)*smax; retain singularvalues>tol. No regularizer/model/thresholdsearch. Literal estimator saved before holdout scoring; trust r>0 only. Rank=22/22; tol=1.3483126631998407e-12; retained condition=47988741.60477236; reconstruction relativeerror=1.453812170294546e-10; zero-std coordinates=[].

## Holdout gate

|Metric|Overall120|Corrupted60|Required|
|---|---:|---:|---|
|AUROC|0.52625|0.6576704545454546|>=.70 / >=.65|
|Trusted coverage|0.6583333333333333|0.7166666666666667|[.25,.80] both|
|Precision gain|0.029535864978903037|0.08062015503875974|>=.10 both|
|Trusted median utility|0.03801710587715246|0.09037239109732108|>0 both|

Positive precision-gain blocks 2/4, required>=3. Ungated prevalence overall/corrupted=0.6666666666666666/0.7333333333333333; trusted positive fraction=0.6962025316455697/0.813953488372093. Clean trusted coverage=0.6. Gate conjunction FAIL; full boolean ledger in summary.json. No diagnostic substitutes for a failed criterion. The high retained condition number and two near-dependent singular directions are recorded without changing the prescribed tolerance, deleting coordinates or fitting a regularized model.

## Pair-preserving null

128fixed PCG64(20260928) iterations, sorted family strata, permute complete(clean,corrupted) training-label pairs withinfamily, features/trainstandardization/SVD unchanged. Every permutation index, nullweight and120holdout scores retained. No secondseed. Average ranks for AUROC ties; undefined AUROC automatically fails. Null95th percentile uses linear interpolation; observed percentile=fraction(null<observed); correctedtail=(1+count(null>=observed))/129.

Overall observed/q95=0.52625/0.612984375; percentile=0.6796875; correctedtail=0.32558139534883723.
Corrupted observed/q95=0.6576704545454546/0.6713778409090909; percentile=0.9140625; correctedtail=0.09302325581395349.

Summary retains overall,clean,corrupt,eachcorruption,fourblocks: prevalence,AUROC,Spearman,coverage,trusted/untrustedpositivefractions,precisiongain,utility mean/median/linearquantiles and nonzero-gradient counts. Every train/holdout episode and21Dfeature is in per_episode.tsv; train scores are intentionally blank because no training metric selects the model. Original360reference gradients and labels remain raw receipts.

## Integrity and execution

All360reference integrity and candidate isolation pass. Zero pseudo/CLIP gradients=2/0, retained. Maximum relativeL2 pseudo/CLIP/reference parity=2.8866419881397285e-06/1.9857948644072256e-06/2.017280702514556e-06; partition maxerror=0.0.

FasterRCNN state73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf, CLIP state6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c unchanged, frozen/eval/parametergradNone. Same generic prompts/CLIP revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. All model computation on A6000 CUDA0, torch2.4.0+cu121; CPU only manifest/report and mandated NumPyfloat64SVD/nulls. No detector/CLIP optimization or test-label adaptation.

Baseline12pass2.82s;candidatefocused12pass5.33s;finalfocused17pass5.15s;full236pass11skip4warnings12.56s. ExistingNVML/protobuf warnings retained. Candidate20260914-082515-taisp-t028a-candidates360 79.12334913399536s; reference+fit20260914-083057-taisp-t028a-reference360, reference 54.831734517996665s. Both jobs exit0; rawarchives verified before extraction, all per-file digests verified.

Stop NEEDS_REVIEW. No retraining/extrafeatures/splitchange/moredata/threshold/model/holdoutrescue or T028-B without new research. Future fresh cohorts use T028A_train_cohort.json additional_source, cumulative2711+180=2891source IDs plus5000val exclusion. Exact estimator, source stats, all360candidates/references and128nulls preserved.
