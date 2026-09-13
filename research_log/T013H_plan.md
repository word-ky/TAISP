# T013-H pre-outcome source replication plan

R022/4802c88 in CHATGPT_TO_CODEX.md accepts/closes T013-G. Commit this plan and
the selected manifest before any outcome-bearing model call. One source diagnostic,
no predictor training. No adaptive cohort replacement or repeated outcome averaging.

## Cohort and exclusions

Seed20260913. Enumerate existing readable train2017 JPEGs with valid noncrowd source
boxes (w,h,area>0), using the same installed annotation/image roots and eligibility
as prepare_t013b.py. Exclude original IDs65088,426525,541157,129068; all evaluated IDs
in T002–T012 environment receipts (exclusion_ledger.json); AND all5000 full official
COCO val2017 image IDs. Verify evaluation-ledger IDs are included in that fullval set.
Source annotationSHA610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d;
fullval annotationSHAe8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f.
No dataset download or permission change. Deterministic selection: rank eligible IDs
by (SHA256(UTF8('20260913:'+decimalID)).hexdigest(),ID), take first32 in rankedorder.
Record allselected IDs/JPEGhashes/annotations, inputhashes andeligible/excludedcounts.
Commit resulting research_log/T013H_train_cohort.json before any gradient/model call.

Corruptions cycle selectedorder gamma_s1,gamma_s2,contrast_s2,color_cast_s2 (8each).
Each image emits clean thencorrupt, exactly64records. Blocks are consecutive8images,
fourblocks, eachcontaining2images percondition. No post-outcome filtering/replacement.

## Frozen collection protocol

Same originalzerohead checkpoint fromT013C:
a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120.
Same sourceweightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384;
CLIPSHAa63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f,
openai/clip-vit-base-patch32 revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.
Seed20260913 andconstructororder source,CLIP,ISP,predictor; exactcheckpointverification.
Normalfloat32 CUDA path deterministicFalse,cudnn.benchmarkFalse,CUBLASconfigunset,
threads1. Accepted8D ISP,K3/innerlr.1/eps1e-12, sourceoracle samplingseed20260913,
CLIP prompts/preprocess/losses unchanged. Explicitwrapper.eval beforeearlyisolation.
Pseudo supports fromoriginal image sourceprediction exactlyonce pernewepisode,
scorethreshold.5/top20, detached; retainboxes/labels/scores. Sourceannotations onlyouter.

Exactlyone primarygradient trajectory perepisode fromfresh originalpredictor copy.
Reuse source_meta_smoke/load_episodes and gradient_conflict.evaluate(gradients=True).
Capture h using a feature forward hook during that same predictor forward; no second
feature forward. Retain h,g,per-episodeheadgradients,loss/components,phi0/phi3,saturation,
support/empty status, block/order andisolation. Saveeachrecord immediately; no optimizer.
Check source/CLIP parameters/buffers/gradNone, original/copy/ISP unchanged throughout.
Retain exactfailure andstop if primaryexecution blocked; do not change kernels/tolerances.

## Offline replication

Float64CPU fromsaved64records only. Overall andeach16episodeblock use T013G definitions:
mu norm, feature raw/centeredfraction,SVD/participationrank, d/rho (denominator median
distance toothercleanimages within that scope), A,C,gbar,rC,rCout,cos/cross, fullimplied
eta=.001 output and3component raw/centeredenergies/allcross terms. Also clean/corrupt
mean-gradientcosine. No literal optimizer/checkpoint update or counterfactual re-evaluation.
Check savedheadgradient identities with unchanged8eps32 artifactbound fromT013G.

Structural replication rule: overallmedianrho>=.10 AND (overallrC>=.10 ORrCout>=.10)
AND overallfullcenteredfraction<.05 AND >=3of4blockfullcenteredfractions<.10.
Report eachblockrho/covarianceratios andallthreshold flags; no extra acceptancecondition
is invented from theblockresults. These are engineering thresholds, not populationtests.

## Four-fold algebraic cross-fit

Hold outone8imageblock (16episodes) atatime. Fitmu,gbar,A,C only onother24images/
48episodes; bothmembers ofeveryimage stayintogether. Exactlythree perturbations onholdout:
full=-.001*(gbar+Htest@(A+C).T), common=-.001*(gbar+Htest@A.T),
cov=-.001*(Htest-mutrain)@C.T. Retain allfoldmoments/indices/deltas, gdotdelta,
cos(delta,-g), norm andpairseparation. No fittingbeyond algebraicmeans/covariances.
Cosine zero-denominator convention:0 (no improving direction); recordrawnorms.
Covutility passes iff >=40/64 negativegdotdelta including>=18/32clean and>=18/32corrupt,
AND overallmediancosine>0. Keepzero outcomes, no tolerance/signadjustment or calibration.

## Testing and stop

Reuse/generalize only analysis feature/outputstatistics helpers tovariablecohortsize,
preserving T013G8episodeoutputs. Synthetic tests covervariable-size factorization,
image-grouped train/testsplit, heldoutgradient exclusion, andstrict decisionthresholds.
Use knownserverCPU environment for NumPy+torch tests (localduplicateOpenMPerror already
documented). Baseline andfocusedtests precede realcollection; fullregression thenone
64record collection, then offlinealgebra. Commitmanifest beforegradientstage.
Persistallreceipts/recoverylogs/report andmirrorprojectroot. Stop T013-H forreview.
No T013-I/training/biasremoval/centering/covariance-onlydeployment/redesign/regularizer/
newtargetAP/spatialISP/gating/dose/deterministickernel work automatically.
