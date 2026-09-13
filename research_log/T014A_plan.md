# T014-A pre-outcome spatial action-space audit plan

R024/f5da87a accepts/closes T013-I and authorizes this bounded source-only audit.
Commit this plan and T014A_source_manifest.json before new model outcomes. No optimizer,
finite-step spatial update, deployment/predictor edit, CLIP call, new cohort or target/AP.

## Cohort and pins

Take image positions0..3,8..11,16..19,24..27 of the accepted H manifest, preservingorder.
Exactly16images/32episodes, clean then its original corruption; fourblocks offourimages,
oneimagepercorruptionperblock. IDs:
182164,522454,425480,410054,496294,364188,83968,92109,
469614,298468,365614,117105,449936,534000,121632,83257.
Selected manifest SHA256 a4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e.
H parent cohortSHA99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357.
Reuse exact original-image H pseudo supports, saved under H run's artifacts/collection:
supports.json SHA bf8f688741fcb4956c7ed40f52cc5f0a63d3030cad3ada968f1457a38c2cf747;
environment.json SHA d0635da6c52a1f41a86181ace053c6c6fe1f38f342530b6fd9d806e8ee98cc42.
These supports came from the accepted score>=.5, descending top20 original-only source
prediction. Reuse cached supports exactly; no resampling from adapted images or labels.
Source FasterRCNN weightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384,
loaded stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf.
Existing global8D DifferentiableISP and detector_task_loss(seed20260913) unchanged.
Verify JPEG hashes using existing source_meta_smoke.load_episodes. No post-outcome exclusion.

## Analysis compositor and masks

Add only taisp.analysis helpers/tests. Detached binary BCHW mask has shape1x1xHxW.
Clip each cached xyxy box to imagebounds, floorleft/top, ceilright/bottom, and union
the half-open integer rectangles. Save originalboxes, rasterrectangles, shape,
maskarea andSHA256 of row-major uint8mask so mask is exactly recoverable without labels.
Use G2R=m*G(x,phi_obj)+(1-m)*G(x,phi_bg), with two independent external8Dzero states.
Empty supports =>mask0, objectgradient0, backgroundglobal; pseudo loss/gradients0 via
existing empty-support objective. Retain all such records. No annotation/case/gradient
input to mask helper. No modifications under taisp/isp or deployment.

## Identity gradients and reconstruction

For each episode and each objective (task oracle, then accepted det_pseudo confidence),
build the direct global ISP output at phi=0 and execute that loss once. Obtain both
the direct global8Dgradient and its image-output cotangent dL/dy from that same loss.
Build the independent two-region ISP graph at bothzero states and verify its output
equals the direct global output. Backpropagate the detached same dL/dy through this
regional graph to obtain object/background gradients. This is the exact chain rule
at the identical output, not an approximation or a new objective. It avoids unrelated
detector-repeat CUDA noise contaminating a compositor-Jacobian identity check. No reuse
of H's K3 gradients. Two source loss executions per episode; no regional model repeat.

For each objective compare direct global gradient with regional sum. Predeclare element
bound abs(error)<=1e-7+1e-5*abs(direct_global). Report maxabs, L2relative andmaxerror/bound.
Do not relax it; any failed bound blocks scientific interpretation. Direct/regional
equal-state image bound inherits ISP tests: atol2e-7,rtol1e-6; identityversusinput same.
Synthetic double gradient comparisons use atol1e-12,rtol1e-10. Tests include unequalstates,
sharedstate gradients, clipped/fractional maskboxes, empty/full masks anddetachment.

Normalfloat32 CUDA, deterministicFalse,cudnn.benchmarkFalse, CUBLASconfigunset, threads1,
seed20260913; use inherited detector/oracle/native helpers andfixed sourceweights.
Record taskloss/components,pseudoloss, allsix8Dgradients, mask/support/isolation,
identity/saturation, detectorstatehash before/after andper-episodeunchanged/gradNone/eval
checks. ISPownedphi stayszero/gradNone. Saveeachrecord beforechecking reconstruction;
ifblocked retainrawdata/log andstop, no fallback or extra modelrun.

## Saved-vector metrics and exact gate

Only float64CPU algebra after all32 records pass reconstruction andisolation.
Use eps1e-12 in R024's written denominators for cancellation,R_task,D_global,D_spatial;
cosine returns0 for exactzero-norm vectors, otherwise ordinarycosine. Retainzero-gradient
flags; the literal cancellationformula returns1 when both taskregionszero, which must
not be interpreted as opposingnonzero gradients.
Metrics exactlyR024: regionaltask/pseudocosines; taskcancellation; R_task;
A_global/A_spatial/Delta_A; D_global/D_spatial/Delta_D. No unstable D ratios.
All rawregionnorms andgradients retained. Summariesoverall,clean/corrupt, fourfixedblocks,
eachcorruptedcase. Descriptive strata: empty/nonemptysupport; maskarea0, (0,.05],
(.05,.95), [.95,1),1. Strata do not change primarycounts/gate. Report empty strata asN0.
Degenerate masks can have R_task=sqrt2 from the specified concatenated-budget convention;
do not mistake that alone for evidence of opposing object/background gradients.

Advancement conjunction: medianR_task>=1.20 overall AND>=1.15 corrupted;
Delta_D>0 on>=20/32 overall AND>=10/16corrupted; overallmedianDelta_D>0;
positive blockmedianDelta_D in>=3/4blocks. Delta_A/cosines/cancellation arediagnostic only.
No threshold change, image replacement or positive-result tuning. Pass permitsonly
researchreview oflaterfixedprototype; fail means no spatialISPimplementation fromthispartition.

## Tests, artifacts and stop

Baseline existingISP/detector-native tests onknownremoteCPU; focusedcompositor/gradient/
geometry/gate tests, then fullregression beforeone32recordidentityaudit. Persistexact
command/revision/testlogs/cohort/supports/records/summaries/report inresearch_log,
appendCODEX_TO_CHATGPT, commit/push andmirrorA6000 projectroot. StopT014-A NEEDS_REVIEW;
noT014-B/spatialdeployment/spatialCLIP/finite-step/meta-training/redesign/gate-dose/
FCOS/SSD/AP/newvalidation work until explicitresearchreview.
