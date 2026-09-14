# T030-A — BLOCKED on predeclared component-gradient additivity

R045/d694c81 execution stopped at candidate episode0 (image304815, clean_s0). No scientific PASS/FAIL conclusion and no annotated source reference, AP or runtime test. The complete120candidate lock was not reached;119episodes were not attempted. No tolerance, seed, loss weights or model/environment changes were made after failure.

## Plan, code and provenance

Plan/cohort bbdbd65:60fresh train2017 images,120paired clean/s2episodes,20eachgamma/contrast/colorcast;four15imageblocks5perfamily. Selectionseed20260930,2951cumulative prior/source/memory IDs and5000val excluded. Existing outcome-free GT-valid/readable-image eligibility retained;candidate manifest has no annotationpath. CohortSHA 3ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752.

Candidatecode9fd4a7e committed/pushed before run20260914-130040-taisp-t030a-candidates120, exactrelease20260914-125919-taisp-t030a-candidate; actual TAISP_SOURCE_REVISION=9fd4a7e inrawmetadata. Onlynew analysishelper/runner/tests were added. Neutral native_task_loss mechanically copies the existing oracle function body (AST equality test); the original oracle/protected files remain unchanged. All42protected/prior files unchanged locally,45protected/prior/newcode hashes verified onremote release.

## Candidate and locked tolerance

Original frozen FasterRCNN score>=.50/stabledescendingtop20 supports; detachment retains exactboxes/classes. Hard baseline unchangedDetectorNativeLoss(det_pseudo); native objective uses the sameboxes/classes without confidence weights. Native call is literal unweighted loss_classifier+loss_box_reg+loss_objectness+loss_rpn_box_reg; empty support would give differentiablezero. Scoped samplingseed20260930; current source reference would remain20260913, but was not run.

Totalnative and fourcomponent image-cotangents are obtained by separate backward calls on the same forward graph, then reduced through the accepted shared8column float32ISPJVP withfloat64 contractions. Components are diagnostics only; no selection/reweighting. Precommitted per-coordinate sum check:

abs(g_native - sum_k g_k) <= 1e-7 + 1e-4 * sum_k abs(g_k).

The tolerance and internal samplingseed were frozen inbbdbd65 before modeloutcomes. No alternative totalgradient construction, resampling, averaging, deterministic-kernel change or cutoff relaxation was tried.

## Observed failure

Firstepisode:4supports, exactpseudo-target/support match, RNG restored, frozen/eval/parameter-grad-None/ISP-identity checks alltrue. Allsix hard/native/component directreverse-to-JVP checks pass. Componentadditivity fails in the first7of8coordinates.

Maximum absolute difference=5.210441840866088e-05; maximum error/bound=13.682925966575075. Nativegradientnorm=0.10523684906262826; L2(sumdifference)/normnative=0.0006445020062823797.

`T030A/component_sum.tsv` records every coordinate's nativegradient, componentgradient sum, error, originalbound andratio. Fullgradients/losses and allparity values remain inrawrecord_000.json. Maxindividual reverse/JVP relativeL2=1.4126384394012988e-06, below1e-5.

Native scalar=0.2728840112686157; components:
- loss_classifier: 0.12688198685646057
- loss_box_reg: 0.11353391408920288
- loss_objectness: 0.004102985840290785
- loss_rpn_box_reg: 0.028365131467580795

This demonstrates a failed numerical prerequisite in this execution, not lack of scientific utility. Individual JVP checks validate each saved image-cotangent contraction; they do not establish additivity across the separate backward computations. The cause is not established by these receipts. No additional model replay was authorized or performed to attribute it to CUDA nondeterminism versus accumulation effects.

The run exits1 immediately after saving the failedrecord. Its completion.json/final state-hash check was not reached. The initial detector state hash is73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf; per-call frozen/eval/gradNone checks passed, but do not claim a completed full-run before/after statehash verification. Candidate process is source-annotation-free by imports and executed code. No whole120record candidate lock orGTreference was created. Reference/helper test drafts prepared while candidate ran are preserved underT030A/drafts and were never executed on annotations.

## Tests, recovery and handoff

Baseline10pass5.41s; focused11pass5.81s; full249pass11skip4warnings17.95s. Tests include mechanical oracle equivalence, fixedseed/RNGrestore, trainflag restoration including exception, frozenparameters/buffers/statehash, detachedtargets, emptyzero, exactnative scalar sum, synthetic componentgradient sum/JVP and transitive candidate import isolation. The real-model firstepisode nevertheless fails the predeclared additivity bound; green unit tests do not override that blocker.

Allmodel-bearing work on A6000 CUDA0; native code/ISP/detector keptunchanged after failure. IntermittentSSHtimeouts recovered with existingworkflow status/retry; these did not cause the saved scientific/numerical assertion. No duplicated candidate launch. Fullregression after failure uses tests only, not cohort replay. CPU used for manifests, saved-vector diagnostics andreport.

Raw archiveSHA45b85ec3dd027335b8b0e5bc4ad2d939c7ada901d2cff3d637573052367cb4c2 verified before extraction. Since the abortedjob did not write its normal finalmanifest, a local SHA manifest of allreceivedrawfiles is preserved inT030A/blocked_run_sha256.json. Originalrun.sh/meta/environment/log/failedrecord retained without rewriting.

Stop BLOCKED perR045. Research review must decide a bounded diagnosis or a revised numerical protocol before resuming. Do not loadreferenceGT, relax tolerance, change samplingseed, tunecomponentweights/supports, runremaining119episodes, or declare a scientific failure/pass. Futurefreshcohort exclusion should conservatively useT030A_train_cohort.json:all60reservedIDs plus2951prior=3011source IDs, and5000val; onlyfirstimage currently had modeloutcomes.
