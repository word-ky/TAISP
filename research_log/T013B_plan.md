# T013-B pre-outcome plan

2026-09-13 01:34 +08. R016 synchronized at83f734e; T013-A CLOSED.
Task scope: first-order fidelity audit, then conditional tiny source smoke only.
No deployment/ISP/predictor/prompt/update changes. Reuse bfd2484 initialization
entry point, DifferentiableISP, transfer_norm and existing oracle outer loss.

## Part A fixed fixtures and reference

12 episodes indexed i=0..11; CPU float64, one thread, seed20260913 (no random
fixture selection). For j=i%3, use a1x3x9x11 linear RGB ramp with endpoints
(.08+.015*j, .70+.025*j). For group g=i//3, set phi0 to
amplitude[g]*linspace(-1,1,8), amplitudes=[0,.08,.20,.35].
These span three image intensities and four identity-to-nonlinear initial states.
Detector mock: mean((enhanced-.9)^2); semantic mock: mean((enhanced-.4)^2).
Both are fixed frozen scalar-buffer losses (the T013-A toy pattern).
Outer scalar: mean((final_enhanced-.55)^2). No loss-weight search.

K=3, inner lr=.1, norm epsilon1e-12. FO uses the current training entry point;
EXACT is analysis-only functional unroll using create_graph=True and the identical
transfer_norm. FD uses central differences of the same scalar forward at
phi0 +/- **1e-5** along all8 coordinates. All fixtures/epsilon fixed before run.

Record raw FO/EXACT/FD vectors, all trajectories, cosine(FO,EXACT),
cosine(EXACT,FD), coordinate sign agreement, FO/exact norm ratio, maximum
trajectory discrepancy and saturation. Validate reference first: every exact/FD
cosine >=.999 on these nonsingular fixtures; if not, diagnose reference without
tuning fixtures/epsilon to obtain a continuation pass.

R016 continuation gate unchanged: median FO/exact cosine>=.5, at least9/12
positive cosines, all gradients finite. Negative results retained. Gate is only
a feasibility threshold, not a statistical claim. No Part B if gate fails.

## Conditional Part B

Only after A passes, inspect A6000 existing train2017 images plus annotations.
If unavailable, stop and document locations checked; no val substitution/download.
If available, choose4 eligible image IDs deterministically with random.Random
(20260913).sample(sorted(valid available IDs),4), save orderedIDs/file hashes and
annotation hash to T013B_train_microset.json and commit before optimizer execution.
Eligible: at least one non-crowd positive-area valid bbox, existing image file.
Each image has clean then one corruption; cyclic corrupted cases gamma_s2,
contrast_s2,color_cast_s2,gamma_s1; fixed image order. Reuse existing corruptions.
Original episode pseudo support uses source score>=.5/top20 detached, held fixed.
Initializer input only episode RGB. Labels go solely to detector_task_loss outer
objective with fixed sampling seed20260913; all models frozen/eval as existing
oracle machinery permits its loss-return branch flags transiently.
Predictor-only SGD lr1e-3, exactly3 optimizer steps, each mean of8 episodes.
Record step0..3 outer losses/components, head/trunk gradients, parameter delta,
clean/corrupt phi0/phi3 statistics, saturation, empty support, frozenstate equality,
CUDA peak allocated memory/time. No loss-decrease requirement, tuning or AP.

## Checks and stop

Run relevant baseline tests before editing; focused reference/FD/algebra tests,
deployment label-isolation and reset tests, then relevant regression. Exact
unroll stays in analysis. Never alter T013-A strict CUDA assertion/failure.
Report commands/commits/results/environment and any actual stop reason. Stop at
T013-B review; no T013-C, longer training, target evaluation or new target cohort.
