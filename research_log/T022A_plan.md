# T022-A pre-outcome plan: direction-locked spatial dose

R034/6001e14. Keep current Ours and all prior method modules unchanged. Only an isolated
two-region dose candidate is authorized; no independent regional update directions.

## Cohort and fixed protocol

Reuse cumulative source preparation, eligibility and exclusions through T021-A:
1236 prior source/development/debug IDs, all5000val and existing evaluation ledger.
First200 by (sha256('20260923:'+decimalID),ID), four consecutive50-image blocks.
Store JPEG hashes, ordered IDs/blocks, exclusion count/hash and environment/module pins
before model outcomes. Selection seed20260923; modelseed20260912 stays unchanged.
Exactly no_adapt,current_ours,spatial_dose_ours, clean plus six existing corruptions.
No labels/GT in teacher,mask,pseudo loss,CLIP,dose or updates. GT only official evaluation
after all prediction files are complete. No source fitting or meta-training.

## Formula, mapping and numerical implementation

Current score>=.5/top20 original-view supports only. Freeze mask m as box union;
copy the existing T014 support_mask rasterization exactly: clip coordinates to image,
floor left/top,ceil right/bottom, binary rectangles, no dilation/softening. Freeze m.
Spatial y=m*G(x,phi_obj)+(1-m)*G(x,phi_bg), both initial states zero, each8D.
Keep existing ISP operators/ranges and exact current fixed-ROI pseudo/CLIP losses.

go=dLp/dphi_obj, gb=dLp/dphi_bg, gs=go+gb, u=gs/(norm(gs)+1e-12).
ao=dot(go,u),ab=dot(gb,u),c=(ao-ab)/(abs(ao)+abs(ab)+1e-12).
r=norm(gclip_obj+gclip_bg),v=-.1*r*u,mo=1+.5*c,mb=1-.5*c.
Update obj+=mo*v,bg+=mb*v, exactly3updates with steps0..3diagnostics. A represented
zero norm(gs) produces no update and records the event; no alternative direction.
rho=.5 fixed. c in[-1,1], multipliers[.5,1.5],mean1; no average-dose increase.
Inactive region gradient is exactly zero for empty/full masks. Its state may still
receive the shared half-dose, but is unobservable in the composed image. Empty pseudo
supports give zero gs and both states remain identity with no fallback.

Reuse the already-validated T014-A1 common-Jacobian approach for stable ISP gradients,
not its task/oracle analysis code. The earlier direct reverse regional partition had
an observed float32 closure failure; A1's JVP plus float64 reductions passed. At each
actual regional state obtain8 exact ISP JVP columns (16 total for two states), reuse
each column for detached pseudo andCLIP image cotangents, multiply/reduce in float64,
then cast the resulting8Dvectors to runtime float32 before the frozen update formula.
This is the chain rule for the same composed objective, not a new gradient estimator,
finite difference or changed loss. No detector call per region/JVP. Detector/CLIP/ISP
image computation remains CUDAfloat32. JVP/reductions stay CUDA; no CPU model fallback.
Port only pure mask/compose/JVP operations into new deployment modules so no labels,
oracle or source-analysis objects enter runtime. Existing donor files stay unchanged.

## Numerical checks before AP

Baseline spatial_action/common_jacobian/trust_radius:13passed1skipped2.38s on accepted
release222014. Increment1 tests equations/direction/zero/mask/reset and original donor
equivalence; pass before shared driver integration. Full suite then CUDA smoke.

Predeclare inherited A1 comparison: at equal states, common-shift pseudo andCLIP
gradients versus direct global reverse reference must have relativeL2<=1e-5 and
cosine>=.999999. Bothzero gives rel0/cos1; onlyonezero fails. Norm convention uses
the same reference denominator. Processed images atol2e-7,rtol1e-6. Float64 partition
closure bound1e-12+1e-10*(abs(go)+abs(gb)); do not derive either region by subtraction.
Bounded coefficients/meanone float32 tolerance1e-6; no clipping to hide a failure.
Synthetic c=0 must reproduce globalK3 and image; real equal-state mean/base update
must match current update under the same vector tolerance (c0 mathematical limit).

Real parity smoke uses first2cohort images, all7conditions, equal states phi=0 and
phi=0.01percoordinate. For each, evaluate pseudo andCLIP losses once on global ISP;
retain image cotangent/direct globalgradient and compare regional JVP chain-rule sums.
Same forward/cotangent avoids confounding two nondeterministic detector backwards.
Save every check and vector before asserting. A material parity/isolation failure
stops the task with its receipt; no tolerance changes, AP run or numerical sweeps.
Then runtimeK3smoke on the same2images/7conditions,0AP. Test full/empty masks and
inactive-region zero gradients with synthetic frozen losses; preserve exact reset.

## Formal run, diagnostics and decision

After all tests/smoke pass, one fixed200run:1400teacherforwards,2800adaptiveepisodes,
21predictionfiles,105officialCOCOevaluations. Current shared pipeline/evaluator reused.
No modification of currentOurs losses/adaptation/ISP/detector/CLIP/support modules.
Retain masks/rectangles/hash/area, regional gradients/commonvector/CLIPsum/r/u/c,
both multipliers, phi_obj/bg norms and difference, zero events, all step updates.
Report clean/corrupted/block distributions and all condition/AP/AP50/AP75 tables.
Candidate latency includes its JVP work; teacher-inclusive adds original teacher,
excludes finalprediction/stateverification/AP. Record peak memory and full timing.

All six gates: corruption macro-current>=+.10AP;3/4positiveblocks;4/6positiveconditions;
macro>raw;clean>=current-.10AP;no parity/isolation/leakage/numerical/repro blocker.
APscale0-100;AP50/AP75 and mechanism distributions never substitute for the gate.
Fail closes fixed two-region direction-locked dose; pass developmental only. No rho,
mask/region/threshold/K/LR/CLIP tuning, independent directions, labels, source/meta,
predictor/native-loss/FCOS/SSD/val/deployment expansion. Commit/push all receipts/report,
mirror A6000 and stop NEEDS_REVIEW (BLOCKED if pre-AP numerical checks fail).
