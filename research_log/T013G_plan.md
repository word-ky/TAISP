# T013-G pre-outcome feature/factorization audit plan

R021/3a1f363 accepts and closes T013-F. Commit this plan before new numeric summaries.
All required saved arrays are present; no model forward/backward or optimizer needed.

## Provenance

T013C run20260913-030301-taisp-t013c-conflict-conditioning/artifacts/audit:
- receipt.json SHA256 7c35fb9e5f7b1c923c06f625ddf806d443e763b56aa52d8027594f0b95cce943
- original_predictor.pt SHA256 a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120
- joint_one_step_predictor.pt SHA256 fec4271dcfda344023521c17070d117d08082664fdb6ddd6934adb47e727c634

Saved-array hashes use Python json.dumps(array,separators=(',',':'),ensure_ascii=True,
allow_nan=False).encode('utf-8'), before conversion to float64:
- H = probes.joint.conditioning.features (8x16):
  4ffe547ee5d3373a6c5258a1fa89582dea6d04389eb8b82aef5a37f53af066bf
- G = gradient_audit.episodes[].phi0_gradient (8x8):
  ba330697a73af6b2e1017944f0446de85039f4406850b3b5dd28f842f42df90d
- saved dL/dW = gradient_audit.episodes[].head_weight_gradient (8x8x16):
  ebf8700a86f71e250258d138c19616621797639c3e2b8fc6b99d5be81c82a901
- saved dL/db = gradient_audit.episodes[].head_bias_gradient (8x8):
  ba330697a73af6b2e1017944f0446de85039f4406850b3b5dd28f842f42df90d
- Saved float64 Wh+b = probes.joint.conditioning.outputs (8x8):
  e9d159188699e2a25c52fda856b178fd9af3a2de0457ef05b2c9b9122e4586c9
- Saved executed float32 phi0 = probes.joint.episodes[].phi0 (8x8):
  174b8e0d3bc034bdd18500a41e1508ee9e0455458ec997744aa277e96444fa9c

Use exact order 65088 clean/gamma_s2,426525 clean/contrast_s2,
541157 clean/color_cast_s2,129068 clean/gamma_s1. Same saved original zero-head
Conv3->16/SiLU/GAP trunk; original/joint feature-state tensors must be identical.
No new images, annotations or corruptions. Model/support/method pins remain those of
T013C; this task consumes only saved tensors, with no performance evaluation.

## Precision and reconstruction tolerance fixed before results

All new algebra uses NumPy float64 on CPU. Check per-episode W gradient outerproduct,
bias gradient, mean W gradient A+C, joint checkpoint W/b=-1e-3*(GW,gbar), and summed
one-step outputs against BOTH saved conditioning output and executed phi0.

For each comparison let u=np.finfo(np.float32).eps=1.1920928955078125e-7.
Elementwise acceptance is abs(calculated-saved) <= 8*u*(max(abs(saved))+abs(saved)).
The scale max is taken over that explicitly named comparison tensor. This fixed bound
allows float32 product/8-way reduction/update rounding; it is not a claim of bitwise
identity. Record maxabs error, Frobenius error, scale, largest bound, and maximum
error/bound ratio. For a zero saved tensor the bound is zero and equality is required.
No epsilon increase, precision workaround or recomputation after a failure. Save exact
failure diagnostics and stop interpretation if any comparison fails. Also report the
float64 algebraic mean(G_i h_i^T)-(A+C) residual separately.

## Required summaries

H,mu,Z: normmu, raw/centered energy/fraction, centered SVD, participation effective
rank (sum s²)²/sum s⁴, full Euclidean/cosine8x8 matrices. Four paired displacements
di; each rho=di/median distance to other3cleanfeatures, allrho+median. For eachcorrupt
nearest of4cleanfeatures, distance and ownimage flag. No universalcorruptiondirection.

Gbar, A=Gbar mu^T, C=mean((G-Gbar)(H-mu)^T): retain full matrices, Frobeniusnorms,
cosine, crossenergy2<A,C>, bounded rC=normC/(normA+normC), savedGW reconstruction.
Output components at eta=.001 are repeated -eta*gbar, -eta*H@A.T, -eta*H@C.T.
For each: rawenergy, centeredenergy, fulloutputvectors, sameimage distances. Retain
all three pairwise cross terms for raw AND centered energy plus fullsum errors.
rCout=norm(center(H@C.T))/(norm(center(H@A.T))+norm(center(H@C.T))).

Exactly two algebraic counterfactuals: common-only=-eta*(gbar+H@A.T);
covariance-only=-eta*(H-mu)@C.T. Record centeredenergy, paired and fullpairwise
distances. Not candidate deployment parameterizations; no loss/AP/model calls.

## Triage and scope

Apply R021 unchanged: medianrho<.10 => representation-insensitive on thismicroset.
Otherwise bothrC<.10 andrCout<.10 => zero-head/common-gradientcollapse despitevariation.
Otherwise => mixed/common-term domination. Describe which commonterm masks centered
energy using measured ratios/crossterms; no redesign choice fromeightepisodes alone.
No performance interpretation from the T013F pooled exception.

Reuse saved-artifact provenance and existing analysis/testing environment. New small
offline analysis module and synthetic tests for factorization, energycross terms,
effective-rank/rho andfixedtriage/tolerance. Baseline existingalgebra tests beforeedits,
focusedtests thenremote CPU regression andoffline audit. Checkpoint deserialization
with torch.load(map_location='cpu',weights_only=True) is permitted; no model creation.
Save everyarray/check/error/table andreport; stop T013-G forreview. No T013-H/training/
regularizer/biasremoval/centering/redesign/newbackbone/newdata/targetAP/spatialISP/
gating/dose/deterministickernel work automatically.
