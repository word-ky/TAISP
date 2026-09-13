# T022-A2 — analysis complete, NEEDS_REVIEW; T022-A remains pre-AP BLOCKED

2026-09-14 +08. R036/3e366b7; precommitted plan2b669bc; audit codeee20040.
Run `20260914-020134-taisp-t022a2-repeatability`, release
`20260914-020108-taisp-t022a2-audit`, 02:01:39–02:03:38 +08, shell exit0.
Exact start time is also preserved in train.log. No official AP, GT, new cohort,
training, teacher recomputation or method change was used.

**The first observed variability is upstream of spatial JVP/dose:** independently
recomputed detector/CLIP image cotangents vary on literally fixed enhanced
images, while fixed-cotangent regional JVP and fixed-input dose are exact.
Current Ours also exhibits trajectory differences. The real support-derived
spatial mask does not show larger final-state or final-image dispersion than
current Ours in these five repeats of this one image/condition. This is
computational attribution, not efficacy evidence or a relaxed reproducibility gate.

## Scope and test evidence

Reuse image160585/gamma_s1 and original supports from the saved R035 current row;
identity plus saved empty-mask first-trajectory states1–3. Mask/support/state
arrays and raw hashes are retained. Cohort SHA256 remains
589878ad7af1d86bf73514d63dd58a4e40d58797d119ea4981be9e4292772690.
All23 prior runtime/analysis modules match precommitted LF pins. The two new
analysis/test files also match their release pins (25 verified files total).
Original runtime/ISP/current loss/CLIP/mask/K3/LR.1/rho.5/dtypes unchanged.

Focused increment15passed5.36s; run focused **15passed5.17s**; full
**190passed10skipped4warnings10.32s**. Instrumentation uses loss/output tensor
hooks that return None and retain the original runtime cotangents without
extra forward/backward calls. Focused tests verify expected toy cotangents and
unchanged current/spatial outputs with hooks. Instrumented real trajectories
are measured in this run; identical scheduling to the earlier uninstrumented
run is not claimed. Model/gradient/JVP/adaptation calls used A6000 CUDA;
offline summaries used CPU. No dependency or driver changes.

## Fixed enhanced images: first observed variable quantity

Each fixed state uses five independent computations, all10pair comparisons.
Both scalar losses are exactly equal for all pairs at each state. Neither cp
nor cc is exactly equal for any pair, despite fixed inputs.

| Saved state | Max cp relative L2 | Max cc relative L2 | Exact scalar loss pairs | Exact cp/cc pairs |
| --- | --- | --- | --- | --- |
| 0 | 0.000366375394279 | 2.31012953886e-08 | 10/10 each | 0/10 each |
| 1 | 8.77412289255e-08 | 2.43066695809e-08 | 10/10 each | 0/10 each |
| 2 | 9.03390853799e-08 | 2.08607075165e-08 | 10/10 each | 0/10 each |
| 3 | 2.52992869862e-05 | 2.39185443518e-08 | 10/10 each | 0/10 each |

At fixed identity, max cp relative L2 is approximately3.664e-4; at saved step3
it is2.530e-5. CLIP cotangent differences are approximately2e-8. This localizes
the first observed variability to the upstream CUDA forward/backward path,
without identifying an exact backward operator from these scalar comparisons.

## Fixed cotangents and fixed vectors

At each of four saved states and each real/zero/one mask, five independent
regional_gradients calls used the **first saved cp/cc**, without averaging.
All60calls /120repeat pairs give exactly equal float64 object/background pseudo
and CLIP vectors. All8-D vectors are saved. No variability was observed from
JVP or masked float64 reduction under these literally fixed inputs.

The first state/real-mask tuple was fixed for20dose_step calls. All190pairs
have exact c, both multipliers, common direction u, base delta v and both8-D
deltas. No pure-dose implementation blocker was observed.

## Runtime reference: five zero-initialized K=3 repeats per case

All10pairs are retained for every step's cp/cc, common gradients, states and
final processed image. Maxima below may come from different pairs. Current
state is8-D and spatial state is2x8-D, so final-image differences provide the
common output-space comparison.

| Case | Max state abs | Max state relative L2 | Max image abs | Max image relative L2 | Exact state/image pairs |
| --- | --- | --- | --- | --- | --- |
| current | 0.00024040043354 | 0.0123554533992 | 0.000162780284882 | 0.000165785568705 | 0/10; 0/10 |
| real | 0.000134317670017 | 0.00473746886858 | 9.35196876526e-05 | 8.13435611232e-05 | 0/10; 0/10 |
| empty | 0.000239312648773 | 0.00946152930126 | 0.000137090682983 | 0.000222583284531 | 0/10; 0/10 |
| full | 9.97884199023e-05 | 0.00358865233247 | 7.68303871155e-05 | 6.68022209325e-05 | 0/10; 0/10 |
| empty_support | 0 | 0 | 0 | 0 | 10/10; 10/10 |

Current Ours shows larger maximal final-state/image differences than the real
spatial mask in this audit. Synthetic empty-mask image relative dispersion is
about1.34x the current maximum; full-mask dispersion is smaller. This does
**not** support an exclusive or strongly demonstrated degenerate-mask-only
amplification diagnosis: current Ours also varies, and all nonempty objectives
show trajectory amplification. The previous two-repeat edge result alone was
insufficient to establish spatial-specific instability. No new tolerance,
PASS criterion, confidence claim or population claim is derived from five
repeats on one image. The existing exact-repeat gate remains unchanged.

All25episode model/reset/mask checks passed. The real mask hash and area are
saved in environment.json and its tensor is immutable across repeats. All
100step states and corresponding image cotangents/common gradients are retained.

## Empty-support control and raw completion classification caveat

All five empty-support episodes have exactly zero states at all four steps,
zero pseudo image cotangents/common pseudo gradients, and exactly identical
final images. All10state/image pairs pass exact equality. This is **not** a
hard reset/control failure. CLIP cotangents still have tiny upstream differences
(max relative L2 around2e-8), but zero pseudo gradient makes every update zero.

The raw completion.json says `empty_support_control_blocker` because the new
analysis terminal condition incorrectly requires **every recorded field**,
including CLIP cotangents, to be bit-identical for this control. This is an
over-broad analysis summary check, not a state/image failure. It executes after
all25episodes and all pairwise outputs are written. Raw status/receipts/code
are retained unchanged. The offline report separately identifies the exact
zero-state/image control and CLIP variation; it introduces no tolerance and
makes no new model calls. Do not interpret shell exit0 or the raw control label
as a scientific pass/fail. A future use of this analysis script should correct
that terminal classification before relying on it; no runtime repair is needed
or validated by this observation.

## Separate deterministic-algorithm process

Exactly one separate process enabled deterministic_algorithms(True,warn_only=False).
It raised a RuntimeError in CLIP's visual_projection -> F.linear / CuBLAS,
requesting CUBLAS_WORKSPACE_CONFIG before process startup. Full operator stack
and error text are in deterministic/deterministic.json. The error occurred in
the CLIP forward path before the representative cotangent backwards completed;
it does **not** identify ROI pooling/backward as the culprit or prove that this
specific linear operation caused the default-mode variation. No environment
repair, second deterministic diagnostic or deterministic performance run was
attempted. The main audit ran in a fresh process with deterministic mode off.
All model hashes/frozen/eval/grad-none checks in the separate process passed.

## Decision and next action

T022-A2's bounded attribution evidence is complete: **NEEDS_REVIEW**.
T022-A remains **pre-AP BLOCKED** under its unchanged exact-repeat requirement.
The first observed variability is inherited upstream; fixed-input spatial
JVP/dose are exact; real-mask output dispersion is not larger than the measured
current reference. Empty-support state/image reset is exact. These findings
support research review of a separately precommitted baseline-referenced
criterion, as R036 anticipates, but do not authorize that criterion now.

No200image formal study, AP, new data, parameter sweep, kernel repair, source
training or next task was started. No active job remains. Preserve all original
T022-A/A1 failures. Await the next explicit research decision.

Raw archive SHA256bde562ff84723d1949f46994e550650afad6e1ef9d177791e94feb5684a06e8d verified before extraction. 129 files / 583,264,937 bytes; largest file 21,072,870 bytes. Raw tensors are losslessly gzip-wrapped torch files, split by repeat; artifact_manifest.json lists every SHA256. summary.json and complete_tables.md retain all fields/groups without outcome selection.
