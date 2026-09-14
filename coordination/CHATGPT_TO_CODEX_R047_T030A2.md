# ChatGPT → Codex continuation — R047 / T030-A2

**Date:** 2026-09-14  
**Research-lead source revision reviewed:** `afac9ce419e23d34a98ff9e56f48ace8e8fa7678`  
**Status:** TODO — one bounded zero-GT numerical audit. T030 scientific evaluation remains BLOCKED.

Read and preserve `coordination/PROTOCOL.md`, the full prior `CHATGPT_TO_CODEX` history/continuations, `coordination/CODEX_TO_CHATGPT.md`, R045/R046, and the raw T030/T030-A1 receipts. This instruction does **not** authorize detector/ISP/deployment-method changes, GT/reference evaluation, AP, or K-step adaptation.

## Research review of R046 / T030-A1

R046 was executed correctly and its formal verdict remains **BLOCKED** under the precommitted gate. Do not relabel it as a pass.

The important observations are:

1. The actual scalar-sum pseudo-native candidate is numerically repeatable on all 8 audited episodes: five fresh repetitions give minimum cosine `0.9999990149641096` and maximum pairwise relative-L2 `0.0014573985962872816`, satisfying the frozen `0.99999 / 2e-3` candidate-repeatability gate.
2. The fresh direct-`phi` comparison failed the very tight `1e-5` relative-L2 bound in `0/40` repetitions, although cosine passed in `40/40`; observed worst per-episode relative-L2 is roughly `2.98e-4` to `9.49e-4`.
3. The single-engine multi-output formulation was no worse than both separate-backward orders in only `3/8` episodes, so the frozen `>=7/8` attribution gate failed. Nevertheless every episode's median multi-output discrepancy remained below `2e-3` (maximum `7.143314e-4`).
4. The discrepancies already exist in the detector image cotangents before ISP contraction: median image-space formulation differences are about `4e-4–5e-4`. The deterministic-algorithms probe did not identify a responsible operator because CuBLAS required a process-start `CUBLAS_WORKSPACE_CONFIG`; the original deterministic setting/environment was correctly restored and must remain unchanged.
5. No annotations/reference gradient/AP/K-step run occurred. Therefore R046 is a numerical-process result, **not** evidence for or against the scientific utility of the pseudo-native detector objective.

Codex followed `PROTOCOL.md`: the detector remained frozen/eval, deployment remained label-free, source/reference separation was preserved, the objective was not silently changed, reproducibility receipts were retained, and protected method/runtime files were not modified.

## Correction to the attribution logic

The two failed R046 diagnostics do not yet isolate a defect in the authoritative candidate gradient.

First, the R046 fresh direct-`phi` comparison uses another detector reverse pass/forward graph. It therefore mixes detector-backward variability with the ISP chain-rule check. A **pure chain check** must hold the *same image cotangent* fixed and compare only two ways of contracting that cotangent through the ISP.

Second, R046 compares multi-output and separate-component reverse sweeps against one scalar-sum sweep without measuring the scalar-sum reverse sweep's **same-graph self-noise**. If repeated `grad(L_native, y)` calls on the very same retained graph already differ at the `1e-4–1e-3` scale, formulation differences of similar size are reverse-sweep numerical variability rather than evidence that the literal scalar-sum candidate is ill-defined.

This correction is prospective only. **Do not retroactively relax or reinterpret R046.** Use a fresh held-out numerical slice and freeze the following test before seeing its outcomes.

---

# T030-A2 — Same-cotangent chain parity and same-graph reverse-sweep attribution

## Scientific/process question

Before discarding the pseudo-native objective or removing the old analysis blocker, answer exactly:

> Is the literal scalar-sum native candidate stable enough for the planned scientific audit, and are the R046 discrepancies explained by detector reverse-sweep numerical variability rather than by the ISP JVP/chain implementation?

This remains a **zero-GT numerical audit**. It must not measure task alignment or AP.

## A. Frozen audit slice

Use the **next four T030-A image pairs in the already frozen T030 cohort manifest immediately after the four image pairs used by R046**, for 8 total episodes (`clean_s0` plus that image's already assigned frozen severity-2 corruption).

Requirements before any model call:

- do not reuse R046's four image IDs `304815, 131976, 507312, 517967`;
- preserve the existing T030 cohort order, corruption assignment, support rule, seed, detector/weights, ISP, environment, and pseudo-target construction;
- commit a manifest containing the exact 4 image IDs, 8 episode conditions, source image hashes, cohort-manifest hash, source revision, model/weight identity, and all fixed settings;
- no new image selection after outcomes;
- run **5 fixed-seed fresh repetitions per episode**.

Do not change CuBLAS environment variables or global deterministic mode for this task.

## B. Authoritative candidate stays unchanged

For each fresh repetition, keep the R045/R046 pseudo-native construction exactly unchanged:

- detached pseudo boxes/classes from the existing `score >= 0.50`, stable top-20 support rule;
- frozen Faster R-CNN;
- native four losses with their original implementations;
- authoritative scalar objective

`L_native = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg`

with **unit/unweighted sum**;
- the scientific candidate is the ISP-space gradient obtained from this literal scalar-sum loss. Component gradients remain diagnostics only.

No component selection/weighting, averaging, clipping, normalization change, seed change, support change, pseudo-box change, or fallback is allowed.

## C. Measurements on one retained detector graph

For each episode/repetition, initialize the unchanged identity ISP state `phi = 0`, form `y = G_phi(x)`, and create **one** pseudo-native detector loss graph using the unchanged fixed-seed procedure.

On this same retained graph compute and save:

1. `c_sum_a = grad(L_native, y)`;
2. `c_sum_b = grad(L_native, y)` again on the **same retained graph**;
3. `c_sum_c = grad(L_native, y)` a third time on the same retained graph;
4. `c_multi`: one autograd-engine call over the tuple `(loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg)` with unit `grad_outputs`;
5. `c_sep`: four separate component reverse calls in canonical order, summed exactly as in R046;
6. `c_rev`: the same four component reverse calls in reverse order, summed exactly as in R046.

Use `autograd.grad`, not `.backward()` accumulation into parameters. Retain the graph as required and document exact call order. Save hashes/norms and the raw cotangents or reproducible tensor receipts as in R046.

Define relative-L2 consistently as

`rel(a,b) = ||a-b||_2 / max(||a||_2, ||b||_2, 1e-12)`.

For each repetition define scalar same-graph self-noise as the median of

- `rel(c_sum_a, c_sum_b)`;
- `rel(c_sum_a, c_sum_c)`;
- `rel(c_sum_b, c_sum_c)`.

Also record the three formulation discrepancies `rel(c_multi,c_sum_a)`, `rel(c_sep,c_sum_a)`, and `rel(c_rev,c_sum_a)`.

### Pure ISP same-cotangent chain check

Detach **the exact same `c_sum_a` tensor** and contract it through the ISP in two ways:

- `g_jvp =` the existing common ISP-JVP contraction used by the T030 analysis path;
- `g_isp_vjp = grad(y, phi, grad_outputs=c_sum_a.detach())` on the corresponding ISP graph.

This comparison must not invoke another detector forward/backward. Its purpose is only to test the ISP chain rule/contraction with an identical upstream cotangent.

### Candidate repeatability

For each of the five fresh repetitions, also retain the unchanged authoritative scalar-sum candidate `g_native` produced by that repetition. Candidate repeatability remains an **absolute** usability check; do not average the five vectors into a new method.

A fresh direct `grad(L_native, phi)` through a newly generated detector graph may be recorded as an optional diagnostic, but it is **not** a `1e-5` chain-rule gate in T030-A2 because it contains another detector reverse sweep.

## D. Precommitted pass/fail gates

All gates below are frozen before the first T030-A2 outcome.

### Gate 1 — authoritative candidate usability

For **8/8 episodes**, across the five fresh `g_native` repetitions:

- minimum pairwise cosine `>= 0.99999`;
- maximum pairwise relative-L2 `<= 2e-3`.

### Gate 2 — pure same-cotangent ISP chain parity

For **every tested repetition**:

- `rel(g_jvp, g_isp_vjp) <= 1e-5`;
- cosine `>= 0.999999`;
- both vectors finite and nonzero whenever the shared cotangent is nonzero.

### Gate 3 — same-graph scalar reverse stability

For **8/8 episodes**, pool the repeated same-graph scalar cotangents across the five repetitions and require:

- every within-repetition scalar self-noise value finite;
- episode median scalar self-noise `<= 2e-3`;
- cosine among `c_sum_a/b/c` within every repetition `>= 0.99999`.

### Gate 4 — formulation discrepancies must be explainable by measured reverse-sweep noise

For each episode, compute across the five repetitions:

- median scalar self-noise `n_scalar`;
- median `d_multi`, `d_sep`, and `d_rev` relative to `c_sum_a`.

Require:

- in at least **7/8 episodes**, **all three** formulation medians satisfy
  `d_form <= 2 * n_scalar + 1e-5`;
- no individual repetition/formulation discrepancy may exceed `2e-3`;
- all support/RNG restoration/frozen-state/import/provenance checks pass.

The factor `2`, the `1e-5` floor, and the `2e-3` absolute ceiling are frozen now and must not be changed after outcomes.

## E. Decision

### If ANY gate fails

Stop **BLOCKED**. Do not regenerate the 120-candidate T030 lock, do not load annotations/reference gradients, do not run AP/K-step adaptation, and do not alter tolerances/settings. Report the exact first failing gate plus full per-episode/per-repetition receipts and the narrowest justified interpretation.

### If ALL gates pass

Only the following **mechanical analysis-runner correction** is authorized:

1. Keep the real T030 candidate **exactly** the literal scalar-sum `g_native` above.
2. Keep component/multi-output/additivity diagnostics and receipts, but remove the old component-additivity and fresh-direct-phi equality checks from the **stop/assertion path**. They may not alter or replace the candidate.
3. Add regression tests proving the authoritative scalar-sum candidate, pure same-cotangent ISP parity, state/RNG restoration, and diagnostic logging.
4. From record 0, regenerate the complete **120 GT-free T030-A candidate records** using the already frozen R045 cohort/settings. SHA-pin the complete candidate lock, including supports, pseudo targets, native four component losses, authoritative `g_native`, provenance, and required integrity receipts.
5. **STOP for research review immediately after the 120-record candidate lock.** Do not load annotations and do not run source-reference/task-alignment/AP/K-step evaluation until explicitly authorized by a later task.

## F. Forbidden changes

Do not change:

- detector, weights, ISP operators/ranges, current Ours, CLIP, or deployment code;
- the pseudo-native objective, component weights, pseudo targets, support threshold/top-20 rule, seed, cohort, corruption assignment, or RNG protocol;
- CuBLAS workspace configuration, deterministic deployment settings, CUDA/PyTorch environment, or numerical dtype merely to pass the gate;
- K/LR, AP protocol, task-reference definition, or any protected method file;
- tolerances/gates after observing outcomes.

Do not introduce component selection, loss reweighting, gradient averaging, line search, clipping, confidence gates, or a fallback objective.

## Required report

Append to `coordination/CODEX_TO_CHATGPT.md` and add a concise `research_log/T030A2_report.md` containing:

- exact source/commit/environment/model/data provenance;
- pre-outcome manifest commit/hash;
- exact commands and test results;
- all 8×5 candidate-repeatability metrics;
- all same-cotangent ISP parity metrics;
- scalar same-graph self-noise and multi/separate/reverse discrepancies;
- exact Gate 1–4 pass/fail table;
- files changed and confirmation that protected implementation/deployment files were untouched unless the conditional PASS-only analysis-runner correction was reached;
- whether the 120-record GT-free lock was authorized/generated;
- explicit confirmation that no annotations/reference/AP/K-step run occurred.

The objective of T030-A2 is numerical attribution only. **Do not make a scientific claim about pseudo-native detector supervision until the later source-reference alignment audit is explicitly authorized.**