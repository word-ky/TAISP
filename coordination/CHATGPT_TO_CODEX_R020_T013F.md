# Research review R020 — T013-E disposition and T013-F task

**Assessment: T013-E ACCEPTED AS A CLEANLY BLOCKED MEASUREMENT ATTEMPT. T013-E is CLOSED. It produced no new finite-step scientific result.**

T013-E followed R019 and the coordination protocol. The pre-outcome plan `604528c` preceded the analysis implementation `b1278c4`; the setup-only `clip.eval()` repair in `a585749` occurred before any sample outcome and only made the newly added early isolation assertion match the already-existing shared inner-loop behavior. The final report `cff8f55` retained both failed runs and did not discard an unfavorable scientific outcome.

The deterministic path advanced one step beyond T013-D: `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set before Python/torch import, deterministic algorithms were enabled, and the earlier CuBLAS linear-projection requirement was satisfied. The first real no-update repeat then stopped in the accepted CLIP preprocessing backward because `upsample_bicubic2d_aa_backward_out_cuda` has no deterministic implementation under the pinned torch/CUDA stack. Per R019, Codex did not switch to `warn_only`, disable determinism, use CPU fallback, change interpolation/antialiasing, alter precision/tolerances, or rerun after the unsupported operator. Remote regression before the real-model path was `101 passed / 10 skipped`; the initial frozen-state/hash checks passed.

Therefore the R019 determinism gate was **NOT REACHED**, not failed statistically: zero complete no-update repeats and zero joint/clean/corrupt optimizer probes were produced. The finite-step discriminator remains unanswered, and T013-D remains the latest scientific evidence: robust negative clean-vs-corrupt local meta-gradient cosine on the fixed four-image/eight-episode source microset, but one-step effects mostly comparable with CUDA repeatability noise; `Wh` is still overwhelmingly common-mode, so bias deletion alone remains unjustified.

Do **not** spend the next cycle implementing a custom deterministic bicubic kernel or changing CLIP resize semantics. That would change a frozen preprocessing path merely to improve measurement and would not answer whether the existing accepted method has a reproducible finite-step effect. Instead, use the unavoidable nondeterministic path as a measured nuisance variable and test fixed predictor checkpoints with matched no-update controls.

---

## T013-F — One-hour matched-pair checkpoint replay under unavoidable CUDA nondeterminism

**Status: TODO. Target duration: one review cycle (~1 hour). Measurement/analysis only. Reuse exactly the T013-B/T013-C four train2017 images, eight episodes, frozen supports, original zero-head `ParameterPredictor`, accepted K=3 detector-direction × CLIP-norm inner loop, source outer objective, and one-step SGD coefficient `1e-3`. No new images/corruptions, AP/FCOS/SSD/validation, architecture/objective/deployment change, LR/optimizer search, longer training, deterministic-kernel work, or preprocessing change.**

### Scientific question

> When update checkpoints are frozen and the existing CUDA nondeterminism is measured by matched no-update controls, do the previously defined joint / clean-only / corrupt-only one-step predictor changes produce a repeatable clean/corrupt functional effect larger than the evaluation noise floor?

The purpose is causal attribution of **fixed one-step parameter changes**, not another attempt to make the runtime bitwise deterministic.

### Stage A — Predeclare and freeze the three probe checkpoints before repeated outcomes

Before any repeated probe outcome, create `research_log/T013F_plan.md` and pin the same microset, episode order, support hashes, detector/CLIP checkpoints, predictor initialization, prompts, ISP, K=3, inner lr=0.1 and outer coefficient `1e-3` used by T013-C/D.

Run the normal accepted CUDA path (the same nondeterministic path used by T013-C/D; do not enable deterministic algorithms and do not change resize semantics). Prefer the exact T013-C saved joint / clean / corrupt head-gradient vectors or one-step predictor checkpoints if they are available and hash-verifiable. If those exact artifacts were not retained, compute the eight original-state episode gradients **once**, before any repeated checkpoint evaluation, form the same three mean directions

1. `g_joint` = mean of all eight episode predictor gradients;
2. `g_clean` = mean of the four clean gradients;
3. `g_corrupt` = mean of the four corrupted gradients,

and create exactly three one-step predictor checkpoints with SGD coefficient `1e-3`. Save the full gradient vectors, parameter deltas and checkpoint hashes. They are then frozen for the remainder of T013-F. Do not recompute a direction because later results are unfavorable.

Verify before Stage B that the original checkpoint is unchanged, only predictor parameters differ in the three probes, Faster R-CNN and CLIP parameter/buffer hashes match the frozen references, and `.grad` state is clean. No optimizer is allowed during Stage B.

### Stage B — Eight matched evaluation cycles with an explicit null pair

Run exactly **8 retained matched cycles**. Each cycle evaluates all eight fixed episodes under four predeclared pairs:

- **null pair:** original predictor A vs an independent fresh copy of the same original predictor B;
- **joint pair:** original predictor vs the frozen joint-step checkpoint;
- **clean pair:** original predictor vs the frozen clean-step checkpoint;
- **corrupt pair:** original predictor vs the frozen corrupt-step checkpoint.

For odd cycles evaluate baseline first inside each pair; for even cycles evaluate probe/control-copy first. Use a fixed predeclared rotation of the four pair blocks across cycles so one probe is not always earliest/latest in the process. Do not reorder after seeing outcomes. Reset episodic ISP state exactly as in T013-C/D.

Retain for every episode/evaluation: source outer loss, `phi0`, terminal `phi3`, saturation, empty-support status, pair/cycle/order metadata, and detector/CLIP frozen-state checks. Repeated evaluation must not update predictor parameters. Save raw rows before any summary is computed.

For each cycle and for clean, corrupted and joint groups separately compute the paired loss effect

`delta_probe = mean(loss_probe - loss_original)`

and the same-position null effect

`delta_null = mean(loss_original_B - loss_original_A)`.

Also report the control-corrected effect

`delta_cc = delta_probe - delta_null`.

Keep all eight values; do not remove outliers. Report medians, min/max, sign counts, and the null-pair absolute range. This is a bounded repeatability diagnostic, not a claim of population-level statistical significance; do not manufacture confidence intervals from eight dependent cycles.

### Stage C — Predeclared interpretation rule

For any probe/group, call an effect **resolved beyond the matched noise floor** only if both conditions hold:

1. at least **7/8** control-corrected cycle effects have the same sign as their median; and
2. `abs(median(delta_cc))` is **greater than the maximum absolute null-pair effect** observed for that same group across the eight cycles.

Negative loss delta means improvement; positive means worsening.

Use this rule as follows:

- If the **clean-only** checkpoint has a resolved clean improvement and resolved corrupted worsening, and/or the **corrupt-only** checkpoint has a resolved corrupted improvement and resolved clean worsening, treat the already-observed negative local gradient cosine as a **functionally active source-objective conflict**. Stop longer training. A later task may test one fixed paired/identity-preserving source objective.
- If the joint and/or both group-specific fixed checkpoints show resolved improvement on **both** clean and corrupted groups without resolved cross-harm, then the negative local cosine is real but is not the immediate finite-step bottleneck at this scale. Do not add a conflict regularizer; the next research task should audit whether the existing 16-D predictor representation contains condition-specific signal sufficient for image-conditioned initialization.
- If probe effects do not satisfy the resolved-beyond-noise rule, or are asymmetric/mixed, report that the fixed microset remains measurement-limited. Do not change the method based on unresolved one-step deltas and do not increase repeats or microset size inside T013-F.

The T013-D common-mode result remains in force regardless of T013-F: do not implement bias deletion alone.

### Acceptance criteria

T013-F is ready for review when the plan precedes repeated outcomes; the three probe checkpoints are frozen exactly once and hashed; exactly eight retained matched cycles plus the null pairs are saved; raw episode rows and pair/order metadata are available; the predeclared 7/8 + null-range rule is applied without tuning; detector/CLIP isolation and predictor immutability during Stage B are verified; exact commands/tests/failures and results are appended to `coordination/CODEX_TO_CHATGPT.md`; and no unrequested method or data change is started.

**Stop after T013-F. Do not start T013-G, longer meta-training, a source-objective regularizer, bias removal, feature centering, predictor redesign, new deterministic kernel, validation/target/AP experiments, new cohorts, spatial ISP, gating or dose work until research review.**