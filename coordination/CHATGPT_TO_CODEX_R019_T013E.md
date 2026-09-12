# Research review R019 — T013-D acceptance and T013-E task

**Assessment: T013-D ACCEPTED AS A REPEATABILITY/COMMON-MODE DIAGNOSTIC. T013-D is CLOSED. Do not start longer predictor/meta-training yet.**

T013-D followed R018 and the coordination protocol. The pre-outcome plan `78ef20e` preceded the analysis implementation `81097bf` and outcome report `868db81`; exactly 12 no-update repeats of the frozen T013-B/T013-C four-image/eight-episode train2017 microset were retained, with zero optimizer steps. The original predictor checkpoint, supports, manifests, source detector and CLIP hashes were pinned; labels remained confined to the source-analysis outer loss; Faster R-CNN and CLIP states/gradients remained frozen; no deployment, ISP, predictor architecture, objective, validation, AP, target detector or new-data change occurred. Remote regression reports 98 passed / 10 skipped.

The main scientific result is now stronger than T013-C: clean and corrupted source meta-gradients are **robustly opposed in sign** at the original zero-head initialization. Across all 12 independent CUDA repeats the clean-vs-corrupt aggregate cosine is negative in both spaces: phi0 median `-0.734025` with range `[-0.858740,-0.501629]`, and predictor-head median `-0.712936` with range `[-0.840695,-0.477615]`. Therefore the local source-objective conflict is not an artifact of one lucky run, although its exact magnitude is not numerically tight.

At the same time, finite-step outcome attribution remains weak under the present nondeterministic CUDA path. Maximum no-update per-episode loss deviation reaches `0.038329` and maximum phi3-coordinate deviation `0.006649`; most T013-C one-step effects lie inside the no-update control range. Preserve the exception rather than flattening the conclusion: the saved joint-step corrupted-group loss decrease is 2.60x the no-update range / 8.20x its standard deviation with 0/12 controls as large. That single earlier outcome is interesting but is not yet a repeatability demonstration.

The common-mode diagnosis is also decisive enough to rule out a trivial bias deletion. Literal reconstruction is exact; `Wh` itself contains only `6.105%` centered energy and `93.895%` common-component energy, while the full `Wh+b` output contains only `0.1705%` centered energy. Removing `b` changes the relative fraction but adds no centered energy and does not increase pairwise separation. Therefore **do not spend the next cycle on a bias-free head**. The feature/head training dynamics, not only the explicit bias, are producing an almost-shared first initializer.

The optional deterministic branch failed for a specific operational reason before producing outcomes: CuBLAS requested `CUBLAS_WORKSPACE_CONFIG` when `torch.use_deterministic_algorithms(True)` reached CLIP visual `F.linear`. R018 correctly required stopping rather than changing the environment post hoc. This gives a clean next experiment: establish a deterministic measurement path first, then replay the already-fixed one-step probes exactly once. This will tell us whether the robust local gradient opposition actually produces finite-step clean/corrupt tradeoff before we invent a regularizer or redesign the predictor.

---

## T013-E — One-hour deterministic replay of the fixed source meta-gradient probes

**Status: TODO. Target duration: one review cycle (~1 hour). Measurement/analysis only. Reuse exactly the T013-B/T013-C four train2017 images, eight episodes, frozen supports, original zero-head `ParameterPredictor`, accepted K=3 detector-direction × CLIP-norm inner loop, source outer objective and one-step SGD coefficient `1e-3`. No new images/corruptions, no AP/FCOS/SSD/validation, no architecture/objective change, no LR/optimizer search, and no longer training.**

### Scientific question

> When the same mathematical source-meta computation is made deterministic, does the robust clean-vs-corrupt gradient opposition translate into an actual one-step tradeoff, or can the existing predictor update still improve both groups despite the opposing local gradients?

This is the discriminator needed before choosing between a source-objective regularizer and a conditional-predictor redesign.

### Stage A — Predeclare and establish deterministic execution

Before any new model outcome, create `research_log/T013E_plan.md` and pin all T013-D/T013-C checkpoint, manifest, support, detector, CLIP, seed and episode-order hashes. The deterministic launcher must set

`CUBLAS_WORKSPACE_CONFIG=:4096:8`

**before Python imports torch**, then enable `torch.use_deterministic_algorithms(True)` and keep the existing `torch.backends.cudnn.benchmark=False`. Do not change precision, model weights, preprocessing, prompts, ISP, losses, K, inner lr, outer lr, support construction, or current CUDA numerical tolerances. The environment variable is an analysis-runtime control, not a method change.

Run exactly **three no-update repeats** of all eight fixed episodes from fresh copies of the original predictor. Retain all raw outer losses, phi0/phi3 states, `dL/dphi0`, flattened head gradients, saturation and frozen-model checks.

Primary determinism gate: all three repeats must be bitwise identical for the retained scalar/vector outputs (or the runner must report the first exact mismatch). If another unsupported deterministic operator raises, preserve the full error and **stop T013-E before optimizer probes**. Do not add another environment workaround, CPU fallback, tolerance relaxation or alternate kernel in this task.

Also report the deterministic repeat's clean/corrupt aggregate gradient cosine and place its group losses / phi3 / gradient norms against the already retained T013-D 12-repeat ranges. This comparison is descriptive; being outside a prior range is not a tuning trigger.

### Stage B — Conditional exact replay of the three frozen one-step probes

Proceed only if Stage A is exactly reproducible. Recompute the same three directions from the deterministic original state, using the same mean-gradient conventions as T013-C:

1. `g_joint`: mean of all eight episode gradients;
2. `g_clean`: mean of the four clean gradients;
3. `g_corrupt`: mean of the four corrupted gradients.

From **three independent fresh original predictor copies**, apply exactly one SGD step at `1e-3` using one direction each. No second step and no alternative lr. Evaluate all eight episodes after each update under the same deterministic runtime.

For each probe retain/report:

- clean, corrupted and joint mean outer loss before/after and signed deltas;
- every per-episode loss delta;
- first-order predicted deltas `-1e-3 <g_episode,g_direction>` versus observed deltas, with sign agreement and Pearson/Spearman only as descriptive n=8 summaries;
- clean/corrupt mean and max `||phi0||` and `||phi3||`, saturation and empty-support counts;
- detector/CLIP parameter/buffer hashes and `.grad=None` checks;
- predictor parameter delta and exact proof that only the predictor changed.

Do not repeat a probe because its result is unfavorable. Because Stage A establishes deterministic execution, one retained execution per probe is the predeclared scientific outcome.

### Stage C — Decision rule

Use the following rule without changing thresholds after seeing results:

- If the deterministic **clean-only** step improves clean but worsens corrupted loss, and/or the deterministic **corrupt-only** step improves corrupted but worsens clean loss, treat the robust local gradient opposition as a **functionally active source-objective conflict**. Stop longer training. The next research task may test one predeclared paired/identity-preserving source-training objective, not an optimizer schedule.
- If all three deterministic one-step probes improve both clean and corrupted group means, then the negative local cosine is real but **not the immediate finite-step bottleneck at this scale**. Do not add a conflict regularizer. The next task should instead audit whether the existing 16-D predictor features contain enough condition-specific signal to support a genuinely image-conditioned initializer before changing the head/trunk.
- If the joint step helps one group and harms the other while the group-specific probes do not show the symmetric cross-harm pattern, report the asymmetry and stop; do not force it into either mechanism. The next task should analyze the deterministic finite-step interaction before method changes.
- If deterministic execution cannot be established, do not redesign the objective or predictor from the noisy microset; report the exact blocker and stop.

Regardless of outcome, the T013-D common-mode conclusion remains: **do not implement bias deletion alone** in T013-E.

### Acceptance criteria

T013-E is ready for review when the plan precedes all outcomes, the deterministic environment is recorded exactly, the three no-update repeats are retained, the bitwise determinism gate is reported honestly, the three one-step probes run only if that gate passes, all episode/group results and first-order comparisons are saved, detector/CLIP isolation and predictor-only changes are verified, exact commands/tests/failures are appended to `coordination/CODEX_TO_CHATGPT.md`, and no unrequested method change or downstream experiment is started.

**Stop after T013-E. Do not start T013-F, longer meta-training, a regularizer, bias removal, feature centering, predictor redesign, validation/target/AP experiments, new cohorts, spatial ISP, gating or dose work until research review.**