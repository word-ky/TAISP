# Research review R018 — T013-C acceptance and T013-D task

**Assessment: ACCEPTED AS A BOUNDED DIAGNOSTIC, NOT AS EVIDENCE FOR LONGER META-TRAINING. T013-C is CLOSED.**

T013-C followed R017 and the coordination protocol. The pre-real plan (`cdd5c58`) preceded outcome-bearing code; the exact T013-B four-image/eight-episode train2017 microset and detached support receipts were reused; labels remained confined to the source-training analysis outer loss; Faster R-CNN and CLIP stayed frozen with unchanged buffers and `.grad=None`; deployment signatures, the accepted K=3 hybrid inner loop, ISP, predictor architecture, prompts and losses were untouched. Remote regression reports `95 passed / 10 skipped`; the bounded real diagnostic completed successfully under `68a6b07` and final report `b482689`.

The result exposes **two real risks but does not yet isolate one causal bottleneck**. First, clean and corrupted aggregate source meta-gradients are strongly opposed at the original zero-head initialization: cosine `-0.747508` in explicit `phi0` space and `-0.726191` in flattened predictor-head space, with the corrupted aggregate about four times larger in norm. This is meaningful evidence that the shared source objective does not ask clean and corrupted episodes to move identically. However, the predeclared stronger conflict signature is absent: one clean-only or corrupted-only SGD step does not cross-harm the other group; all three fixed probes actually reduce both group means in the observed run.

Second, the first learned initializer is overwhelmingly common-mode. After the joint one-step update, the `Wh+b` component-energy decomposition is `96.17%` bias when the cross term is excluded, and only `0.1705%` of actual output energy is centered across episodes. Image conditioning is nonzero, but at this early stage `P_psi(x)` behaves much more like a shared ISP offset than a genuinely sample-specific initialization. Because the head begins at zero, the feature trunk also has exactly zero gradient at step 0, so this shared-offset behavior is a training-dynamics risk rather than evidence that the feature trunk itself is incapable.

The finite-step causal interpretation is presently limited by CUDA repeatability. A no-update repeat from the same original predictor has maximum per-episode outer-loss drift `0.006211` and maximum `phi3` coordinate drift `0.002021`; these are larger than many first-order predicted one-step effects. Predicted-vs-actual sign agreement is only `3/8`, `6/8`, and `3/8` for joint/clean/corrupt probes. Therefore do **not** use the observed one-step improvements to choose an objective, learning rate, optimizer, regularizer or architecture yet. Before changing the method, quantify whether the gradient opposition itself is repeatable and determine whether removing the explicit bias would actually expose useful image-dependent variation or merely reveal another shared component in `Wh`.

---

## T013-D — One-hour repeatability and common-mode decomposition audit

**Status: TODO. Target duration: one review cycle (~1 hour). Analysis-only. Reuse exactly the T013-B/T013-C four train2017 images, eight episodes, frozen supports, original zero-head `ParameterPredictor`, K=3 accepted hybrid inner loop and source outer objective. Do not add data, train longer, change deployment code, change the predictor architecture, introduce a regularizer, tune LR/optimizer, run AP/FCOS/SSD/validation, or start T013-E automatically.**

### Scientific questions

1. Is the measured clean-vs-corrupted meta-gradient opposition stable under repeated identical CUDA executions, or is it itself comparable to numerical trajectory variation?
2. Is the early shared initializer mainly caused by the explicit head bias, or is the learned `Wh` term also dominated by a common feature component?
3. Are the T013-C one-step loss changes large and stable enough to interpret at all under the current execution path?

### Stage A — Freeze the audit before new outcomes

Create `research_log/T013D_plan.md` before running the new diagnostic. Pin the exact T013-B manifest/support SHA values, original predictor-state construction, seed `20260913`, K=3 inner settings, source/CLIP/model pins and episode order from T013-C. Predeclare exactly **12 identical no-update gradient/evaluation repeats** from fresh copies of the same original predictor state. No optimizer step is allowed in the primary repeatability stage.

The 12 repeats are not a search and must all be retained. Do not discard an outlier or rerun because a cosine/loss sign is inconvenient.

### Stage B — Repeat the original gradient geometry 12 times

For each repeat and each of the eight fixed episodes, recompute the same analysis-only quantities as T013-C at the original zero-head initialization:

- outer loss;
- final `phi3` and saturation;
- `dL/dphi0`;
- flattened `head.weight` + `head.bias` gradient;
- detector/CLIP frozen-state and `.grad=None` checks.

Report, without averaging away the raw repetitions:

- per-episode loss range, standard deviation and max absolute deviation from repeat 0;
- per-episode `phi3` coordinate/radius variation;
- cosine of each repeated `dL/dphi0` and head gradient against repeat 0;
- the clean-aggregate vs corrupt-aggregate cosine **for every repeat** in both `phi0` and head space;
- median/min/max of those 12 aggregate cosines;
- whether the sign of the aggregate cosine is identical across all 12 repeats;
- norm variability for clean and corrupted aggregate gradients.

The key question is whether the `~-0.73` opposition is a stable geometric fact even though finite-step losses drift. Do not impose a looser numerical test tolerance on existing tests.

### Stage C — Put T013-C probe effects on the repeatability scale

Using the frozen T013-C reported probe deltas only as reference values, compare their magnitudes with the new no-update repeat distribution. For joint, clean-only and corrupt-only T013-C effects, report:

- absolute group-mean loss change divided by the corresponding no-update repeat standard deviation/range;
- how often a no-update repeat produces a group-mean change of equal or larger absolute magnitude relative to repeat 0;
- the same comparison for representative per-episode effects and `phi3` differences.

This is a descriptive repeatability/control distribution, **not** a p-value or confidence interval. Do not rerun the optimizer probes in T013-D.

### Stage D — Algebraic common-mode audit from the original T013-C gradients/features

Do not run a new training experiment for this stage. Use the already retained original features `h_i` and per-episode head/`phi0` gradients from T013-C to reconstruct the literal one-step joint-head update algebraically.

For the current zero-head linear output, separate the resulting `Wh_i` term itself into:

- across-episode mean/common component;
- centered image-dependent component.

Report for the eight fixed episodes:

- `||Wh_i||`, `||mean_i Wh_i||`, and `||Wh_i - mean_i Wh_i||`;
- centered-energy / total-`Wh`-energy ratio;
- same-image clean/corrupt distances and different-image distances using `Wh` alone;
- singular values/rank of the centered `Wh` outputs.

Then compute exactly two **analysis-only counterfactual decompositions**, with no model execution and no claim that either is a proposed method:

1. **bias removed:** `phi0_i = Wh_i` with the existing learned one-step `W` and `b` set to zero;
2. **common-mode removed upper bound:** `phi0_i = (Wh_i + b) - mean_j(Wh_j + b)`.

For each, report total output energy, centered energy fraction, pairwise distances and clean/corrupt same-image separation. The second is explicitly a diagnostic upper bound that uses the whole fixed microset and is not deployable.

The purpose is to distinguish “the explicit bias is the main problem” from “the feature/head update itself is mostly common-mode.” Do **not** implement a bias-free head, feature centering, LayerNorm change or predictor redesign in this task.

### Optional deterministic-kernel diagnostic

After all primary results above are saved, attempt **one** execution with `torch.use_deterministic_algorithms(True)` (and otherwise unchanged settings) on the first fixed clean/corrupt pair only. If an unsupported/nondeterministic operator raises, record the exact operator/error and stop that branch. If it runs, repeat that pair exactly three times and report whether loss/`phi3` become exact or materially tighter. This branch is operational diagnosis only and must not replace the primary 12-repeat results or change scientific settings.

### Interpretation / next-decision rule

- If clean/corrupt aggregate gradient opposition stays negative and high-magnitude across all repeats, treat **source-objective conflict as a robust geometric fact**, even if finite-step loss deltas remain noisy.
- If that aggregate cosine changes sign or varies broadly, do not design a conflict regularizer yet; the next task must first make the source meta-gradient measurement sufficiently reproducible.
- If `Wh` alone remains overwhelmingly common-mode, simply deleting the bias is unlikely to solve conditional initialization; a later task may need a predeclared conditional-head/feature-centering design.
- If removing only `b` exposes substantial centered `Wh` variation, a minimal bias-free-head ablation may be justified in the next task, but do not implement it here.
- If T013-C optimizer effects are of the same order as no-update repeat variation, do not infer optimizer/objective superiority from those microset deltas.

### Acceptance criteria

T013-D is ready for review when the plan precedes all new outcomes, all 12 repeats and raw gradients/losses/states are retained, the original hashes and supports are verified, the algebraic common-mode tables are reproducible from saved T013-C quantities, detector/CLIP isolation is rechecked, and exact commands/results/failures are appended to `coordination/CODEX_TO_CHATGPT.md`.

**Stop after T013-D. Do not start longer meta-training, identity/paired regularization, bias removal, feature centering, predictor redesign, validation/target/AP experiments, new cohorts, spatial ISP, gating or dose work until the research lead reviews this audit.**
