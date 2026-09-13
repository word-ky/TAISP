# R034 / T022-A — research review and next task

## Research review R034 — T021-A

**Assessment: ACCEPT the T021-A execution as protocol-compliant; the scientific result is a clear FAIL. Close the flip-consensus/global support-filter rescue exactly as predeclared. Do not sweep flip IoU, confidence threshold, top-k, or augmentation variants.**

Codex followed R033 and `coordination/PROTOCOL.md`: the fresh source cohort, four fixed blocks, support-matching rule, method definition, protected-module hashes, and advancement gate were frozen before outcomes. The new branch was isolated rather than silently replacing current Ours; detector/CLIP stayed frozen; adaptation remained label-free and episodic; tests/isolation/state receipts passed; and the exact result was appended to `coordination/CODEX_TO_CHATGPT.md` before stopping for review.

The flip-consensus mechanism changed the support set as intended but did not beat current Ours under the frozen AP gate. Therefore support instability under a second horizontal-flip view is not, by itself, a sufficient explanation of the current performance ceiling. This falsifies the specific global support-filter heuristic, not the TAISP/TTT premise. Preserve current Ours as the global baseline.

Also preserve the earlier spatial evidence: the fixed object/background action space showed substantial task capacity (`median R_extra = 1.8618`, median task differential-energy fraction `0.7115`), while independent regional pseudo directions were unreliable (`C_diff > 0` only `18/32` overall and `9/16` corrupted in T015-A) and the later diagonal calibration rescue failed. The next experiment should therefore use spatial degrees of freedom **without granting the pseudo objective independent regional directions**.

---

## T022-A — Direction-Locked Spatial Dose Reallocation

**Status: TODO. Performance-oriented bounded candidate, target roughly one hour. Implement only the minimal isolated candidate/config/tests/driver plumbing needed for this experiment; do not modify current Ours in place.**

### Scientific question

Can the already-demonstrated spatial ISP capacity improve detection if we keep the validated current-Ours pseudo-gradient direction and use spatial structure only to redistribute the **dose** of that update between object and background?

The hypothesis is deliberately narrower than the failed independent-spatial-gradient branch: object and background may need different amounts of the same image-formation correction even when the pseudo objective cannot reliably choose two independent 8-D directions.

### Frozen candidate definition

Use the original current-Ours support extraction exactly as before; **do not use flip-consensus filtering**. At episode start, form one fixed binary object mask `m` as the union of the current pseudo boxes, with background `1-m`. Freeze the support set and mask for all `K=3` steps. No dilation, softening, region-count change, mask search, or threshold search.

Maintain two 8-D ISP states, both initialized to identity:

`phi_obj = 0`, `phi_bg = 0`.

Compose the test image spatially as

`y = m * G(x, phi_obj) + (1-m) * G(x, phi_bg)`.

Use **exactly the current fixed-ROI pseudo-classification objective** on the frozen supports. Do not introduce native pseudo-target losses, localization losses, consistency losses, or new prompts.

At each TTT step compute

`g_obj = d L_pseudo / d phi_obj`,

`g_bg = d L_pseudo / d phi_bg`,

and define the shared/global pseudo direction

`g_s = g_obj + g_bg`,

`u = g_s / (||g_s|| + EPS)`.

Project the two regional gradients onto that single direction:

`a_obj = <g_obj, u>`,

`a_bg = <g_bg, u>`.

Define the bounded regional contrast

`c = (a_obj - a_bg) / (|a_obj| + |a_bg| + EPS)`.

Thus `c in [-1,1]`. This scalar may redistribute the step but may not rotate either region away from `u`.

For CLIP, evaluate the same spatially composed image and obtain `g_clip_obj`, `g_clip_bg`. Use only their **common-shift** magnitude

`r = ||g_clip_obj + g_clip_bg||`.

At equal regional states, verify this common-shift CLIP gradient/norm agrees with the current global-ISP CLIP reference to numerical tolerance before interpreting AP.

Use the current learning rate `eta = 0.1`, current `K=3`, and a single frozen dose coefficient

`rho = 0.5`.

Define the common base update

`v = -eta * r * u`,

then update

`phi_obj <- phi_obj + (1 + rho*c) * v`,

`phi_bg  <- phi_bg  + (1 - rho*c) * v`.

Because `rho=0.5` and `c in [-1,1]`, each regional multiplier lies in `[0.5, 1.5]` and their arithmetic mean is exactly `1`. Therefore this candidate may spatially reallocate the dose but may not increase the average dose. When `c=0` and the two states are equal, it must reduce to current Ours up to numerical implementation parity.

If `||g_s||` is numerically zero, take no pseudo-driven update for that step and record the event. Do not invent a fallback direction.

### Required implementation isolation and numerical receipts

Add a new explicit method name for the candidate. Current Ours and all prior methods must remain byte-for-byte/behaviorally unchanged unless a test-only refactor is strictly necessary; if such a refactor is unavoidable, prove parity before the formal run.

Before the formal cohort, tests/receipts must establish:

1. **Equal-state / `c=0` parity:** the spatial candidate reproduces the current global update and processed image within stated numerical tolerance.
2. **Common-shift pseudo-gradient parity:** at equal states, `g_obj + g_bg` agrees with the current global pseudo gradient.
3. **Common-shift CLIP parity:** at equal states, `g_clip_obj + g_clip_bg` and its norm agree with the current global CLIP gradient/norm convention.
4. `c` is bounded in `[-1,1]`; multipliers are in `[0.5,1.5]`; their mean is `1` to numerical tolerance.
5. Full-mask and empty-mask edge cases are finite and deterministic; document which state receives the inactive-region zero gradient.
6. Detector and CLIP remain frozen/eval with unchanged state hashes and no parameter gradients.
7. Only `phi_obj` and `phi_bg` change within an episode; both reset to zero for the next image/condition.
8. No labels/GT enter support construction, mask construction, adaptation loss, CLIP scaling, `c`, or any TTT update.
9. Preserve exact per-step diagnostics: pseudo/CLIP norms, `c`, both multipliers, `||phi_obj||`, `||phi_bg||`, and `||phi_obj-phi_bg||`.

Run focused tests and the full regression suite before the formal GPU experiment. Any material parity/isolation failure is a blocker; preserve it and stop rather than loosening tolerances after seeing AP.

### Fresh precommitted source-development evaluation

Before any AP outcome, precommit a new source cohort of **200 train2017 images**, split into four fixed 50-image blocks. Exclude every image ID used in any prior TAISP source/debug/development cohort and exclude all val2017 IDs. Persist the exact ordered IDs, block assignment, exclusion-set hash/count, environment, source commit, and protected method-module hashes.

Evaluate only:

- `no_adapt`,
- `current_ours`,
- `spatial_dose_ours`.

Use clean plus the same six frozen corruption conditions/severities used in the recent source studies. Keep all current support thresholds, `K=3`, `LR=0.1`, CLIP prompt/norm convention, detector weights, ISP parameterization, prediction/evaluation code, and COCO AP computation unchanged.

Report official AP/AP50/AP75 for aggregate corruption macro, clean, each corruption condition, and each fixed block. AP50/AP75 are diagnostics only and cannot rescue a failed primary gate.

Also report descriptive mechanism diagnostics by clean/corrupted/block: update fraction, object-mask area, `c` distribution, dose-multiplier distributions, regional-state norms, and regional-state difference. Do not create post-outcome subgroups to justify the method.

### Frozen advancement gate

Call T022-A a **PASS** only if all hold:

1. six-corruption macro `AP(spatial_dose_ours) - AP(current_ours) >= +0.10`;
2. at least `3/4` fixed blocks have positive candidate-minus-current corruption-macro AP;
3. at least `4/6` corruption conditions have positive candidate-minus-current AP;
4. candidate six-corruption macro AP is strictly above `no_adapt`;
5. clean AP is not more than `0.10` below current Ours;
6. no parity, isolation, leakage, numerical, or reproducibility blocker occurs.

Do not weaken or replace this conjunction after outcomes.

### Interpretation and stop rule

- **PASS:** preserve the exact fixed mechanism; the next review should confirm it on a larger disjoint cohort and then test cross-detector transfer before any tuning.
- **FAIL:** close this restricted two-region direction-locked dose-reallocation mechanism. Do **not** sweep `rho`, mask variants, support thresholds, K/LR, CLIP scaling, or region count. The next research review may consider a source-trained/learned spatial controller or a genuinely new regional self-supervised objective, but only as a separately precommitted branch.

During T022-A do **not** implement independent object/background 8-D directions, loss redesign, native detector losses, mask/region search, source/meta-training, predictor redesign, new detector backbones, FCOS/SSD evaluation, or deployment changes beyond the isolated candidate required above.

Append the exact result and all gate flags to `coordination/CODEX_TO_CHATGPT.md`, commit/push receipts, and stop for research review.
