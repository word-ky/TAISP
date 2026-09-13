# R033 / T021-A — research review and next task

## Research review R033 — T020-A acceptance and closure of global linear transport

**Assessment: T020-A is ACCEPTED as protocol-compliant, but the predeclared performance conjunction is a clear FAIL. Close the source-trained global orthogonal-gradient-transport rescue. Do not enlarge the matrix family, add ridge/nonlinearity, select coordinates, fit condition-specific maps, or turn the transport into an MLP.**

Codex followed R032 and `coordination/PROTOCOL.md`: the 200-image fresh train2017 cohort, four image-level cross-fitting folds, fitting rule, normalized Procrustes construction, held-out evaluation, protected method settings, and performance gate were frozen before outcomes. The implementation changes were limited to the explicitly authorized analysis/source-fit/runtime plumbing, tests, config, receipts, and reporting. Fold leakage checks, detector/CLIP/ISP isolation, state/hash checks, and orthogonal norm-preservation checks passed; there is no engineering blocker that rescues the scientific outcome.

The authoritative T020-A report gives six-corruption macro AP `45.7285369463` for `no_adapt`, `45.8688848296` for `current_ours`, and `45.8767699326` for `grad_transport_ours`. The transported candidate therefore improves over current Ours by only **`+0.007885103 AP`**, far below the frozen `+0.15 AP` advancement requirement. Only **`1/4` held-out folds** are positive and only **`3/6` corruption conditions** are positive. Clean candidate-minus-current is `-0.027302686 AP`, so clean safety is not the blocker. Gradient alignment also fails to show a robust cross-fitted improvement: overall median cosine falls from `0.252397539` raw to `0.223561864` transported, and only about `42.6%` of defined pairs improve. Since the orthogonal/numerical, leakage, and isolation checks pass, the correct interpretation is that a single low-capacity global source-learned rotation does not solve the task-alignment problem. Preserve the result as a useful negative; do not tune Q after seeing it.

Across T018–T020, replacing current Ours with native detector training losses, native-loss subsets, and now a global linear task-gradient transport has failed to produce a reproducible material improvement. The most defensible performance-oriented move is therefore to **preserve the current-Ours pseudo objective and TTT update logic, but improve the reliability of the pseudo supports themselves**. R032 explicitly allowed one separately predeclared teacher-consensus/stability objective before committing to higher-complexity spatial TTT.

---

## T021-A — Flip-consensus pseudo-support TTT

**Status: TODO. One bounded performance-oriented experiment, target roughly one hour of implementation/validation work plus the formal A6000 run. Preserve current Ours except for how fixed pseudo supports are selected.**

### Scientific hypothesis

The current detector-native fixed-ROI classification pseudo gradient has repeatedly been more reliable than replacement objectives, but its supports are produced by a single corrupted view. Under adverse appearance, false positives, unstable classes, and poorly localized detections can therefore inject noisy gradients into the ISP update. A prediction that survives a deterministic horizontal-view transformation is a low-cost, label-free estimate of teacher stability.

Test the narrow hypothesis:

> **Filtering current-Ours supports by same-detector horizontal-flip consensus improves test-time ISP adaptation without changing the pseudo loss, CLIP trust mechanism, detector weights, or ISP action space.**

This is not a new detector-training loss and not an ensemble at final inference. The second teacher view is used only to decide which detached supports are allowed to supervise the existing TTT episode.

### Stage A — precommit cohort and exact support rule before AP outcomes

Before running the formal experiment, create `research_log/T021A_plan.md` and commit all of the following.

Use **200 completely fresh COCO train2017 images** that do not overlap:

1. any prior TAISP source/development cohort through T020-A;
2. any image used in previous small debug/scientific cohorts where IDs are recorded;
3. any COCO val2017 image.

Freeze the ordered 200 IDs and split them into **four fixed 50-image blocks**. Use exactly the existing seven conditions: clean plus the six previously frozen corruptions/severities. Do not select images based on detector predictions or consensus retention.

For each test image/condition, construct `flip_consensus` supports exactly as follows:

1. Run the same frozen Faster R-CNN teacher on the original current test image `x` using the existing current-Ours preprocessing.
2. Run that same frozen teacher on the deterministic horizontal flip `F(x)`. Map predicted flip-view boxes back into the original image coordinates exactly.
3. On **each view independently**, retain detections with score `>= 0.50`. Do not change the teacher score threshold.
4. Candidate matches must have the **same predicted class** and original-coordinate box IoU `>= 0.60`.
5. Construct a deterministic one-to-one greedy matching. Rank all eligible pairs by descending **geometric-mean confidence** `sqrt(s_orig * s_flip)`; break exact ties by original detection index, then flip detection index. Traverse once, accepting a pair only if neither detection has been used.
6. For an accepted pair, define the frozen pseudo support using the **original-view box and original-view class**. Do not average boxes, scores, logits, or classes across views. The flip view is only a stability filter.
7. Rank accepted supports by the same geometric-mean confidence with the same deterministic tie break and keep at most **top 20**.
8. Detach/freeze this support set for the entire `K=3` episode exactly as in current Ours. Do not rematch after ISP updates.
9. If no consensus support remains, the candidate must perform **identity/no ISP update** for that episode. Do not fall back to unfiltered supports.

These constants (`0.50`, `0.60`, top20, horizontal flip, original-box retention, geometric-mean ordering) are frozen for T021-A. **No IoU sweep, score sweep, top-k sweep, alternate augmentation, box averaging, soft weighting, fallback rule, or post-outcome support rescue is allowed.**

### Stage B — preserve the current Ours update rule

Define `flip_consensus_ours` by changing **only** support construction. Once supports are frozen, use the authoritative current-Ours fixed-ROI detector-native classification pseudo loss unchanged.

Keep unchanged:

- frozen Faster R-CNN task model;
- frozen CLIP model and prompts;
- current differentiable **global 8-D ISP** and all parameter ranges/operators;
- identity ISP initialization;
- current-Ours pseudo-loss formula on frozen supports;
- CLIP gradient-norm transfer / trust-radius formula;
- `K=3` test-time steps;
- `LR=0.1`;
- all six corruption definitions/severities;
- per-image episodic reset and all isolation/state/hash checks.

Do not introduce detector native training losses, source labels, learned calibration, a gate, half-dose, spatial ISP, meta-initialization, gradient transport, or any extra trainable network in T021-A.

Minimal implementation code needed to construct/verify the flip-consensus support set is explicitly authorized by this task. Keep it modular so `current_ours` is bitwise/behaviorally unchanged. Add deterministic unit tests for horizontal box inversion, same-class/IoU eligibility, one-to-one greedy matching, tie breaking, top20 truncation, and zero-support identity behavior.

### Stage C — methods and formal evaluation

Evaluate exactly three methods on the same 200 images and seven conditions:

1. `no_adapt`;
2. authoritative `current_ours`;
3. `flip_consensus_ours`.

Use official COCO `AP`, `AP50`, and `AP75`. Primary advancement uses AP only. Preserve complete per-condition and per-block tables.

Also record, as **diagnostics only**:

- number of original-view supports before consensus;
- number retained after consensus;
- retention fraction;
- fraction of episodes with zero consensus supports;
- these quantities overall, clean/corrupted, per fixed block, and existing corruption conditions;
- mean/final `||phi||`, update incidence, and runtime overhead.

Do not use retention statistics to alter thresholds after results are visible.

### Frozen performance gate

Call T021-A a development **PASS** only if all hold:

1. six-corruption macro `AP(flip_consensus_ours) - AP(current_ours) >= +0.10`;
2. at least `3/4` fixed blocks have positive candidate-minus-current corruption-macro AP;
3. at least `4/6` corruption conditions have positive candidate-minus-current AP;
4. candidate six-corruption macro AP is greater than `no_adapt`;
5. clean `AP(flip_consensus_ours) - AP(current_ours) >= -0.10`;
6. no reproducibility, isolation, state/hash, support-matching, or method-contamination blocker occurs.

Report AP50/AP75 completely but do not add, remove, or substitute gates based on their outcome.

If multiple reruns are required because of an engineering crash, preserve the failed receipt and rerun only after fixing the engineering issue without changing the scientific constants. There is no scientific rerun with altered thresholds in this task.

### Interpretation and next decision

- **PASS:** treat horizontal-view consensus as a promising reliability mechanism, not yet as final evidence. The next task should replicate the exact frozen candidate on a larger fresh source cohort before any FCOS/SSD claim; if that replication passes, immediately test whether the enhanced images transfer to FCOS and SSD.
- **FAIL:** close the horizontal-flip support-filter/teacher-consensus rescue. Do **not** sweep IoU thresholds, confidence thresholds, augmentations, top-k, or matching variants on this cohort. The next research move should be an explicitly spatially structured TTT/ISP candidate, leveraging the already-established large spatial task capacity rather than continuing global-support heuristics.

### Deliverables and stop condition

Before outcome: commit the cohort IDs, block split, exact consensus rule, protected method hashes/settings, test plan, and advancement gate. Run focused tests, full regression, and a tiny CUDA smoke with no scientific AP interpretation. Then execute one formal A6000 study, save all prediction/AP/support diagnostics and hashes, append the exact outcome to `coordination/CODEX_TO_CHATGPT.md`, commit/push receipts, and stop for research review.

**Do not run FCOS/SSD or val2017, tune thresholds, modify current Ours, start spatial ISP, reopen native-loss/gradient-transport branches, or make any other performance-method change during T021-A.**
