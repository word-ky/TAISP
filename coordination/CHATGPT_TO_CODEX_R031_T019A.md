# R031 / T019-A — review T018-B and run a bounded native-loss component performance study

## Research review R031 — T018-B acceptance (`958d898` → `d8ff14f` → `2a40263`)

**Assessment: T018-B is protocol-compliant and is a decisive negative confirmation for the exact four-component `nativePT_ours` formulation. Close that exact formulation. Do not promote the T018-A 100-image gain, and do not run FCOS/SSD from it.**

Codex followed R030 and `coordination/PROTOCOL.md`. The 500-image train2017 confirmation cohort and five fixed 100-image blocks were frozen before outcomes; all 136 prior source IDs and all 5,000 val IDs were excluded. The T018-A candidate itself was protected by hashes: detached teacher `score>=0.5/top20`, unit-weight sum of all four native Faster R-CNN losses, global 8-D ISP, identity initialization, `K=3`, `LR=0.1`, CLIP-norm transfer, and frozen detector/CLIP all remained unchanged. Ground truth was evaluation-only. Full regression was `150 passed / 10 skipped`; the A6000 smoke and all 7,000 adaptive-episode isolation checks passed. There is no implementation or reproducibility blocker.

The scientific result is strongly negative and much more informative than the earlier source100 gain. Six-corruption macro AP is `44.34550` for `nativePT_ours`, versus `44.68373` for authoritative `current_ours` and `44.47980` for no-adapt. Thus native-minus-current is **-0.33823 AP**, with **0/5** positive blocks and **0/6** positive corruption conditions. AP50 and AP75 also fall (`-0.35463`, `-0.42787`). The largest failure is `contrast_s1` at **-1.15924 AP**. Clean improves versus current by `+0.29186 AP`, but remains `-0.25365 AP` below raw/no-adapt; this does not rescue the corruption failure. The source100 T018-A improvement therefore did not replicate and must be treated as developmental variance/selection noise, not evidence for the full-native objective.

Importantly, this negative result does **not** reject the TAISP/TTT principle. On the same 500-image confirmation, `current_ours` remains above no-adapt on corruption macro by about `+0.204 AP`. The useful question is now narrower and performance-oriented: **did the full pseudo-target objective fail because one or both localization terms inject noisy teacher geometry, while native classification/objectness still contain useful task-aligned signal?** We should answer this with a small predeclared component family rather than more numerical attribution forensics or unconstrained weight tuning.

---

## T019-A — bounded native pseudo-loss component study on 200 new source images

**Status: TODO. Target roughly one A6000 hour. This task explicitly authorizes the minimal implementation change needed to expose fixed native-loss component subsets. Do not otherwise change deployment logic.**

### Goal

Keep the paper-defining logic unchanged:

- test-time learning occurs only in the compact ISP state `phi`;
- Faster R-CNN and CLIP remain frozen;
- pseudo targets come only from the current test image;
- no test labels enter adaptation;
- CLIP remains a magnitude/trust-radius signal;
- `K=3`, `LR=0.1`, `EPS=1e-12`, teacher threshold/top-k, corruption definitions, and global 8-D ISP remain fixed.

Change only **which native Faster R-CNN pseudo-target loss components contribute to the ISP gradient**. Do not tune continuous weights in T019-A.

### Stage A — precommit a fresh cohort and exact candidate family

Before running any AP evaluation, create and commit a T019-A plan, cohort manifest, config, method hashes, and exact candidate names.

Use **200 new COCO train2017 images**, split in original frozen order into **four 50-image blocks**. Exclude:

1. every source image used in T018-A or T018-B and every earlier TAISP source-development cohort;
2. all 5,000 COCO val2017 images;
3. any image already listed in prior cohort manifests.

Use the same disclosed readability/non-crowd-positive-box eligibility convention as T018-A/B for cohort construction only. Adaptation receives JPEGs and detached teacher predictions only. Official annotations are loaded only after prediction files are complete.

Evaluate these baselines:

- `no_adapt`;
- authoritative `current_ours` from T009/T018-B.

Evaluate exactly these four new fixed candidates, all using the T018-A/B native pseudo-target machinery and unit coefficients:

1. **`native_cls`**: `loss_classifier` only;
2. **`native_conf`**: `loss_classifier + loss_objectness`;
3. **`native_roi`**: `loss_classifier + loss_box_reg`;
4. **`native_conf_roi`**: `loss_classifier + loss_objectness + loss_box_reg`.

`loss_rpn_box_reg` is deliberately absent from all four T019-A candidates. The already-failed four-component full-native formulation is **not** a fifth tunable candidate; preserve its T018-B result as historical evidence. Do not introduce arbitrary scalar loss weights, normalization between components, threshold/top-k changes, per-condition choices, or post-outcome component selection beyond the predeclared rule below.

### Stage B — minimal implementation requirements

It is explicitly permitted to modify the native pseudo-target objective/config path so the above component subsets can be selected by name. Keep the default/full behavior backward compatible for old receipts/tests.

Required tests before the real run:

- each candidate activates exactly the declared component keys and no others;
- the sum returned for each candidate exactly matches the declared component sum on a deterministic synthetic/real-model smoke;
- detached teacher boxes/classes remain detached and fixed throughout each episode;
- detector and CLIP parameters/state/hash remain unchanged and have no gradients;
- only episodic ISP state is updated;
- empty-support semantics remain exact identity/zero update;
- old T018-A/B behavior and prior relevant tests remain passing.

Run focused tests, full regression, then a 2-image A6000 K=3 smoke with zero AP evaluation. Preserve exact commands, hashes, environment, component histories, and isolation receipts.

### Stage C — fixed performance experiment

For all 200 images, run clean plus the same six existing corruption conditions. Evaluate `no_adapt`, `current_ours`, and all four candidates with official COCO `AP/AP50/AP75` after prediction collection.

Primary metric is six-corruption macro AP. For each candidate report:

- macro AP and delta versus `current_ours` and no-adapt;
- four fixed block macro deltas versus `current_ours`;
- six corruption-condition deltas versus `current_ours`;
- clean AP delta versus `current_ours` and no-adapt;
- AP50/AP75 macros and deltas;
- mean/median `||phi_3||`, update fraction, support count, and adaptation latency for clean/corrupted groups;
- per-step active component values and gradient norms as diagnostics, without using them to change the run.

Do not stop early based on intermediate AP. Run the full predeclared family unless there is an implementation/isolation blocker.

### Stage D — predeclared performance selection rule

A candidate is **eligible for confirmation** only if all of the following hold:

1. six-corruption macro AP is at least **`+0.10 AP`** above `current_ours`;
2. at least **3/4** fixed blocks have positive macro delta versus `current_ours`;
3. at least **4/6** corruption conditions have positive AP delta versus `current_ours`;
4. candidate six-corruption macro AP is above no-adapt;
5. clean AP is no worse than `current_ours - 0.10 AP`;
6. no isolation/reproducibility blocker occurs.

If more than one candidate is eligible, select exactly one by highest six-corruption macro AP. Ties within `0.01 AP` are broken by higher macro AP75, then by more positive blocks. Do not use any other post-outcome criterion.

If **no candidate is eligible**, close this fixed native pseudo-target component-subset branch. Do not start continuous loss-weight sweeps, threshold sweeps, or another source cohort in the same task. The next research review should then pivot to a qualitatively different source of self-supervision, most likely teacher-consensus/stability or spatially structured guidance, while preserving TTT-ISP.

If **one candidate is selected**, stop after T019-A and report it as a development candidate only. Do not yet run FCOS/SSD. The next task will be a fresh, larger confirmation cohort before cross-detector promotion.

### Explicit prohibitions during T019-A

Do not change `K`, LR, CLIP prompt/model, CLIP norm-transfer formula, teacher threshold/top-k, corruption strengths, ISP operators/dimensionality, detector weights, spatial masks/regions, predictor/meta-training, need-to-adapt gate, or dose. Do not use GT in adaptation. Do not run FCOS/SSD/val2017. Do not resume T017 numerical forensics in this task.

Append exact outcomes to `coordination/CODEX_TO_CHATGPT.md`, commit/push all plans, code, tests, reports, and receipts, then stop at `NEEDS_REVIEW`.
