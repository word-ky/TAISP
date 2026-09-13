# T020-A pre-outcome plan: cross-fitted orthogonal gradient transport

R032 / 2654048 authorizes this fixed study. T018/T019 native-loss branches remain closed.
Current Ours remains authoritative. No AP or task-gradient outcomes precede this plan commit.

## Cohort and formulas

Reuse T018/T019 readable, non-crowd positive-box/area train2017 eligibility. Exclude
T019's 200 images and inherited 636 earlier source IDs, all 5000 val2017 images and
the prior evaluation ledger. Select first 200 by (sha256('20260921:'+decimalID),ID).
Four consecutive 50-image blocks define image-level folds, shared by all seven conditions.
Selection seed 20260921 is independent of unchanged model/oracle seed 20260912.

Collect at identity, for every image and existing condition (six corruptions plus clean),
raw 8-D gradients gp of exact DetectorNativeLoss(det_pseudo), and gt of the existing
analysis-only oracle.detector_task_loss. Save vectors, norms, losses, detached teacher
supports, image IDs, blocks, hashes and isolation receipts. Source annotations are allowed
only in this separate collection process. Each image's gt serves fitting for other folds
and held-out diagnostics for its own fold; it never reaches its runtime adaptation.

In float64 form row vectors p=gp/(||gp||+1e-12), t=gt/(||gt||+1e-12).
For each held-out block use only the other 150 images (1050 episodes). If P^T T=U S Vh,
Q=U Vh, solving min ||P Q-T||F subject to Q^T Q=I. Do not force determinant +1.
EPS makes zero normalization defined: retain zero vectors in fitting and report them;
zero rows contribute exactly zero to P^T T. No exclusions based on gradient outcomes.
Save all train/held-out IDs and row indices, singular values, Q, determinant, fit residual,
and orthogonality error. Four independent matrices only; no optimizer or tuning.

Runtime accepts only Q and its fold provenance, never annotations or oracle gradients.
At every step row gq=gp @ Q, then reuse unchanged transfer_norm(gq,gclip,EPS), phi-=.1*update.
Store Q in float64; cast to runtime float32 on CUDA. Predeclare orthogonality max error
<=1e-10 in float64, <=1e-5 after float32 conversion; per-step norm rtol=1e-5, atol=1e-8.
These are numerical correctness tolerances, never performance-selection parameters.
Identity Q must reproduce current Ours exactly. Empty supports preserve exact zero phi.

## Reuse and increments

Existing repository is the donor; no external code/dependencies. Accepted release
20260913-182322-taisp-t019a-components baseline command:
`.venv/bin/python -m pytest -q tests/test_native_pseudo_target.py tests/test_detector_native.py tests/test_trust_radius.py`
(absolute existing interpreter): 10 passed, 2 skipped, 1.79s on A6000 environment.

1. Add closed-form fold fitting and runtime transport, with one optional private
   trust_radius hook. Public current Ours and all default paths stay unchanged.
   Focused tests: known Q recovery, reflection, row convention, norm, label/fold isolation,
   identity equivalence, frozen state and empty support. Pass before integrating driver.
2. Separate source-gradient collector uses existing oracle/ISP/teacher/image loading.
   Extend shared run_t018a only to load fold Q and select transport candidate. Reuse
   existing prediction, metrics, timing and state-check loops. No duplicated evaluator.
   Test collector interfaces/fit exclusion and fixed +.15 selection, then full regression.
3. Two-image CUDA smoke: first two frozen images assigned separate smoke folds, fit on
   the other image's seven episodes, K3 current/candidate, zero AP. This validates the
   complete collect-fit-runtime boundary but its matrices are never used in formal work.
   Formal collects all 1400 pairs, fits four 150-image maps, evaluates all 200 held-out
   images: 2800 adaptive episodes, 21 prediction files, 105 official COCO evaluations.

Keep detector/CLIP weights/prompts/teacher/current objective/ISP/corruptions fixed; K3,
LR.1, EPS1e-12, identity phi, score>=.5/top20, threads1, CUDA float32. Fit uses float64
on CUDA for tiny SVD; official COCO evaluation and small report aggregation use CPU.
Before/after source and CLIP hashes, state/eval/gradNone, teacher, phi and finite checks
apply. Save per-step raw/transported gradients/norms and Q fold reference. Adaptation
latency retains steps0..3 diagnostic convention; teacher-inclusive timing adds setup,
excludes fit/collection/final prediction/state copies/AP. Report fit/collection separately.

## Frozen assessment and deliverables

All six requirements: corruption macro >= current+.15 AP; >=3/4 positive held-out blocks;
>=4/6 positive conditions; macro above no-adapt; clean >=current-.10AP; no leakage,
isolation, norm or reproducibility blocker. All differences use 0-100 AP points.
Alignment is a diagnostic, never another performance gate. Report raw/transported
cosine to held-out gt at identity, overall/corrupted/clean/block/condition, improved
fraction; undefined zero-vector cosine is null with explicit counts. Report singular
values, determinants, orthogonality, pairwise matrix Frobenius distance/cosine, full
AP/AP50/AP75, phi/support/update/latency diagnostics. Preserve every raw receipt.
Pass => development candidate only; fail => close this fixed global linear transport
branch. No new cohort, capacity/ridge/coordinate search, FCOS/SSD/val, spatial ISP,
meta-initialization, gates/dose or T017 forensics. Append CODEX report, commit/push,
mirror A6000 and stop NEEDS_REVIEW for new research instruction.
