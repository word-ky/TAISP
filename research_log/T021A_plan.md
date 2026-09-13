# T021-A pre-outcome plan: fixed horizontal-flip consensus supports

R033 / a5bae26 accepts T020-A as a scientific failure. Current Ours stays authoritative.
This study changes only support construction; no fitting, labels or gradient transport.

## Cohort

Reuse the disclosed T018-T020 readable/non-crowd-positive-box/area eligibility and
the cumulative manifest exclusion ledger. Exclude T020's 200 images and its inherited
836 previous source/development/debug IDs, all 5000 val2017 IDs and prior evaluation
IDs. Earlier small debug/scientific cohorts are included in the inherited T013H/T014A
ledger; all later two-image smokes are subsets of their respective frozen cohorts.
Select first200 by (sha256('20260922:'+decimalID),ID); four consecutive50-image blocks.
No detector or consensus outcomes enter selection. JPEG hashes, IDs, folds, annotation
hash and exclusion provenance are committed before outcomes. Model seed20260912 fixed.

## Exact candidate and numerical convention

For each existing clean/corrupted image, compute the original teacher prediction once
and the same frozen teacher's prediction on image.flip(-1) once. Map flip xyxy boxes
back using existing flip_boxes(width). On each view retain score>=.50 **before any
top20 truncation**. Enumerate same-class pairs with float32 original-coordinate IoU>=.60.
Sort eligible pairs by descending sqrt(original_score*flip_score), then original
detection index, then flip detection index; use Python float geometric means of stored
float32 scores for deterministic ordering. Greedily accept each pair only if both
detections are unused; rank retained pairs by that same order and keep first20.
Keep detached original box, original class **and original confidence score**. The
geometric mean is only for pair ranking: do not replace the existing loss weights
with averaged/geometric scores. The unchanged DetectorNativeLoss normalizes the
retained original scores exactly as current Ours. Do not average boxes/logits/classes.
Freeze support throughout steps0..3 diagnostics and three updates; never rematch.
Empty consensus => exactly zero phi and unchanged ISP(phi0) output; no fallback.

Baseline current Ours still uses original score>=.5, descending original score top20.
Consensus can retain original detections below that baseline top20: matching precedes
truncation as R033 specifies. Report original eligible count (before top20), baseline
current top20 count, all matched count, retained top20 count, retained/original-eligible
fraction (null when denominator zero), retained/base-top20 count ratio separately,
and zero-consensus incidence. Preserve original/flip predictions and pair receipts.

## Reuse and tests

Existing TAISP modules are the donor; no new dependencies. Reuse flip_boxes, frozen
Faster R-CNN, DetectorNativeLoss(det_pseudo), CLIP, ISP, adapt_clip_radius, corruption,
prediction/evaluation routines and shared run_t018a loop. Existing stable_pairs cannot
be reused unchanged: it sorts by IoU and pretruncates supports, whereas R033 freezes
geometric-mean priority and match-before-top20. Add a small independent consensus module.
Shared runner adds only candidate support setup, diagnostics and fixed assessment.
No edits to current loss, teacher preprocessing, adaptation, detector/CLIP/ISP modules.

Baseline on accepted release20260913-202732: detector_native/trust_radius/gradient_transport
12passed2skipped1.77s. Increment1: consensus eligibility/inversion/greedy/ties/top20
and loss-score preservation/empty identity tests. Pass before integrating shared runner.
Increment2: candidate runtime and support summaries, fixed+.10 assessment; focused tests,
full regression, then first2 frozen images CUDA K3 smoke with zeroAP. Formal same release,
200images x7conditions x3methods; 2800teacherforwards,2800adaptiveepisodes,
21predictionfiles,105officialCOCOevaluations. No source gradients/labels/fits.

All model forwards/gradients/adaptation and IoU computation run on A6000 CUDA float32.
Small deterministic pair sorting, diagnostics and officialCOCO aggregation use CPU.
K3/LR.1/EPS1e-12, score.5/top20/IoU.6, CLIP prompts/weights/magnitude rule, global8D ISP,
identity reset, model seed, threads1, all six corruptions remain fixed.
Model/state/hash/gradNone, functionalphi, detached support and no-rematching checks
remain. Save exact commands, source revision, environment, per-step phi/gradients.

Candidate latency includes original teacher setup plus extra flip forward/matching;
report extra setup separately. Shared current timing remains original teacher only.
Adaptation timing includes terminal diagnostics (four evaluations, three updates),
excludes final inference/state verification/AP; total teacher-inclusive adds setup.
Final detection is one original-orientation enhanced image, never a flip ensemble.

## Frozen decision and report

All required: candidate-current corruption macro>=+.10 AP, >=3/4 positive blocks,
>=4/6 positive conditions, candidate macro above raw, clean-current>=-.10 AP,
no support-matching/isolation/reproducibility/method-contamination blocker.
AP scale0-100. AP50/AP75 and support retention/zero incidence are diagnostics only.
Report overall/clean/corrupted/block/condition supports, phi norms/update incidence,
runtime overhead, all AP tables and paired prediction/gradient receipts.
Pass => developmental candidate pending larger fresh confirmation; fail => close
fixed horizontal-flip support filtering. No score/IoU/topk/augmentation sweeps,
fallbacks, altered confidence weighting, box averaging, source labels, native losses,
Q, spatial ISP, meta/predictor/gate/dose or FCOS/SSD/val work. Stop NEEDS_REVIEW
after report/raw receipts are committed, pushed and mirrored to A6000.
