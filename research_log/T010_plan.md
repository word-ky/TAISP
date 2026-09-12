# T010 predeclared offline scalar gating study

2026-09-12, before any T010 gated AP calculation. R012 `6012f1d`, pointer `1a4df0c`, accepts/closes T009 and authorizes this analysis only. Frozen input run `20260912-144439-taisp-t009-coco1000`, source `69bfb66`, final report `f3cb238`. No model/ISP/adaptation rerun or deployment edit.

## Scores and rank rule frozen before AP

Use only the 7,000 hybrid rows (1,000 images × seven conditions) from T009 samples, each at diagnostics[0], phi=0. The seven scalar scores are:

1. `clip_norm`: saved clip_gradient_norm.
2. `det_norm`: saved detector_gradient_norm.
3. `log_norm_ratio`: log((det_norm+1e-12)/(clip_norm+1e-12)).
4. `gradient_cosine`: dot(saved detector_gradient, saved clip_gradient) divided by their float64 Euclidean norms. Define zero-vector cosine as 0 and report its count explicitly; no unavailable direction is interpreted as agreement.
5. `pseudo_loss`: saved step-zero total (source pseudo loss in the hybrid diagnostics).
6. `support_count`: saved original source support count.
7. `support_confidence`: arithmetic mean of saved original support scores; empty support = 0, fixed before AP.

The inspected hybrid receipt and accepted trust_radius.py contain all seven fields. Audit their full-cohort availability before evaluation. If any exact field is missing, report/omit that score rather than inferring it from outcomes or rerunning inference.

One pooled rank reference is used across all 7,000 observations, not separate cutoffs by clean/corruption/family or block. This makes clean/family coverage a measured outcome. The candidate score itself never reads image ID, condition, target output, annotation, post-update phi or AP. For each score, rank high-first and low-first; break score ties by ascending image ID, then original saved hybrid-row ordinal for any remaining same-image tie. The ordinal is immutable receipt order, not a condition-dependent score. Select exactly floor(N*c) rows for c={.25,.50,.75}; full N=7,000 gives 1,750/3,500/5,250. The same once-fixed decisions are used in every detector and block. Never recalibrate thresholds per block. Save cutoff score and boundary tie identifiers, all ranks/decisions and hashes before loading labels.

This is exploratory pooled-cohort rank gating, not an independently calibrated deployment threshold. R012 explicitly reserves independent-cohort threshold validation for a later task. Every one of 7×2×3=42 configurations must remain in the report, plus no-adapt/full-hybrid anchors. No feature combination or best-threshold implementation.

## Reuse and increments

| Responsibility | Existing code/evidence | T010 increment and test |
| --- | --- | --- |
| Baseline AP | taisp.analysis.coco.subset_ap, replication.replication_ap; T009 official metrics | Reuse unchanged evaluator and five blocks; baseline tests 2 passed remotely in 1.29s. Local macro fixture 1 passed in 8.54s. |
| Scalar inputs | T009 hybrid diagnostics/support | Pure offline extraction and fixed rank/tie selection; known vectors, zero support, poisoned post-update/label fields, ties and exact coverage fixtures. |
| Prediction choice | Saved no-adapt and hybrid JSONs | Select whole image prediction lists in original image/native prediction order; preserve empty lists; test 0%/100% exact anchors, no input mutation, identical detector decisions and block filtering. |
| Evaluation | Official COCOeval and fixed blocks | CPU-only parallel independent detector/condition jobs; no model loading; smoke and full affected regression. |
| Reporting | T009 macro/safety conventions | All 5,544 detector/condition/group/config rows: 5,292 new gated evaluations +252 exact reused anchor evaluations. Independently check summary arithmetic. |

First remote baseline attempt used project root, where tests are not deployed, and exited4 (file not found). Retried the exact T009 release path; 2 tests passed. No code repair or environment change was required. Existing NVML warning is irrelevant to CPU-only COCOeval. Server reports128logicalCPUs and~498GB available RAM; use12CPU workers andoneBLAS/OpenMPthread each.

## Artifacts and timing

Stage A writes scores, distributions (clean/corrupted/each condition, aggregate and five fixed blocks), availability audit, full grid, cutoff metadata and decision matrix without reading annotations or target predictions. Commit these before gated AP inspection. Composition outputs retain input hashes, per-decision hashes shared across detectors, and composed-prediction hashes; original predictions plus decisions reconstruct every composed set without storing dozens of redundant GB.

Official COCOeval mutates input record dictionaries. Compose fresh dictionaries so evaluation cannot alter the frozen anchor records used by later configurations. This directly preserves reused inputs and anchor identity.

Coverage is both selected coverage and effective nonzero-phi coverage (empty-support hybrid can itself remain identity). Effective phi3 = saved phi3 when selected, zero otherwise. Report clean and six corruption conditions, including all five blocks.

Receipt-derived latency is explicitly an estimate. Let S=source_setup_seconds, A=hybrid adapt_seconds_3, G=hybrid diagnostics[0].step_seconds. Support-count/confidence use S+selected*A. Gradient/norm/cosine/pseudo-loss scores use S+G+selected*(A-G), reflecting the only available joint source+CLIP identity computation timing. This is a coarse estimate for pseudo loss and individual gradient scores because their separate costs were not recorded; no timing is invented or benchmark rerun. Anchors use S and S+A. Estimates exclude final target evaluation, IO and ranking overhead, and retain original terminal diagnostics cost. Do not claim target-specific end-to-end speedups.

## Decision and reporting

Follow R012 exactly: clean selected coverage <=.5, all three clean AP deltas >=−.10, both target aggregate corruption macro AP deltas >0, and both targets >=4/5 positive macro blocks. Report every passing and failing configuration; source evidence cannot replace target evidence. Clean/family distributions and AP deltas are descriptive; no localization/FP proxy analysis. No formal AP interval, no new inference, no T011/meta/new objective/gate implementation without research review.
