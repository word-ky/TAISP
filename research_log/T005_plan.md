# T005 pre-run contract

2026-09-12 Asia/Shanghai. R007 / 24114d2 accepts T004 and assigns T005.
IN_PROGRESS. Baseline local ISP/adapt11 passed6.38s; remote42 passed8.23s.
Initial local command used a nonexistent test filename; corrected after inventory,
no tests had run and no source failure was involved.

Reuse: unchanged FrozenDetector/model weights, ISP8D/hard clamp, adapt functional
SGD, COCO subset/corruptions, oracle loss, norm_match, environment and evaluator.
New code only for fixed-ROI adapter/support/loss, T005 driver and report.
Baseline reuse and AutoDL skills apply; no meta-learning or spatial ISP.

## Fixed support and losses (declared before smoke/full study)

Original corrupted (or clean control) base and horizontal-flip inference, eval and
no_grad. Each view retains score>=0.5, stable descending score top20; score ties
keep detector output order. Flip xyxy uses (W-x2,y1,W-x1,y2), continuous coordinates.
Map flipped detections back to base coordinates for same-foreground-class IoU>=0.5.
Sort candidate pairs by descending IoU, then base index, then flip index; greedily
take unused endpoints. This is deterministic greedy highest-IoU one-to-one matching,
not maximum-cardinality matching. Stable objects retain BOTH original view-specific
boxes; base ROI uses base prediction, flip ROI uses corresponding flip prediction.
Class is their common foreground label. No adapted predictions or labels enter.

det_pseudo: score-normalized base confidence weights, full91-class ROI CE at fixed
base boxes. det_stable: object weight proportional to arithmetic mean of the two
original confidence scores; normalize across objects. Equal0.5 base/flip CE average.
det_stable_js: same stable CE plus coefficient1.0 times weighted JS(Pbase,Pflip),
using natural logs and full ROI class distributions including background. No box
regression. Empty support => differentiable zero loss, exact zero ISP update.
No substituted pseudo-targets, no excluded fallback cases.

ROI path reuses model.transform on enhanced image without targets, backbone,
roi_heads.box_roi_pool/box_head/box_predictor. Scale detached fixed boxes by actual
resized width/height fractions before pooling. Bypass RPN/proposal/NMS entirely.
Frozen eval parameters and buffers; gradients pass to pixels/phi only. Flip view
is horizontal flip AFTER applying the same global ISP to the base input.

## Study and reporting

200 same IDs, six same corruptions PLUS clean, seed20260912, GPU0, threads1,
phi0=0, lr0.1,K3, external consistency/reg weights0. Variants in fixed order:
global_generic,det_pseudo,det_stable,det_stable_js. CLIP unchanged historical control.
Fresh annotated g_det once per image/case shared across four variants, analysis
only. Extra one-step gradient norm match to same-case global_generic, no labels in
scaling. Zero stays zero; raw cosine undefined for zero gradient. Report valid-only
cosine descriptively AND zero-coded cosine/positive/benefit over all observations
for aggregate paired comparisons, explicitly labeling that convention; no drops.

Save all fixed support/pairs/confidence, gradients, phi trajectories, losses1/3,
saturation, AP1/3 (56 evaluations) plus contemporaneous unadapted AP (7). 5600rows.
Corrupted overall pools six settings per image; clean reported separately. Reuse
2000 paired image-cluster bootstrap seed20260912, exploratory95% percentile CIs.
Support strata predeclared0,1-2,3-5,6+; report fallback, base/stable count, confidence,
match fraction stable/base and2*stable/(base+flip), outcomes by support stratum.
Report clean AP change and raw phi norm1/3, raw and norm-matched comparisons,
per-coordinate energy (zero vectors contribute0, explicitly counted), Taylor
dot-product predictions and measured loss. Fixed support not regenerated per step.
Timing: adaptation phase separately; deploy adds base setup only fordet_pseudo,
both view setup/matching forstable; record setup and adaptation peaks separately.
No prompt/lr/support/JS tuning on smoke or full data. Smoke2images software only.

Increment order: support/ROI adapter + focused real tests; objectives/fallback/reset
+ equation tests; study/report integration + full real suite and2-image smoke;
then fixed full run, collect all receipts, paired report/negative findings and
NEEDS_REVIEW. No automatic T006 before research review.

## Implementation checks

Fixed-ROI adapter real gate5 passed4.67s, run20260912-063355-taisp-t005-roi-tests.
The new empty-support test initially asserted bitwise equality to the raw image;
phi stayed exactly0 but existing ISP identity arithmetic differs by roundoff.
Corrected the assertion to exact unchanged ISP(phi0) output (and exact phi0),
not a widened tolerance or changed ISP. This is the no-update contract.
