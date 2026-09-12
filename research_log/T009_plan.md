# T009 predeclared frozen-method scale validation

2026-09-12: R011184a3a0 accepts/closes T008; pointer5885716 queues T009.
Scope: external validation only. Reuse accepted T007 source/loss/ISP/adapt functions unchanged.
No target gradients/oracle-loss study; labels used for COCOeval only. No proxy mining.

## Cohort before any T009 model execution

Use official COCO val2017, same annotation hash e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f.
Exclude historical T002–T006200 and T007200 IDs. From sorted remaining4600,
random.Random(20260914).sample(pool,1000); seed not previously consumed in project.
Retain random selection order; consecutive chunks of200 become block_1..block_5.
Sorted image_ids is presentation/execution order only; block membership never changes.
Commit complete manifest with JPEG hashes, annotation hash, both excluded manifest hashes,
explicit overlap counts0/0 before SSD/source/CLIP/FCOS T009 execution.
Download root: /home/liujianhua/wjq/TAISP/shared/coco1000_t009.

## Reuse map and increments

| Responsibility | Existing accepted code | T009 change | Focused evidence |
| --- | --- | --- | --- |
| Cohort download/hash | scripts/prepare_coco_subset.py | multiple exclusions and saved random-order blocks | deterministic two-exclusion/block fixture |
| Adaptation | taisp/tta/adapt.py, trust_radius.py; detector_native/CLIP/ISP | call unchanged functions and fixed configs | original tests plus real SSD-present/absent equivalence |
| Target output | FrozenDetector, fcos_target, prediction_records | SSD300-VGG16 COCO_V1 evaluation adapter | class map, freeze, repeat inference, serialize/COCOeval |
| Data/study | COCOSubset, T007 driver flow | K3-only three-detector loop; no oracle diagnostics | two-image end-to-end smoke after pins |
| Evaluation | subset_ap | same official COCOeval on aggregate and five fixed blocks | known-prediction/block fixture and real smoke |
| Reporting | prior AP/distribution reporting | all detector/condition/block deltas and fixed interpretation | hand-calculated macro/sign fixture |

Baseline reproduction already exists: T007 run20260912-105119,source0df7e13,
59real-model/regression tests and200-image completion. Fresh local relevant baseline:
3passed,1real-model skipped10.74s. Cohort increment:2passed0.57s.
Full fresh real-model regression required before full1000. No framework upgrade.

## Model/endpoint pins before scientific results

Source FasterRCNN/FCOS/CLIP and historical prompt bank unchanged. SSD exact
torchvision ssd300_vgg16 with SSD300_VGG16_Weights.COCO_V1, no substitute.
Use the installed torchvision implementation; explicit native inference defaults,
weight URL/SHA256, class mapping and transform settings will be committed in
T009_ssd_pin.json after loading and before full-cohort result inspection.
Official API reference: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html
Actual pinned server code/metadata governs version-specific settings.

Methods: no_adapt, global_generic, det_pseudo, det_pseudo_clip_radius.
Global8D phi0=0, lr.1, K3, hard clamp, eps1e-12, source original score>=.5 top20.
Six unchanged corruptions plus clean. Save phi1/2/3 and existing every-step
diagnostics including terminal gradient atstep3 (no fourth update).
AP only K3; no outcome-driven K choice. Same enhanced tensor sent to all detectors.
Source original output supplies support and no_adapt prediction, evaluation targets
only receive detached images under no_grad and never enter adaptation arguments.
Real compatibility check uses exact CPU repeats for adaptation (known CUDA backward
variability retained) and repeated GPU SSD inference under benchmark=False; checks
source support/phi/enhanced outputs with SSD absent/present, frozen parameters/buffers,
mapping to COCO category IDs, prediction serialization and official evaluator.

## Report and interpretation

84 official aggregate evaluations=3detectors*7conditions*4methods, plus420 block
evaluations=84*5. AP/AP50/AP75 values and deltas versus no_adapt/CLIP/raw, macro
unweighted mean of six corrupted AP deltas, all five block signs, positive-family
counts and clean deltas. No AP confidence intervals or per-image AP.
Retain all negative results; no tuning or block-dependent changes.
Report phi3/saturation/support/fallback, synchronized latency and peak allocated
memory (all models resident; not incremental deployment-only model memory), clean
and corrupted detector/CLIP norm ratios and scales. No T008 proxy search.

External support requires both independent targets' aggregate macro AP positive
versus no_adapt and raw, and >=4/5 positive hybrid-minus-no_adapt blocks each.
Otherwise follow R011 detector-coupling/relative-only/no-replication/clean-harm
branches explicitly; do not turn source-only gains into external validity.
No T010, new loss/gate/cap/predictor/spatialISP or meta-training automatically.
