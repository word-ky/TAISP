# T003 pre-run contract

2026-09-12 (Asia/Shanghai), research assignment R004 edf75c0. T002 CLOSED;
T003 IN_PROGRESS. No active remote jobs at start. Baseline: current final T002
release real suite 27 passed in 6.24s; local adapt/loss/linear suite 12 passed
in 7.43s. Continue implementation-method-from-baselines and autodl-workflow.

## Fixed scientific settings (before new experiment results)

Use exactly the final T002 200 IDs, six corruptions, frozen CLIP ViT-B/32 revision,
FasterRCNN COCO_V1, tensor preprocessing and hard-clamped eight-coordinate ISP.
Same phi0=0, SGD lr=0.1, K=3, consistency/regularization=0, paired native oracle
sampling seed=20260912+image_id. No prompts/lr/temperature tuned on these images.

Positive bank unchanged. Split the six existing negative prompts into consecutive
pairs: darkness (0:2), low contrast/haze (2:4), color cast (4:6). Normalize each
prompt feature, average within a bank, normalize each bank vector. For the original
corrupted image z=normalize(CLIP(x)), w=softmax(z @ negative_banks.T / 0.05).
Temperature 0.05 is fixed a priori; no use of CLIP learned logit scale.
Detach original z, weights, direction and masks for the whole episode.
d=normalize(t_positive - sum_m w_m t_negative_m), without renormalizing the
weighted negative mixture. Generic baseline keeps its original T002 formula.

Coordinate order: gamma, red_gain, green_gain, blue_gain, contrast, brightness,
tone, sharpening. Masks: darkness [1,0,0,0,0,1,1,0]; contrast [0,0,0,0,1,0,1,0];
color [0,1,1,1,0,0,0,0]. m=w@M, no norm rescaling or compensation for gate size.
The effective update is phi <- phi - 0.1*(m*gradient). Report both raw objective
gradient and effective update gradient; alignment/Taylor use the effective one.

Comparisons: reuse final T002 generic and clean/corrupted AP receipts unchanged;
run soft direction, oracle-family direction, soft direction + soft gate,
soft direction + oracle-family gate (isolates gating), oracle-family direction
+ oracle-family gate (joint best-case identification). Oracle selection exists
only in taisp.analysis; the deployable conditioner accepts only the image.
Oracle is an identification diagnostic, not a mathematical performance bound.

Offline Stage A uses only saved T002 gradients/outcomes. Coordinate contribution
is g_det*g_sem; exact sign agreement includes zero/zero (also report nonzero
pair count/agreement). Energy is g_sem_j^2/sum_j g_sem_j^2 per sample, then mean;
outside-subspace norm fraction is norm(g_out)/norm(g). Saturation buckets, fixed
before analysis: [0,1%], (1%,5%], (5%,10%], (10%,100%]. Cross-tab initial and
post-one-step saturation, count harm/delta>0, benefit/delta<0, zero separately;
also cross-talk within buckets. This is descriptive, not causal conditioning.

## Reuse map and increments

- Stage A: new offline arithmetic/report script, reuse final samples.jsonl and
  eight-coordinate order; verify with hand-computed coordinate/outcome examples.
- Stage B: retain FrozenCLIPEncoder/CLIPTensorPreprocess/SemanticDirectionLoss;
  add text-bank conditioner with detached episode inference. Test explicit
  weights/direction equations, frozen state, nonzero phi gradients and reset.
- Stage C: narrow optional coordinate_gate in adapt; default path unchanged.
  Test masked coordinates exactly unchanged and masked one-step equation.
- Driver: new analysis orchestration, reuse T002 data/corrupt/oracle/AP/metadata/
  physical_vector/summarize and linear-analysis helpers, no new evaluator.
  Run real model suite and tiny two-image smoke before full fixed study.
- Report all six cases, all variants: cosine mean/median/positive, measured loss
  benefit, Taylor sign/correlations, AP1/AP3, own/generic CLIP loss, raw/physical
  trajectories, saturation, per-episode timing including conditioning, GPU memory.
  Preserve raw receipts and compare inference weights across known synthetic
  cases only in analysis. Exploratory 2000-image-cluster bootstrap CIs seed20260912
  reuse T002 helper; paired deltas where useful. No sample deletion or tuning.

Source attribution: all reused code is this repository's verified T001/T002
implementation. No external code port, new model, learned prompts, predictor
training, meta-training, smooth clamp or detector updates.

## Observed-failure amendment before full run (03:05)
Smoke failed on strict cached-detector-gradient reuse. Four repeated native detector backwards in one process gave identical forward loss0.3455415964 but max gradient differences2.36e-5 to7.35e-5 between repeats; cross-run smoke mismatch1.08e-4. No prompts/temperature/lr changed. Repair: compute one fresh initial g_det per image-condition and share it across ALL six variants; rerun generic direction within T003. Stage A still uses original T002 receipts. Only clean/corrupted AP reused (same deterministic forward); report T003-vs-T002 generic numerical differences and original gradient discrepancy. No tolerance widening. This removes cross-run mixing of gradient evidence and supplies contemporaneous paired comparisons.
