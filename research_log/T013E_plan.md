# T013-E pre-outcome deterministic replay plan

2026-09-13 04:52 +08. R019/74cbdce accepts and closes T013-D. Measurement only.
Commit this plan before any new real-model outcomes. Do not start T013-F afterward.

## Frozen inputs

SHA256:
- T013B manifest: 164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493
- T013B supports: 6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35
- T013C receipt: 7c35fb9e5f7b1c923c06f625ddf806d443e763b56aa52d8027594f0b95cce943
- T013C original predictor: a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120
- T013D artifact manifest: 6e689f25b061f94018499d25d889f29143e79f64bfa593339f9c907093a40afb
  This pins all twelve prior raw repeats and their summary. Verify used files against it.
- T013D repeatability summary: 21c4f4f80fa416500ba9adde20436bcb4d26dd63093284e0ff9e1035d31191d9
- Faster R-CNN: 258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384
- CLIP: a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f;
  openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.

Seed 20260913; constructor order source, CLIP, ISP, predictor. Verify saved original
predictor exactly. Episode order: 65088 clean/gamma_s2, 426525 clean/contrast_s2,
541157 clean/color_cast_s2, 129068 clean/gamma_s1. Image hashes and source annotations
remain in the pinned manifest. Reuse supports field-exact, with no recomputation.
Same float32 model execution, preprocessing/prompts/ISP/predictor/losses, K3,
inner lr .1, norm eps 1e-12, oracle sampling seed 20260913, outer SGD coefficient .001.

## Runtime and gate

Launcher exports CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python/torch.
Enable torch.use_deterministic_algorithms(True), cudnn.benchmark=False, threads1.
Keep other precision/runtime options unchanged. Log environment and model state hashes.
Exactly three no-update eight-episode repeats from fresh original copies, reset seed
20260913 each time. Retain all raw losses/components/phi0/phi3/gradients/norms/saturation.
Compare retained scalar/vector outputs bitwise, including signed zero, against repeat0.
Report first exact mismatch path/values/float bytes and do not run probes if any mismatch.
An unsupported deterministic operator stops the task with full traceback; no additional
environment workaround, CPU fallback, alternate kernel, relaxed tolerance or retry.

Compare repeat0 against prior T013D ranges for group losses, every phi3 coordinate/radius,
per-episode phi/head gradient norms, aggregate phi/head gradient cosines and norms.
Record inside/outside range descriptively; no outcome selects new settings.

## Conditional probes

Only after exact three-repeat gate passes: reuse T013C directions/one_step/evaluate.
Mean of eight head gradients (joint), four even/clean, four odd/corrupt, same float32
mean convention. Three independent original copies, exactly one SGD1e-3 update each.
Evaluate all eight after each update once; no second step or unfavorable-outcome retry.
Save each predictor checkpoint, parameter delta norm/per-tensor changed names, original
and feature-trunk equality, frozen detector/CLIP states and .grad=None, unchanged ISP.
Report every loss delta, before/after group loss, -1e-3 dot(g_episode,g_direction)
predictions and sign agreement; Pearson and average-tie-rank Spearman only n8 descriptions.
Group phi0/phi3 mean/max, saturation mean/max, empty supports from existing group_statistics.

Predeclared R019 decision: clean improves (<0) and corrupt worsens (>0) for clean-only,
and/or corrupt improves and clean worsens for corrupt-only => functional conflict.
All three improve both (<0) => local opposition not immediate finite-step bottleneck.
Joint helps one/harms other without group-specific cross-harm => asymmetric interaction.
Other patterns are explicitly unresolved. No tolerance or threshold added after results.

## Reuse and checks

Existing gradient_conflict owns gradients, geometry, mean directions, literal SGD,
first-order comparison; source_meta_smoke owns episodes, outer loss, frozen checks and
group statistics. New analysis runner only adds exact repeat gate, prior-range comparison,
runtime control, report correlations/decision. No production or existing test edits.
Baseline focused suite before edits; new focused tests for exact mismatch (including one
ULP/signed zero), rank correlation and fixed decision patterns. Full remote regression
then one bounded real run. All outcomes and any failure retained and reported to mailbox.

Stop after T013-E for review. No regularization, bias removal, centering, redesign,
longer training, LR/optimizer search, data expansion, target/validation/AP, spatial ISP,
gating/dose work, or T013-F is authorized by this task.
