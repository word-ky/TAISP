# T013-F pre-outcome matched checkpoint replay plan

2026-09-13T05:52:33.4678224+08:00. R020/7efdb89 accepts and closes T013-E. Measurement only.
Commit this plan before any repeated outcomes. Prefer exact retained T013-C checkpoints:
all three are available, so **no new gradient computation or optimizer call** is needed.

## Frozen inputs and method

SHA256:
- Original predictor: a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120
- Joint checkpoint: fec4271dcfda344023521c17070d117d08082664fdb6ddd6934adb47e727c634
- Clean checkpoint: 06e424abceb8004ac35538e41636814cd5ba5dc85bc4d191429a93dbd64b9cb8
- Corrupt checkpoint: b1edfca935f8e8dc90206f11b123e6d40524588cc4351f767d4af79dc8d5ba4e
- T013C receipt (full original gradients and three mean directions):
  7c35fb9e5f7b1c923c06f625ddf806d443e763b56aa52d8027594f0b95cce943
- T013B manifest: 164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493
- T013B supports: 6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35
- Source weights: 258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384
- CLIP weights: a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f;
  openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.

Existing T013-C checkpoint directory is run20260913-030301-taisp-t013c-conflict-conditioning/artifacts/audit.
No checkpoint is regenerated. Verify only head parameters differ from original, trunk
exactly unchanged, and saved direction times -1e-3 exactly reconstructs checkpoint head.
Retain saved full gradient vectors, direction/parameter deltas and file/state hashes
before the repeated stage. Source/CLIP reference parameter-buffer hashes and .grad=None
checked before and throughout evaluation. Keep all checkpoints immutable.

Seed20260913; original constructor order source, CLIP, ISP, predictor, verify checkpoint.
Explicit clip.eval() before early isolation (the accepted T013-E setup repair).
Same8episode order: 65088clean/gamma_s2, 426525clean/contrast_s2,
541157clean/color_cast_s2, 129068clean/gamma_s1. Image/annotation hashes from manifest.
Reuse supports field-exact. Same original-resolution RGB float32/preprocess/prompts,
8-D ISP, K3/lr.1/eps1e-12 hybrid, source outer loss and oracle sampling seed20260913.
Outer coefficient .001 describes the saved T013-C update; no new optimizer runs.

## Exactly eight cycles and fixed order

Normal T013-C/D CUDA runtime: deterministic algorithms False, cudnn.benchmark=False,
threads1, CUBLAS_WORKSPACE_CONFIG unset before Python. No precision, interpolation,
antialias, kernel or tolerance change. Seed20260913 reset before each eight-episode
evaluation; explicit RNG is held fixed, unavoidable CUDA execution variation remains.

Cycle numbers are 1 through8. Rotate the base blocks [null,joint,clean,corrupted]
left by (cycle-1) modulo4. Thus the full fixed schedule is:

| Cycle | Pair blocks in order | Inside each pair |
| --- | --- | --- |
| 1 | null, joint, clean, corrupted | baseline A, probe B |
| 2 | joint, clean, corrupted, null | probe B, baseline A |
| 3 | clean, corrupted, null, joint | baseline A, probe B |
| 4 | corrupted, null, joint, clean | probe B, baseline A |
| 5 | null, joint, clean, corrupted | baseline A, probe B |
| 6 | joint, clean, corrupted, null | probe B, baseline A |
| 7 | clean, corrupted, null, joint | baseline A, probe B |
| 8 | corrupted, null, joint, clean | probe B, baseline A |

Each baseline A is a fresh original copy; null B is a separate fresh original copy.
Other B copies come from the fixed saved checkpoint, with no state carried across calls.
Reuse source_meta_smoke/gradient_conflict.evaluate for fresh episodic ISP behavior.
Exactly8cycles x4pairs x2evaluations x8episodes =512 retained episode rows.
Save raw rows and cycle/pair/block/role/evaluation order before any effect summary.
Rows include losses/components, phi0/phi3, saturation, support/empty status and isolation.
Check predictors unchanged and .grad=None after every eight-episode evaluation;
original templates/source/CLIP/ISP states unchanged. Never retry unfavorable outcomes.

## Effects and fixed interpretation

For each cycle and each group (joint8, clean4, corrupt4), use logical roles regardless
of execution order: delta_probe = mean(loss_B - loss_A). Null is originalB-originalA
in the same cycle and same inside-pair order. delta_cc = delta_probe - delta_null.
Keep all8 values. Summaries: median (average two middle values), min/max/range,
negative/zero/positive counts, and null absolute min/max/range.

Resolved iff >=7/8 corrected effects share the sign of median AND
abs(median(delta_cc)) > max(abs(delta_null)) for that group over all8cycles.
Strict >, zero threshold, no tolerance, no adjustment after results. Zero median cannot
resolve. Negative means improvement. No confidence intervals or population inference.

Apply R020 in order: resolved clean-only clean improvement/corrupt worsening OR
resolved corrupt-only corrupt improvement/clean worsening => functional conflict.
Otherwise joint improves both OR both group-specific probes improve both, with no
resolved cross-harm => local opposition is not immediate finite-step bottleneck.
Unresolved/asymmetric/mixed patterns => measurement-limited; report each effect and stop.
Do not increase repeats/microset or rank optimizers. Bias deletion remains unauthorized.

## Reuse and checks

Existing evaluate/data/outer objective/frozen checks remain unchanged. Reuse model-state
hash routine from deterministic_replay (no call to that deterministic runner).
New analysis runner only loads fixed checkpoints, schedules evaluations and computes
the prescribed null-corrected rule. Baseline affected tests precede code; focused tests
cover fixed rotation, role-correct subtraction under reversed order, and 7/8/strict-null
threshold plus scientific decision patterns. Fullremote regression then bounded run.
Stop at T013-F NEEDS_REVIEW. No T013-G/training/objective/architecture/preprocessing/
deterministic-kernel/newdata/target/AP/spatialISP/gating/dose work automatically.

