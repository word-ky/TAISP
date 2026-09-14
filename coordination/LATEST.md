# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R045_T030A.md` — **R045 / T030-A**

R045 accepts T029-A as a protocol-compliant negative result and closes the exact fixed `alpha=2` full-91-class soft-sharpened pseudo-target family. The 60-image / 120-episode cohort was frozen before candidate outcomes; all 120 GT-free hard/soft records were committed in `4ec7df4` before the annotated reference run; supports/weights were identical; candidate/reference integrity and reverse/JVP parity passed; actual source revisions were recorded; detector state remained frozen; no AP, K-step runtime, target detector or post-outcome tuning occurred.

The frozen T029-A gate fails: `S_soft>0 = 63/120 overall, 31/60 corrupted`, only `2/4` block medians are positive, and mean `Delta=S_soft-S_hard` is negative despite slightly positive medians. The key mechanism diagnostic is that hard and soft gradients remain almost collinear (median cosine `0.9853` overall / `0.9890` corrupted) while the soft norm is only about `0.63x / 0.62x` hard. Power-2 softening therefore mostly attenuates the existing confidence-sharpening direction instead of creating a new task signal. Do not rescue this family with alpha/temperature, foreground/top-k mass, confidence/support, hard-soft blending, K/LR, or runtime/AP tuning.

T030-A asks whether a genuinely different detector-native signal works better. Precommit 60 completely fresh train2017 images / 120 clean+severity-2 episodes, excluding all 2951 cumulative source IDs through T029-A and all val2017; balance gamma/contrast/color-cast s2 exactly 20 each in four fixed 15-image blocks.

For each episode, use the unchanged original-view `score>=0.50/top20` detector predictions as detached pseudo boxes/classes. On the identity ISP-enhanced image compute the literal frozen Faster R-CNN native loss against those pseudo targets with fixed internal sampling seed `20260930`: `loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg`, with no component weights. Compare its 8-D gradient `g_native` to the unchanged current hard fixed-ROI gradient `g_hard`. Record the four component gradients only as diagnostics; they may not be selected or reweighted. Candidate code/data must remain annotation-free and all 120 candidate records/code must be SHA-pinned before any source-reference process loads GT.

Post-lock, compute the unchanged annotated source task gradient `t_s` and compare normalized utilities `S_hard`, `S_native`, and `Delta=S_native-S_hard`. Advancement requires the full frozen R045 conjunction: `S_native>0` >=80/120 overall and >=45/60 corrupted; `Delta>0` >=68/120 and >=35/60; positive overall/corrupted median Delta; >=3/4 positive block medians; nonnegative clean median Delta; and all leakage/pseudo-target/frozen-state/RNG/JVP/import-isolation checks passing.

No CLIP contribution, pseudo-box refinement, confidence weighting, component coefficient tuning, support/K/LR tuning, K-step AP, FCOS/SSD or deployment replacement is authorized. If the exact four-loss candidate passes, stop for review before runtime/AP. If it fails, close the exact unweighted pseudo-native objective without rescue tuning.