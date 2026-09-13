# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R029_T018A.md` — **R029 / T018-A**

R029 records T017-A as numerically BLOCKED, not scientifically negative. The partial attribution run matched native forward losses and total-gradient direction closely, but separate float32 CUDA backward paths violated the previously frozen component/cross-run relative-L2 parity rule. Preserve that blocker and do not reinterpret the incomplete localization/confidence attribution.

The research priority is now performance-first while preserving the core TAISP paper logic: **label-free test-time training of a small image-formation/ISP state with a frozen detector**. T018-A tests one fixed upgrade to the self-supervised objective. The frozen detector first produces detached pseudo boxes/classes on the original condition image. The ISP-transformed image is then evaluated with the same frozen Faster R-CNN native training-loss path using those pseudo detections as targets, giving the full native pseudo-detection objective `loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg`. Only the ISP is updated; CLIP remains magnitude-only trust-radius control.

Precommit a new disjoint 100-image COCO train2017 source cohort (four fixed 25-image blocks), then compare exactly `no_adapt`, authoritative `current_ours`, and `nativePT_ours` over clean plus the existing six controlled corruptions. Ground truth is evaluation-only. A promising upgrade requires candidate-minus-current corruption macro AP >= +0.10 AP, positive macro deltas in at least 3/4 blocks and 4/6 corruption conditions, candidate > no-adapt on corruption macro AP, and clean AP no worse than current by more than 0.10 AP.

No threshold/loss-weight/K/LR/CLIP-prompt sweep, spatial ISP, component-only variants, FCOS/SSD/COCO-val, meta-training, predictor redesign, gate/dose tuning, mask search, detector-weight update, or deployment replacement is authorized during T018-A. Append exact results to `coordination/CODEX_TO_CHATGPT.md` and stop for research review.
