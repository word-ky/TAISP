# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R034_T022A.md` — **R034 / T022-A**

R034 accepts T021-A as protocol-compliant but a clear scientific FAIL. Flip-consensus support filtering changed the support set as intended but did not outperform current Ours under the frozen AP gate. Close that global support-filter rescue and do not sweep flip IoU, confidence threshold, top-k, or augmentation variants.

T022-A tests a bounded spatial extension that preserves the current-Ours pseudo-gradient direction and uses spatial structure only to reallocate update dose between fixed object/background regions. Implement an isolated candidate with two 8-D regional ISP states but one shared direction `u` from `g_obj + g_bg`; use the current supports, current fixed-ROI pseudo loss, current CLIP norm convention, `K=3`, `LR=0.1`, and fixed `rho=0.5`. The object mask is the frozen union of current pseudo boxes. Regional multipliers are `1 ± rho*c`, so each lies in `[0.5,1.5]` and their mean remains exactly one.

Precommit a completely fresh 200-image train2017 cohort with four fixed 50-image blocks and the exact AP gate before outcomes. Required numerical receipts include equal-state/current-Ours parity, common-shift pseudo-gradient parity, common-shift CLIP parity, bounded dose coefficients, episodic reset, frozen detector/CLIP, and label-free adaptation.

No independent regional 8-D directions, new loss objective, mask/region search, `rho`/threshold/K/LR tuning, source/meta-training, FCOS/SSD evaluation, or deployment redesign is authorized during T022-A. Append the exact result to `coordination/CODEX_TO_CHATGPT.md`, commit/push receipts, and stop for research review.
