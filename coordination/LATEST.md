# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R042_T027A.md` — **R042 / T027-A**

R042 accepts T026-A as protocol-compliant and scientifically informative **FAIL**. The balanced 1280-entry source-GT ROI memory was frozen before audit outcomes, all 48 label-free candidate records/anchors were pinned before reference GT, protected detector/ISP/CLIP/deployment modules stayed unchanged, and integrity/JVP/isolation checks passed. The frozen gate nevertheless failed: `S_mem>0` on 26/48 overall and 18/24 corrupted episodes; `Delta_global>0` on 24/48 and 12/24; median `Delta_global` was slightly positive overall/corrupted but mean deltas were negative and only 2/4 block medians were positive. Close the exact clean-source nearest-memory objective without rescue sweeps.

R042 also preserves a correction to research history: authoritative T025 receipts are `S_geom>0=28/48 overall,15/24 corrupted`, `Delta_global>0=23/48,14/24`, median `Delta_global=-0.0195430 overall,+0.00938987 corrupted`, and 2/4 positive block medians. The earlier R041 summary quoted different numbers; both imply FAIL, and no historical file should be rewritten.

The next task, T027-A, is an **analysis-only horizontal-flip gradient-consensus reliability audit** that returns to the source-free deployment assumption. Use 24 brand-new train2017 images (48 clean/corrupted episodes, four blocks), excluding the complete cumulative T026 source/memory/audit manifest and all val2017. Before model outcomes, verify that the unchanged global ISP commutes with horizontal flip at the inherited numerical tolerance.

For every episode, independently generate unchanged score>=0.50/top20 Faster R-CNN supports on the original image and its horizontal flip, then compute the unchanged current fixed-ROI pseudo gradient in the same global 8-D ISP coordinates for each view. Do not match/filter supports across views and do not use memory, CLIP, geometry, feature/logit losses, confidence weighting or corruption-aware branches. Form the fixed symmetric consensus from the sum of the two normalized view gradients; zero/degenerate cases abstain. Pin all 48 GT-free support/gradient/agreement/consensus receipts before any annotation/reference process runs.

After the lock, compute the authoritative original-view task gradient and evaluate two predeclared questions: whether the symmetric consensus direction beats the original current-pseudo direction, and whether label-free view-gradient agreement predicts the usefulness of the original direction. The exact E1 consensus gate and E2 reliability-signal gate are frozen in R042; do not tune thresholds after outcomes. No AP, K-step runtime, FCOS/SSD or protected implementation/deployment modification is authorized. Stop `NEEDS_REVIEW` (or `BLOCKED` on a preflight/integrity failure) with complete receipts.