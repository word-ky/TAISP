# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R041_T026A.md` — **R041 / T026-A**

R041 accepts R040/T025-A as protocol-compliant and scientifically informative **FAIL**. The 24-image / 48-episode exposure-pair ROI box-geometry audit preserved the required GT-free candidate → SHA-pin → source-reference ordering, made no AP/K-step/runtime-method change, and passed its integrity/JVP/isolation checks. The frozen scientific gate nevertheless failed: `S_geom>0` on 27/48 overall and 16/24 corrupted episodes; `Delta_global>0` on 25/48 overall and 12/24 corrupted; overall median `Delta_global=+0.0002156918`, corrupted median `Delta_global=-0.0021218577`, and only 2/4 block medians positive. Close this exact exposure-pair geometry-stability objective without rescue sweeps.

The next task, T026-A, is an **analysis-only clean-source ROI feature-memory alignment audit**. It is intentionally a different objective family and explicitly introduces a source-label-derived offline memory; therefore it is not training-free/source-free in the strongest sense. Source labels are permitted only to construct and freeze a clean source memory before audit outcomes. The audit/test image path itself must remain strictly label-free.

Build a deterministic balanced memory of exactly 16 clean train2017 GT object ROI box-head features per COCO class (1280 entries total), excluding the fresh audit cohort, all prior source/development/debug IDs and all val2017. Freeze/SHA-pin the complete memory and source manifest before audit candidate outcomes. Use 24 brand-new train2017 audit images, four fixed blocks of six, each with clean plus one frozen severity-2 corruption (8 gamma, 8 contrast, 8 color-cast; 48 episodes total).

For each audit episode, freeze the unchanged original-view score>=0.50/top20 Faster R-CNN supports. Use only each support's **predicted class** to retrieve the top-4 cosine-nearest normalized clean-memory ROI features within that class, average/normalize them into a detached anchor, and minimize the mean ROI box-head cosine distance from the processed object-region feature to that anchor. Background stays identity and only the object-region 8-D ISP gradient is analyzed. Do not use audit GT for retrieval/anchors, do not mix current pseudo/CLIP/flip/geometry objectives, and do not run AP or K-step adaptation.

All 48 candidate supports, masks, memory IDs/similarities, anchors, losses and exact 8-D candidate gradients must be completed and SHA-pinned in a GT-free process before a separate reference process can load audit annotations and compute unchanged `t_o`, `t_s`, `p_o`, `p_s`. The frozen advancement gate is: `S_mem>0` at least 30/48 overall and 15/24 corrupted; `Delta_global>0` at least 30/48 overall and 15/24 corrupted; positive overall and corrupted median `Delta_global`; at least 3/4 positive block medians; healthy balanced memory with zero audit overlap; and no leakage/retrieval/JVP/isolation/nonfinite/material-near-zero blocker.

If T026-A passes, commit receipts/report and stop `NEEDS_REVIEW`; runtime/AP requires separate authorization. If it fails, close this exact clean-source nearest-memory ROI feature objective without sweeping memory size/top-k/ROI layer/projection/source selection/threshold/mask/K/LR or blending prior objectives. Do not modify protected implementation/deployment modules in this round.
