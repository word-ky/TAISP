# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R043_T028A.md` — **R043 / T028-A**

R043 accepts T027-A as a protocol-compliant negative reliability result. The 24-image/48-episode cohort and flip-commutation test were frozen before outcomes; all 48 original/flip candidate gradients and the overall median agreement were SHA-pinned in `0b1435b` before annotations/reference were loaded; Faster R-CNN remained frozen/eval; protected runtime modules stayed unchanged; all integrity/JVP checks passed. The exact symmetric flip-consensus gate nevertheless failed (`Delta_cons>0` only 24/48 overall and 10/24 corrupted; corrupted median `Delta_cons=-0.00174812`), and flip-gradient agreement failed as a reliability statistic (Spearman `+0.1054` overall, `-0.1235` corrupted). Close this exact flip-consensus/agreement family without threshold/view/weight/augmentation rescue sweeps or AP/K-step testing.

Retain one mechanistic signal: the unchanged current pseudo direction itself was task-descending on 30/48 episodes, including 18/24 corrupted versus 12/24 clean. The next question is therefore whether reliability can be predicted from a task-proximal pre-update gradient state rather than another hand scalar or flip heuristic.

T028-A is a **source-only gradient-state reliability capacity audit**, not a deployment/AP experiment. Precommit 180 fresh COCO train2017 images (360 clean/corrupted episodes), excluding the cumulative T027 source IDs and all val2017. Freeze 120 train images and 60 untouched holdout images with exactly balanced gamma_s2/contrast_s2/color_cast_s2 corruption assignments and four fixed holdout blocks.

For every episode, before any annotation/reference process runs, compute and SHA-pin exactly the fixed 21-D label-free feature vector specified in R043: normalized current detector-pseudo 8-D gradient, normalized generic-CLIP 8-D gradient, both log norms, their cosine, current pseudo loss and CLIP loss. No flip, memory, T010 scalar set, corruption hint, GT, target detector or post-update quantity is allowed. Only after all 360 candidate records are locked may the source-reference process compute `S_orig=<t_s,g_p>/||g_p||` and binary utility label `y=+1[S_orig>0]` (else `-1`).

Fit exactly one train-only standardized affine least-squares classifier with the fixed SVD pseudoinverse rule and zero decision threshold; no hyperparameter or threshold search. Run the predeclared 128 family-stratified pair-preserving label-permutation nulls. Evaluate only on the untouched 120 holdout episodes using the frozen R043 gates: AUROC >=0.70 overall and >=0.65 corrupted, trusted coverage 25–80%, >=+0.10 positive-utility precision gain overall/corrupted, positive trusted median utility, >=3/4 positive-gain blocks, and both AUROCs above the 95th percentile of their null distributions, with all integrity checks passing.

No protected detector/ISP/CLIP/current-Ours deployment code, K-step gated runtime, AP, FCOS/SSD, target/validation evaluation, MLP/ridge/logistic search, extra features or holdout-driven retry is authorized. Stop `NEEDS_REVIEW` (or `BLOCKED` on integrity/precondition failure) with complete receipts.