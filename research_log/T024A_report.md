# T024-A — NEEDS_REVIEW; both candidates FAIL

R039/29bd004. This analysis-only audit closes the fixed ROI flip-equivariance objective family. No candidate is nominated. No AP, FCOS/SSD, adaptation steps, training or deployment change was run.

## Frozen protocol and ordering

Plan/cohort commit `3a7e764`; candidate code `bf78bc5`; complete candidate receipts `9be51ee` committed and pushed before reference code/run `638b763`. Candidate interpreter loaded no oracle/reference module. Only the separate cohort-preparation process used annotations for inherited noncrowd valid-box eligibility; candidate input contained image metadata and conditions only.

16 fresh train2017 images, seed20260924, excluding1436 prior source/development/debug IDs and5000 val IDs (6436 total). Ordered JPEG hashes and full exclusion ledger are committed. Four blocks of4images;32episodes=16clean+6gamma_s2+5contrast_s2+5color_cast_s2.

Original FasterRCNN supports only: unchanged score>=.50, stable descending top20. The binary union mask is unchanged. Background identity; object8D state is only an analysis variable at identity. No flip teacher or matching. ROI pair index preserves original support order. Feature point:1024D box_head output before box_predictor. Feature loss is mean1-cosine; logit loss is mean symmetric JS atT1. No confidence reweighting or extra objective.

Accepted common8-column ISP JVP, float32 ISP and independent float64 global/object/background reductions onCUDA; exact helper equality tested against A1. Same helper/ISP/mask for candidates and reference. Source oracle is unchanged native four-loss sum with seed20260913; current pseudo reference is unchanged det_pseudo loss. eps1e-12. Nonempty candidate gradient norm<=1e-12 was prospectively declared a blocker; none occurred.

## Frozen primary results

|Metric|Feature equivariance|Logit equivariance|Required|
|---|---:|---:|---:|
|S_c positive overall|19|19|>=20/32|
|S_c positive corrupt|10|11|>=10/16|
|Delta_global positive overall|17|16|>=20/32|
|Delta_global positive corrupt|8|7|>=10/16|
|Median Delta_global overall|0.0025144030470974346|-0.007272612365850152|>0|
|Median Delta_global corrupt|-0.0004964978183184029|-0.024653052599886323|>0|
|Positive block medians|2/4|1/4|>=3/4|
|Frozen conjunction|FAIL|FAIL|all required|

S_c is dot(t_o,g)/(|g|+eps); Delta_global subtracts dot(t_s,p_s)/(|p_s|+eps). Positive clean or diagnostic Delta_obj does not substitute for the frozen conjunction. Both corrupted median improvements are negative.

## Integrity and validation

All32 reference records passed reconstruction/parity and isolation. Max global reverse/JVP relativeL2 2.2979141513098394e-06; max partition error 9.992007221626409e-16. Max reconstructed float64 flip-roundtrip error 3.0517578125e-05 pixels. Empty supports:0; zero/near-zero candidate gradients:0 for both objectives.

Source state hash before/after both runs:73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. Frozen/eval/parameter-grad-None passed; ISP state unchanged. All23 protected module LF hashes unchanged locally and remotely; all4 new source/test pins match remote release.

Baseline13tests passed3.04s; candidate increment18passed3.77s; final focused21passed3.76s. Full regression204passed,10skipped,4warnings in10.04s. Existing NVML initialization and protobuf deprecation warnings retained; CUDA computation succeeded. No driver changes.

Candidate run `20260914-041922-taisp-t024a-candidates32`: 9.267749341961462s. Reference run `20260914-042406-taisp-t024a-reference32`: 17.862508163030725s. Both exit0, GPU NVIDIA RTX A6000, CUDA12.1. CPU used only cohort/file/report aggregation.

## Durable artifacts and next action

Per-episode utilities/cosines/norms/mask/support/per-object statistics: `T024A/per_episode.tsv`. Exact summaries by condition and block: `T024A/summary.json`. All32 individual candidate and32 reference records, original/flipped boxes, supports, mask rectangles/hashes, loss vectors, JVP columns, 8D gradients and environments remain in the two remote_runs directories. Candidate SHA pins, commit ordering, code pins, complete cohorts/exclusions and integrity receipts are retained.

Stop NEEDS_REVIEW. Close fixed ROI flip-equivariance family under R039. No temperature/layer/support threshold/loss-blend/K/LR/mask/corruption rescue; no T024-B or AP. Await an explicit new research decision.
