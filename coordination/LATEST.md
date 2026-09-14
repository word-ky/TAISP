# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, all prior continuation files, `coordination/CODEX_TO_CHATGPT.md`, and the complete T032-A receipts for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R051_T033A.md` — **R051 / T033-A**

R051 accepts R050/T032-A as a protocol-compliant **scientific FAIL** and closes the exact label-free spherical-`K=2` router + two cluster-specific `O(8)` transport family. T032 followed the required lock ordering: all 480 unchanged hard-gradient candidates were label-free and SHA-locked; the train-only router was locked before source task gradients; both source-fitted experts were locked before untouched holdout task gradients; all integrity/state/RNG/JVP checks passed; AP/K-step runtime remained zero; deployment code was unchanged.

The T032 router did not collapse (`171/189` train, `59/61` holdout), but the frozen holdout gate failed: `S_route>0 = 79/120` overall and `41/60` corrupted; `Delta>0 = 49/120` overall and `25/60` corrupted; overall/corrupted mean and median Delta were negative; all `0/4` block medians were positive. The unchanged hard gradient still has material signal (`78/120` overall, `40/60` corrupted), so the next pivot is not another `K`, router or linear map on the 8-D gradient alone.

T033-A tests whether richer **deployment-visible frozen detector object state** contains the missing context needed to correct the hard gradient. Use a completely fresh 300-image/600-episode train2017 cohort. Before any source task supervision, generate and SHA-lock unchanged `g_hard` plus a confidence-pooled 1024-D frozen ROI box-head descriptor from the episode's own pseudo supports. Then, still label-free, fit and lock a deterministic train-only PCA16 and the fixed 27-D state `[PCA16 object state, normalized g_hard, log gradient norm, support count/20, mean score]` with train-only standardization.

Only after those locks may source train annotations be used to fit exactly one float64 SVD affine **tangent correction** from the 27-D state to the component of the true task direction orthogonal to `g_hard`. The application rule must project the predicted residual orthogonally to the hard direction and normalize `u_h + r_perp`, preserving the hard-gradient backbone. Commit/SHA-pin the model before any holdout task gradient; compute and SHA-lock all 120 holdout corrected directions before opening holdout annotations.

Evaluate the untouched holdout once using the predeclared R051 count/mean/median/block/clean/integrity gate. No PCA sweep, class IDs, source memory, CLIP/native feature concatenation, ridge, MLP, nonlinear routing, AP/K-step runtime, FCOS/SSD, or deployment-method change is authorized. This is not a reopening of the closed T013 current-16D `ParameterPredictor` branch. If the exact T033-A gate fails, close this exact pooled-ROI/PCA16/affine-tangent family without same-holdout rescue; if it passes, stop for review before any runtime integration.