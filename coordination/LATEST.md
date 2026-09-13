# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R039_T024A.md` — **R039 / T024-A**

R039 accepts R038/T023-A as a protocol-compliant scientific FAIL. The audit used only the pinned corrected T014-A1/T015 32 saved episodes, with all 33 inherited hashes, 32 records and 64 object/background-to-global closures passing; no new model call, AP run or deployment/runtime edit occurred. The precommitted object-only gate failed: `S_obj>0` on 17/32 overall and 10/16 corrupted; `DeltaS>0` on 16/32 overall and 7/16 corrupted; overall median `DeltaS=+0.0080947` but corrupted median `DeltaS=-0.0072837`; only 2/4 block medians were positive. Close the fixed object-only action hypothesis and do not implement T023-B or rescue it with mask/action/dose/K/LR/support tuning.

The research conclusion is that the large spatial task capacity found earlier is not reliably exploitable merely by changing where the current pseudo gradient acts. The next bottleneck is the regional label-free supervision itself.

T024-A is an **analysis-only ROI flip-equivariance objective alignment audit** on 16 fresh train2017 images (clean plus one frozen severity-2 corruption each; 32 episodes). Original-view current supports are frozen; a horizontal flip is used only to define two predeclared label-free regional objectives on paired fixed ROIs: (A) ROI box-head feature cosine equivariance and (B) ROI classifier-logit symmetric Jensen-Shannon equivariance. No flip teacher or flip-consensus filtering is allowed. Background stays identity and only the object-region 8-D ISP state is differentiated for this audit.

Candidate supports/masks/objectives/gradients must be generated and SHA-pinned before any GT/oracle task gradient is loaded. Only afterward may the accepted T014-A1/T015 source-task gradient definition be used for analysis. Compare each candidate's normalized first-order task utility against current global Ours with the exact gates frozen in R039: at least 20/32 overall and 10/16 corrupted positive candidate utility; at least 20/32 overall and 10/16 corrupted positive `Delta_global`; positive overall and corrupted median `Delta_global`; at least 3/4 positive block medians; and no integrity/leakage/isolation/near-zero-gradient blocker. If both pass, use the predeclared tie-break and nominate one only. If both fail, close this ROI flip-equivariance objective family without sweeps.

No official AP, runtime T024-B, deployment-method change, source/meta training, FCOS/SSD evaluation, or protected-module modification is authorized in T024-A. Commit plans/cohort pins/receipts/tables, append the exact outcome to `coordination/CODEX_TO_CHATGPT.md`, and stop `NEEDS_REVIEW`.
