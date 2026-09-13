# T022-A1 BLOCKED — parity repaired; runtime edge repeatability failed

2026-09-14 +08. R035/0b46353; parity repair8948d4c; passing parity receipts
c713150; original R034 driver integration7001d0c. **No scientific PASS/FAIL.**

The requested `.eval().requires_grad_(False)` initialization was added only to
the standalone parity harness. Separate parameter-frozen/grad-none,
all-modules-eval and unchanged-state-hash checks were added and tested. The
original failed T022-A receipt remains immutable. All 28 frozen parity records
now pass. The subsequent real K=3 runtime smoke stopped on repeated edge
episodes, before AP. No second repair, tolerance change or formal run occurred.

## A–B: PASS

Run `20260914-001023-taisp-t022a1-numerical-parity`, release
`20260914-001000-taisp-t022a1-parity`, 00:10:28–00:11:23 +08, exit0.
Focused **21passed/1skipped/3.57s**; full **183passed/10skipped/4warnings/8.25s**.
Same IDs160585/114830, seven conditions, equal states0/.01, CUDA model/gradient
calls, original tolerance/dtype/config/cohort. All numerical checks, all three
separate CLIP subchecks and final source/CLIP hashes pass for all28records.

Maximum relativeL2: pseudo4.772744148301399e-6, CLIP4.2817806631990866e-7,
mean-update4.3659021589596015e-6. Exact extrema and original records are retained
in T022A1/parity_summary.json and the raw run.

## C: BLOCKED

The original R034 shared driver was extended only to route spatial_dose_ours to
the unchanged runtime, with spatial receipt/summary and frozen gate mapping.
Current-Ours routing and protected method modules remain unchanged.
Driver-focused increment:18passed/2warnings/4.84s.

Run `20260914-001557-taisp-t022a1-runtime-smoke`, release
`20260914-001534-taisp-t022a1-runtime`, 00:16:02–00:16:31 +08, exit1.
Focused **18passed/2warnings/4.86s**; full **186passed/10skipped/4warnings/9.68s**.
CUDA was used for all real forward/gradient/adaptation calls. The known NVML
warning appeared but CUDA execution worked; no dependency or driver changes.

At image160585/gamma_s1, the additional edge test ran two K=3 episodes each for
empty mask, full mask and empty support. Empty/full masks used the same nonempty
current supports; empty support used zero boxes and an empty mask. Each call
starts both states at zero. The exact-repeat check compares final state and
processed image with torch.equal, as committed before the run.

| Edge case | Max absolute final-state difference | Relative L2 final-state difference | Exact repeat |
| --- | --- | --- | --- |
| Empty mask / nonempty support | 0.00014580879360437393 | 0.004515576814211109 | FAIL |
| Full mask / nonempty support | 2.7939677238464355e-9 | 7.933545179559295e-8 | FAIL |
| Empty support | 0 | 0 | PASS |

The tiny full-mask difference is distinguished from the larger empty-mask
state divergence. Neither is AP evidence. Empty-mask pseudo-gradient relative
differences at steps0–3:8.6911993e-8,3.7718332e-4,1.5405530e-2,1.8640228e-2.
This describes saved trajectories and does not establish a root cause. No
extra model calls, deterministic-kernel investigation or rescue runs occurred.

All other saved edge checks pass: finite states/images, exactzero reset,
three updates, fixed support counts, dose bounds/meanone, zero-common no-update,
inactive-region pseudo/CLIP gradients zero, reconstructed final-image equality,
frozen/eval/grad-none models, unchanged source/CLIP hashes and owned ISP state.
Empty mask:object gradient zero; full mask:background gradient zero. Inactive
states may receive a shared dose but cannot affect the blended image. Empty
support gives exactly zero states/no update. Full per-step gradients, common
directions, c, coefficients, regional states and image diagnostics are saved.

The run stopped inside edge_smoke before the first normal spatial candidate row
was appended. samples.jsonl contains **one current-Ours row only**. The normal
candidate call reached the edge call, but its diagnostics were not persisted.
Three paired edge receipts retain all **six complete K=3 trajectories**. The
normal two-image/seven-condition smoke is **incomplete**. There is no completed
prediction collection, official evaluation, formal200run or AP result.

## Stop, provenance and next review

R035: “A material runtime/isolation/reset failure is again a blocker: preserve
it and stop without AP.” Stage C **BLOCKED**. Stage D and the five AP criteria
are **NOT EVALUATED**; the no-blocker criterion is **FALSE**. Spatial-dose efficacy
remains scientifically open. No active job remains.

Research review can decide whether to authorize a separately bounded analysis
of the saved empty-mask divergence. No second repair is validated or applied.
No change to tolerance, rho, mask, support threshold, K/LR, CLIP scaling,
independent directions, objective, training or cohort was made after failure.

Cohort SHA256589878ad7af1d86bf73514d63dd58a4e40d58797d119ea4981be9e4292772690:
200fresh source images/four50blocks;1236prior source/debug plus5000valIDs excluded.
Actual remote hashes pass **18 protected byte-exact modules and 5 authorized
modules using their declared LF-normalized pins**. Raw and normalized hashes
are both saved: three edited text files have CRLF deployment line endings.
Spatial runtime/ISP remain byte-identical to pre-R035 pins. Raw failed receipts
are unchanged; LF comparison applies only to the declared code pins.

Both raw archives were fetched and SHA256 verified; 44 files / 411,594 bytes retained.
See T022A1/artifact_manifest.json, remote_code_hashes.json, parity_summary.json,
repeat_differences.json, and both named remote_runs folders for exact evidence.
