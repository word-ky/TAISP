# R026 / T015-A — research review and next task

## Research review R026 — T014-A1 acceptance (`89cac87` → `0ad53b6` → `ad8e579`)

**Assessment: T014-A1 is ACCEPTED as a protocol-compliant numerical correction and completed scientific audit. The common-Jacobian reference resolves the validation blocker, but the unchanged R024 spatial-utility conjunction FAILS. T014-A/T014-A1 are CLOSED. Do not implement T014-B from this fixed partition/pseudo objective.**

Codex followed R025 and `coordination/PROTOCOL.md` correctly. The outcome-free plan preceded the implementation and real replay; the original failed T014-A receipt/blocker remained unchanged; all new logic was confined to `taisp.analysis`, tests, reporting, and research receipts. The frozen 16-image/32-episode source cohort, cached pseudo supports, source model/weight hashes, seed, mask construction, source oracle loss, and scientific thresholds were reused exactly. Faster R-CNN and ISP state remained frozen; annotations entered only the analysis-only source task loss; there were zero optimizer steps, zero CLIP calls, no test/target labels in deployment, no AP experiment, and no deployment/ISP/loss/model change.

The numerical correction is convincing. On fixed debug indices 0/5/6, all six task/pseudo common-Jacobian checks passed, with minimum direct-vs-reference cosine `0.9999999999998749`, maximum relative L2 error `5.0649e-7`, and maximum float64 partition error `1.11e-15`. The fresh 32-episode rerun also passed all 64 objective checks: minimum parity cosine `0.9999999999979398`, maximum relative L2 `2.0421e-6`, and maximum partition error `1.3323e-15`. The old reverse-mode regional failures at pseudo indices 6 and 13 remain visible as diagnostics and were not retroactively relabeled.

The scientific result must be read literally. The fixed two-region action space shows strong task-gradient heterogeneity (`median R_task=1.8618` overall and corrupted; task regional cosine median `-0.4893`; cancellation median `0.4468`), but the current detector-native pseudo direction does **not** satisfy the predeclared ability-to-use-that-capacity gate. `Delta_D>0` occurs on only `19/32` episodes and `9/16` corrupted episodes, below the frozen `20/32` and `10/16` thresholds. The overall median `Delta_D` is positive (`+0.01293`) and 3/4 block medians are positive, but the mean is negative (`-0.03027`) with large harmful tails; therefore the two missed count conditions are not a harmless rounding issue. No threshold should be relaxed after seeing the result.

This does **not** prove that spatial ISP is useless. It says something narrower: with this fixed object/background partition and the current detector-native pseudo objective, the extra spatial degrees of freedom are not exploited reliably enough to justify a finite-step spatial TTA prototype. Before abandoning the spatial idea entirely, one cheap frozen-array postmortem can distinguish whether the failed utility gate comes from (a) little genuinely differential task signal after projection onto the global/shared subspace, or (b) substantial task-relevant differential signal that the pseudo objective points through inconsistently. That distinction can be answered exactly from the already saved T014-A1 gradients with no new model call.

---

## T015-A — One-hour frozen differential-subspace postmortem

**Status: TODO. Target duration: one review cycle (~1 hour). This is algebraic analysis only. Do not call Faster R-CNN, CLIP, or the ISP; do not run an optimizer; do not change masks, thresholds, cohort, objectives, or implementation/deployment code. Reuse only the authoritative fresh 32-episode T014-A1 common-Jacobian records.**

### Scientific question

For each episode, the authoritative regional gradient is a 16-D vector

`g = [g_obj, g_bg]`.

Decompose it orthogonally into the global/shared subspace and the genuinely spatial differential subspace:

- `s = (g_obj + g_bg) / 2`
- `d = (g_obj - g_bg) / 2`
- `g_shared = [s, s]`
- `g_diff = [d, -d]`

so `g = g_shared + g_diff` and `<g_shared, g_diff> = 0`.

Do this independently for the annotated task gradient `t` and label-free pseudo gradient `p`. The purpose is to determine whether the failed R024 `Delta_D` gate is caused by insufficient task-relevant spatial capacity or by a guidance-direction failure in the differential subspace.

### Stage A — freeze provenance before analysis

Create `research_log/T015A_plan.md` and commit it before computing summary outcomes. Pin the T014-A1 run and SHA256 of its authoritative `records.json` plus the 32 fresh record files. Verify:

- exactly 32 episodes in the original order;
- 16 clean / 16 corrupted;
- four original fixed blocks of eight episodes;
- every task/pseudo vector is the T014-A1 **common-Jacobian reference** object/background vector, not the old reverse-mode regional diagnostic;
- all T014-A1 numerical/isolation flags are already true.

No new sample filtering, replacement, region definition, corruption regrouping, or result-dependent threshold is allowed.

### Stage B — exact orthogonal decomposition

For task and pseudo separately, save for every episode:

1. `||g_shared||`, `||g_diff||`, and `||g||`;
2. differential energy fraction
   `f_diff = ||g_diff||^2 / (||g||^2 + eps)`;
3. exact energy reconstruction error
   `abs(||g||^2 - ||g_shared||^2 - ||g_diff||^2)`;
4. exact vector reconstruction error `||g - g_shared - g_diff||`;
5. shared and differential 8-D vectors `s` and `d`.

For the task gradient also compute an action-space capacity ratio with equal 16-D raw-state norm budget:

`R_extra = ||t|| / (||t_shared|| + eps)`.

`R_extra=1` means the extra differential degrees of freedom offer no first-order oracle capacity beyond the best shared/global action; `R_extra=1.10` means a 10% larger best first-order task decrease is available in the full two-region space at the same raw-state norm.

### Stage C — isolate whether the pseudo differential component helps or hurts

Because shared and differential subspaces are orthogonal, decompose the existing spatial productivity exactly. Define, for `P = p_shared + p_diff` and `T = t_shared + t_diff`,

- `C_shared = <t_shared, p_shared> / (||P|| + eps)`
- `C_diff   = <t_diff,   p_diff>   / (||P|| + eps)`
- `D_reconstructed = C_shared + C_diff`.

Verify `D_reconstructed` equals the saved T014-A1 `D_spatial` within `1e-12` absolute error on every episode. Also report:

- `cos_shared = cos(t_shared, p_shared)`;
- `cos_diff = cos(t_diff, p_diff)` when both differential norms are nonzero, with zero-norm cases explicitly counted rather than silently assigned a favorable value;
- sign of `<t_diff, p_diff>`;
- pseudo differential norm share `||p_diff||^2 / (||p||^2 + eps)`;
- task differential norm share `f_diff_task`;
- `C_diff` and the fraction of full pseudo norm allocated to the differential subspace.

Render overall, clean, corrupted, the four frozen blocks, and the four corruption families already present in T014-A1. Mask area/support size may be correlated with `R_extra`, `f_diff_task`, and `C_diff` descriptively, but do not introduce a mask threshold or subgroup selection.

### Stage D — predeclared interpretation

Use these rules only as a **postmortem triage**, not as permission to implement a spatial TTA method automatically:

1. Call the extra spatial action space **task-relevant on this frozen source cohort** only if:
   - median `R_extra >= 1.10` overall;
   - median `R_extra >= 1.10` on corrupted episodes; and
   - at least 3/4 fixed blocks have median `R_extra > 1.05`.

2. Call the current pseudo objective **differentially useful** only if all of the following hold:
   - `C_diff > 0` on at least `20/32` episodes;
   - `C_diff > 0` on at least `10/16` corrupted episodes;
   - overall median `C_diff > 0`; and
   - at least 3/4 fixed blocks have positive median `C_diff`.

These count/block thresholds intentionally mirror the frozen R024 utility gate rather than inventing a more permissive rule after the near miss.

Interpretation:

- **Task-relevant YES, differential pseudo useful NO:** conclude that the current object/background action space contains real source-task capacity, but the accepted detector-native pseudo objective cannot reliably exploit the additional spatial degree of freedom. Close this fixed pseudo-spatial branch. A future research task may design a new source-trained regional guidance signal, but do not start it automatically.
- **Task-relevant NO:** conclude that the current object/background partition is not sufficiently compelling beyond the shared/global subspace. Close this partition; do not search box thresholds, region counts, or alternate masks on the same cohort.
- **Both YES:** the R024 failure is more specifically due to interaction/normalization between shared and differential pseudo components rather than absence of a useful differential signal. Report this, but still stop for review; do not implement differential-only projection or a spatial TTA prototype automatically.

### Required tests and report

Add only analysis/tests/reporting code as needed. Synthetic tests must verify exact shared+diff reconstruction, orthogonality, energy Pythagorean identity, and exact `D_spatial = C_shared + C_diff` reconstruction on known vectors including zero-differential and one-region-only cases. Run the focused tests and full regression suite. Append the exact outcome, hashes, commands, and all frozen triage flags to `coordination/CODEX_TO_CHATGPT.md`.

**Stop after T015-A. Do not start T014-B/T015-B, finite-step spatial adaptation, differential-only deployment, mask/region search, CLIP regional scaling, source/meta-training, predictor redesign, FCOS/SSD/AP, a new cohort, gate/dose tuning, or any implementation change outside the analysis boundary until research review.**
