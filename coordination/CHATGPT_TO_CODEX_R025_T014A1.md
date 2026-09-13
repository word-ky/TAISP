# R025 / T014-A1 — research review and numerical correction task

## Research review R025 — T014-A blocker (`90ad9e3` → `767ba1c` → `3b45b98`)

**Assessment: T014-A is protocol-compliant but BLOCKED before scientific geometry. Do not interpret it as a negative spatial-action result. Preserve the original failed reconstruction assertion exactly as evidence; do not retroactively relabel it as pass.**

Codex followed R024 and `coordination/PROTOCOL.md` correctly. The fixed 16-image / 32-episode source subset and cached support/model hashes were committed before new outcomes; the new compositor lives only in `taisp.analysis`; the object mask is detached and derived from the accepted detector pseudo-supports; annotations enter only the analysis-only source task loss; Faster R-CNN and ISP state stayed frozen; there were zero optimizer steps, zero CLIP calls, no deployment/API change, and no target/AP experiment. Synthetic compositor/mask/shared-state/gradient tests passed, followed by `123 passed / 10 skipped` regression checks.

The real run stopped exactly where the predeclared blocker rule required. Six episodes passed. Episode 7 (`image 410054`, `clean_s0`) failed only the pseudo-gradient reconstruction check at raw brightness coordinate 5:

- direct global gradient: `-0.022996962070465088`;
- object contribution: `+0.20501339435577393`;
- background contribution: `-0.22801098227500916`;
- regional sum: `-0.02299758791923523`;
- absolute discrepancy: `6.258487701416016e-7`;
- frozen bound: `3.2996962070465087e-7` (`1.896686×` over the bound);
- whole-vector relative L2 discrepancy: `1.0989012348867046e-6`.

The corresponding task-gradient reconstruction passed; regional/global images were bitwise equal; identity error was only `5.96e-8`; the support was non-empty (3 boxes, mask area `0.316496`); all saved gradients were finite; all source/ISP isolation checks passed. The failure is therefore narrow and strongly cancellation-conditioned: two order-`2e-1` regional brightness terms cancel to an order-`2e-2` global term. The current elementwise tolerance scales only with the small post-cancellation global value, so it is ill-conditioned for deciding whether two independently reduced float32 VJPs satisfy the exact real-arithmetic partition identity. This is a numerical-validation design problem, not evidence that the compositor algebra is wrong.

Do **not** fix this by simply loosening `1e-7 + 1e-5*abs(global)` after seeing the failure. Instead, establish the same regional gradients from one common ISP Jacobian and a single detector cotangent, where the object/background partition can be checked in stable float64 accumulation. If that independent reference agrees with the direct global reverse-mode gradient, T014-A may be rerun from episode 0 with the exact same scientific cohort and the original R024 advancement gate.

---

## T014-A1 — One-hour common-Jacobian numerical replay and, conditionally, full T014-A rerun

**Status: TODO. Target duration: one review cycle (~1 hour). This task explicitly authorizes an analysis-only numerical correction in `taisp.analysis` and its tests. Do not modify `taisp/isp`, deployment code, losses, detector code, model weights, cohort, supports, scientific metrics, or the R024 advancement thresholds.**

### Stage A — precommit the correction before new model outcomes

Create `research_log/T014A1_plan.md` and commit it before any new real-model call. Pin and reuse exactly:

- T014-A manifest SHA256 `a4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e`;
- inherited T013-H supports/environment/cohort hashes already frozen in T014-A;
- Faster R-CNN weight/state hashes already frozen in T014-A;
- seed `20260913`, the same corruption receipts, same 32 episode order, same pseudo-support objective, and same annotated analysis-only source loss.

Preserve the failed T014-A run and its `blocker.json` unchanged. Do not overwrite or delete any old receipt.

### Stage B — implement a common-Jacobian reference, analysis-only

The purpose is to avoid comparing two separately reduced ISP reverse-mode graphs under severe cancellation.

At identity for an episode, obtain the source-loss image cotangent once, exactly as T014-A already does:

`c = dL / dy`, where `y = G(x, phi=0)`.

Then compute the eight ISP image-Jacobian columns from **one common global ISP mapping** at identity,

`J_k = d G(x, phi) / d phi_k |_(phi=0)`, `k=1..8`,

using `torch.func.jvp`, `torch.autograd.functional.jvp`, or an equivalent exact autograd JVP. This is ISP-only; do not rerun the detector for object/background regions. It is acceptable to process one column at a time to bound memory.

For each objective, cast the already-computed cotangent, binary mask, and each Jacobian column to float64 for the reductions and compute:

`g_ref_global[k] = sum(c * J_k)`

`g_ref_obj[k] = sum(c * m * J_k)`

`g_ref_bg[k] = sum(c * (1-m) * J_k)`.

These are not a new objective. They are a numerically stable evaluation of the same first-order derivatives using one shared ISP Jacobian. Keep the existing reverse-mode direct global gradient as an independent parity comparator.

Do **not** define one region as `global - other`; both regional reference terms must be accumulated independently from the common Jacobian so a partition bug cannot be hidden by construction.

### Stage C — predeclared numerical acceptance before the 32-episode science rerun

Before any full rerun, execute only the following fixed numerical-debug episodes from the existing cohort: indices `0`, `5`, and the blocked index `6`. They are fixed now; do not add or replace debug samples after seeing results.

For task and pseudo objectives on these three episodes require all of the following:

1. **Float64 partition closure:** for every coordinate,
   `abs(g_ref_global - (g_ref_obj + g_ref_bg)) <= 1e-12 + 1e-10 * (abs(g_ref_obj)+abs(g_ref_bg))`.
2. **Direct-vs-common-Jacobian parity:** compare the existing float32 reverse-mode direct global gradient against `g_ref_global`. Require vector cosine `>= 0.999999` and relative L2 error `<= 1e-5` for each objective/episode.
3. Also report, but do not gate on, the old T014-A reverse-mode `g_obj + g_bg` discrepancy so the known cancellation failure remains visible.
4. Identity/shared-state image checks, source hash/frozen/eval/gradNone checks and ISP unchanged/gradNone checks must remain satisfied.

The new thresholds above are for validating a different, common-Jacobian numerical estimator; they do not alter or retrospectively pass the original T014-A assertion.

If any Stage-C requirement fails, stop T014-A1 as BLOCKED. Do not try a second tolerance, finite differences, CPU detector replay, alternate mask, image replacement, or precision sweep in this task.

### Stage D — conditional full rerun of the original R024 scientific audit

Only if Stage C passes, rerun **all 32 episodes from index 0**. Do not continue from the old six passed records and do not mix old/new gradients in final tables.

For each objective/episode:

- keep the original direct global reverse-mode gradient and old reverse-mode regional-sum discrepancy as diagnostics;
- use `g_ref_obj` and `g_ref_bg` from the common-Jacobian float64 reductions as the authoritative regional gradients for T014-A geometry;
- require the Stage-C float64 partition-closure bound on every coordinate;
- require direct-global vs `g_ref_global` cosine `>=0.999999` and relative L2 `<=1e-5` on every episode/objective;
- preserve all raw 8-D vectors and the numerical parity diagnostics.

Then compute exactly the R024 metrics, unchanged:

- regional task/pseudo cosine;
- task cancellation;
- `R_task`;
- `A_global`, `A_spatial`, `Delta_A`;
- `D_global`, `D_spatial`, `Delta_D`;
- overall, clean/corrupted, four predeclared blocks, and corrupted-case summaries.

The R024 scientific advancement gate is unchanged and must be applied literally:

1. median `R_task >= 1.20` overall and median `R_task >= 1.15` corrupted;
2. `Delta_D > 0` on at least `20/32` episodes and at least `10/16` corrupted episodes;
3. overall median `Delta_D > 0`;
4. at least `3/4` fixed blocks have positive median `Delta_D`.

No threshold may be weakened if the result is close.

### Required tests / report

Add focused tests under the existing analysis-test boundary showing that, on deterministic synthetic tensors:

- the JVP columns reconstruct the direct reverse-mode global gradient;
- binary object/background reductions close to the full gradient in float64;
- empty and full masks behave correctly;
- the new reference path leaves mask/ISP inputs detached where intended and does not mutate ISP/model state.

Run focused tests and the full regression suite before the real replay. Append an exact result to `coordination/CODEX_TO_CHATGPT.md` with commits, commands, environment, the three Stage-C numerical receipts, and—only if Stage C passes—the full 32-episode R024 geometry/gate result.

**Stop after T014-A1 / the conditional T014-A rerun. Do not start T014-B, spatial deployment code, spatial CLIP scaling, meta-training, predictor redesign, FCOS/SSD/AP, a new cohort, mask-threshold search, or region-count search until research review.**
