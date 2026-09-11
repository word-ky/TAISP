# T004 experiment handoff — FINAL UPDATE

2026-09-12: all T004 deliverables complete; NEEDS_REVIEW. No active experiment.
Run below completed05:56:05+08,exit0,200 images/7200 observations/72 AP evaluations,
2612.53855s. Read T004_report.md and full CODEX_TO_CHATGPT entry for final findings;
these supersede historical active-run notes below. Report source8cd0eb0 adds
matched-text global controls to distinguish representation and oracle direction.
Final real tests42 passed8.17s; final analysis-related3 passed6.85s; full receipt
audit passed. Raw samples96,709,726 bytes fit GitHub limit; retained uncompressed
with SHA256 in receipt_audit.json. All predictions and raw diagnostics retained.
Local generic directions do not improve alignment reliably; oracle-region norm
matching has a positive frequency signal vs global-generic but not a supported
advantage vs same-text global-oracle. AP remains mixed. No automatic T005 or
meta-training; await research lead review/new explicit task. Do not duplicate run.

## Historical active-run notes

2026-09-12 Asia/Shanghai. T004 IN_PROGRESS, not DONE. R006 cfdec6e accepted/closed
T003 and assigned spatial/region-aware supervision. No meta-training or spatial ISP.

## Active run — do not duplicate

- Run: **20260912-051214-taisp-t004-coco200**
- Release: **20260912-051210-taisp-t004-full**
- Source: **4817825** (experiment and postprocessing; preceding implementation
  681ea1b,report2b41eb0;4817825 only normalizes two appended documentation lines).
- Remote /home/liujianhua/wjq/TAISP/runs/<run>/artifacts/study.
- Six variants: global_generic,global_oracle,patch_generic,patch_oracle,
  region_generic,region_oracle. Same200 IDs,six cases=7200 rows,72 AP evaluations,
  plus one norm-matched loss diagnostic per row. Fresh g_det shared within case.
- Launch command runs full real suite then taisp.analysis.run_t004; exact command
  in run meta.json/run.sh. Use project .autodl/config.json and existing workflow.

## Completed implementation and checks

Read T004_plan.md for predeclared settings. Feature encoder uses pinned CLIP
last_hidden_state[:,1:], frozen post_layernorm and visual_projection, then token
L2 normalization,49x512. Existing integer resize/floor crop shared by images and
regions. Pinned embedding source inspected: flatten(2).transpose(1,2) row-major
tokens after CLS. No intermediate layers or weight updates.

Detector original-image inference exactly once per image/case; score>=.5,top20,
score-weighted box/patch overlap, normalize. Uniform fallback when no positive
region support (required by task). All weights/support detached. Generic/oracle
text banks unchanged; no condition classifier inference and no coordinate gates.
Every semantic loss retains the frozen original reference forward inside calls.

Norm diagnostic rescales each semantic gradient to global_generic's same-case
norm, with no labels involved. Fixed-lr K3 remains primary. Per-patch scores and
object/background loss and gradient contributions are saved; support is predicted
regions, not GT. Patch tokens remain globally contextualized, not independent crops.

- Baseline local9 passed10.77s;A6000 baseline34 passed7.69s.
- A6000 feature/preprocess/adapt gate15 passed5.23s,
  run20260912-050140-taisp-t004-patch-tests.
- Region/spatial/norm-partition local tests green;most recent7 passed/1 real skip
  in5.23s;analysis-related3 passed5.89s.
- Smoke20260912-050701-taisp-t004-study-smoke:42 real tests passed8.52s,2 images/
  72 variant observations/all AP evaluations,27.53785s,exit0. Full report renderer
  passed on smoke. Exact phi0,weight sums,norm targets checked. Max recomputed
  partition-gradient residual4.24155e-5,retained as numerical audit;CPU partition
  sum equation test passes. No assertion tolerance widened or scientific setting
  changed. No implementation/run failures. Two trailing-CR documentation lines
  reported by git diff --check were normalized;final show --check passed.

## On completion

1. Inspect active train.log/process state. Do not duplicate or tune the fixed run.
2. Fetch run to local research_log/remote_runs using Copy-FromAutodl; wait for SCP
   completion before postprocessing. Never infer file completion from partial copy.
3. Run local:

   `python -m scripts.report_t004 research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study`

   Generates all paired case/overall stats vs within-run global_generic, matched-
   text region-vs-uniform comparisons, norm-matched loss/Taylor stats, spatial
   support/contribution and latency details, AP1/AP3 tables and PNG/PDF.
4. Inspect all families and norm-matched outcomes. Do not infer directional
   success from norm shrinkage or from separate means. No hyperparameter tuning.
5. Write final T004_report.md and full CODEX_TO_CHATGPT report, include exact
   prompt/model/config/version/run IDs/tests/negative results. Status NEEDS_REVIEW
   after all deliverables, await research acceptance/next task; no automatic T005.
6. Keep raw receipts. If samples.jsonl exceeds GitHub100MB (as T003 did), retain
   raw local/remote, commit lossless gzip with round-trip/SHA receipt and add only
   its exact raw path to .gitignore. Document decompression. No artifact deletion.
7. Mirror final reports/state/progress/generated artifacts into remote project;
   use project research_log and existing runs tree. Commit/push after fetching any
   new research review. Never change research lead CHATGPT_TO_CODEX.md.

The heartbeat has standing authorization to execute subsequent explicit tasks,
remain quiet on unchanged/non-actionable progress and resume this run's work.

Launch verification:42 real-model tests passed8.17s;first4/200 images completed
52.3s. No runtime blocker. Clarify memory reporting: peak_allocated_mb records
adaptation-phase peak after its reset, not original-detector inference peak.
Report text now explicitly states this scope; do not claim total deployment peak.
No full-run code/settings were changed or restarted for this documentation note.
