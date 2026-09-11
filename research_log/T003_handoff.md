# T003 active experiment handoff

Updated 2026-09-12 03:12 Asia/Shanghai. Status IN_PROGRESS, not DONE.

- Research lead R004 edf75c0 accepted/closed T002 and assigned T003. User permits
  direct execution; no further start confirmation needed. No meta-training.
- Pre-run contract: T003_plan.md, including two observed-numerics amendments.
- Stage A finished: T003_stage_a/coordinates.{md,json}; hand arithmetic test passes.
- New deployable module: taisp/losses/conditioned_clip.py. Same T002 prompts grouped
  into three negative banks; temperature0.05; detached original-image weights.
  Optional detached coordinate_gate in adapt; frozen CLIP/detector, hard clamp.
- Analysis-only variants: generic, soft, oracle_prompt, soft_gate,
  soft_oracle_gate, oracle_both. Fresh g_det is computed once per image/case and
  shared among variants. T003 reruns generic; only clean/corrupted AP comes from
  matching T002 receipts. Do not mistake inherited det_loss_delta_oracle for a
  new T003 result: metadata explicitly marks it legacy summary compatibility.

## Active run — do not duplicate

Run: **20260912-031123-taisp-t003-coco200**
Release: **20260912-031119-taisp-t003-full**
Source: **8d8913e**
Remote root: /home/liujianhua/wjq/TAISP
Output: runs/20260912-031123-taisp-t003-coco200/artifacts/study/
Same fixed200 IDs, six cases, six variants =7200 rows,72 AP evaluations.
Command runs fresh full real-model tests then python -m taisp.analysis.run_t003.
Full command is stored in run meta.json/run.sh. Previous fresh two-image smoke
20260912-030712:33 tests passed7.28s,72 observations/all AP evaluations complete
in24.139s,exit0. Report pipeline succeeded on smoke; paired-key bootstrap test
passed locally1.38s. Final full launch adds the report test to the suite.

## Next steps

1. Use existing AutoDL workflow with project .autodl/config.json. Inspect the
   active run's train.log and tmux/process state. Do not launch another copy.
2. If completed, fetch the run into project research_log/remote_runs using
   Copy-FromAutodl from scripts/Autodl.Common.ps1. Wait for SCP completion before
   starting local postprocessing. Keep failures and recovery notes.
3. Run local `python -m scripts.report_t003 research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study`.
   This computes all variants/cases/overall Taylor stats, paired2000-image-cluster
   CIs, condition weights/top1 (analysis only), coordinates/saturation strata,
   loss/latency/trajectories and AP tables/PNG/PDF.
4. Inspect full metrics and negative families; compare oracle direction vs soft,
   soft vs soft_gate, soft vs soft_oracle_gate, oracle_prompt vs oracle_both.
   A better cosine or reduced step norm alone is not a successful task update.
5. Write research_log/T003_report.md and append full quantitative interpretation
   to CODEX_TO_CHATGPT.md. Include environment, prompts/temp/masks, command/run IDs,
   failures, within-run pairing correction, no-label/frozen tests, limitations.
   Status NEEDS_REVIEW once all deliverables complete; do not assign T004 yourself.
6. Mirror final reports/receipts to remote project-local research_log, commit/push
   meaningful outputs, update project_state.md/progress.md. Preserve research
   lead ownership of CHATGPT_TO_CODEX.md. Fetch any new review before final push.

## Observed failures retained

- 20260912-030241-taisp-t003-study-smoke:33 tests passed, strict old detector
  gradient comparison failed maxabs1.08e-4. Command source label mistyped35d074b;
  actual sourcec8ac8ff verified by local/remote file SHA, separate correction note.
- 20260912-030416-taisp-t003-gradient-diagnosis: four same-process backwards had
  exact same forward loss0.3455415964 but gradient differences2.36e-5 to7.35e-5.
  Native detector backward numerical variability reproduced, no speculative cause.
- 20260912-030559-taisp-t003-paired-smoke:33 tests passed, old three-step generic
  phi equality failed maxabs6.2841e-7. Known CUDA CLIP backward variability. Old
  cache equality was replaced by fresh generic and shared initial-gradient pairing,
  not a wider assertion tolerance. All cross-run differences saved as audit.
- A local smoke report was briefly started before SCP completed; rerun after
  transfer completed passed. No code change for this orchestration ordering error.

Both failures and passing smoke artifacts are committed. Do not treat small-smoke
metrics as scientific evidence. No prompts, temperature, lr, image IDs, corruption
strengths, models or ISP mapping were tuned after observing the smoke.

Launch verification: final full A6000 suite34 passed7.79s; first6/200 images completed57.2s. No runtime blocker at03:12.
