# T002 live handoff

2026-09-12, Asia/Shanghai. Status IN_PROGRESS.
User authorized direct execution of all assigned tasks. Heartbeat id taisp,
every 15 minutes, now continues authorized work rather than only reporting TODO.

## Current action

Full experiment running on A6000 GPU 0:

- source `c639969`, release `20260912-011912-taisp-t002-full-fixed`;
- run `20260912-011915-taisp-t002-coco200-fixed`;
- remote project `/home/liujianhua/wjq/TAISP`;
- outputs `runs/20260912-011915-taisp-t002-coco200-fixed/artifacts/study/`;
- command: `python -m taisp.analysis.run_t002 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --output "$AUTODL_ARTIFACTS_DIR/study"`;
- venv `.venv/bin/python`, existing AutoDL scripts in
  `D:\work\claude-autodl\autodl-workflow-clean`, configure env
  `AUTODL_CONFIG_PATH=D:\work\fightccfa-agin\CVPR2027\TTT-ISP\.autodl\config.json`.

Read this run's logs before doing anything else on resume. Do not start a second
full experiment. Wait for completion/exit status, fetch the whole run to local
research_log/remote_runs, inspect summary/metrics and raw sample records, write
all-family results/interpretation and report in CODEX_TO_CHATGPT.md, commit/push.
Keep all negative results. Do not tune learning rates/prompts or progress to
meta-training without the research lead's next assigned task.

## Completed evidence

- T001 baseline local regression: 17 passed, 1 skipped in 26.71s.
- Real CLIP integration: 3 passed in 3.09s, run 20260912-010431-taisp-t002-clip-check.
- Detector/oracle: 2 passed in 4.12s, run 20260912-010621-taisp-t002-detector-tests.
- Full A6000 suite: 25 passed in 6.17s.
- Real 2-image study smoke: 12 records, 25 comparison evaluations, 5.60s experiment,
  run 20260912-011042-taisp-t002-study-smoke, exit 0; exact receipts already local
  and committed. No hyperparameter change was made based on the smoke results.
- Data preparation: 200 random images, seed 20260912. Full cached source annotations
  from `/home/wjq/WeDetect/AgenticVocabDet-ICLR2027/shared/data/coco-val200/source/instances_val2017.json`
  copied to this project's shared/coco200 after an official archive download was
  too slow. The original download run 20260912-010134-taisp-t002-data was stopped.
  Successful data run: 20260912-010501-taisp-t002-data-cached. Exact JPEG hashes,
  annotation SHA256, and IDs in shared/coco200/subset.json and copied run metadata.

## Known qualifications

T002_plan.md and docs/T002.md specify the fixed scientific protocol. Generic
prompt direction does not observe corruption metadata; only offline oracle sees
annotations. Native proposal matching/sampling means the oracle loss is piecewise
smooth and not an AP metric. Backbone buffers/weights remain frozen.
CUDA antialiased bicubic backward is not bitwise deterministic; a diagnostic
explicitly identified the PyTorch op. Exact initial episode state plus numerical
update reproducibility is tested; forward preprocessing is deterministic.
No benchmark claims, learned prompts, predictor training or meta-learning.


Update 01:19: original full run failed on image 105335 (612x612), CLIP short-side rounding produced 223. Fixed shortest dimension with integer arithmetic; regression plus full real suite 26 passed in 5.89s. Current run is the full restart above, same 200 IDs and scientific settings. First failed run is retained under local research_log/remote_runs/20260912-011328-taisp-t002-coco200/.
