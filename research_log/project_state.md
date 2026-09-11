# Current project state

Updated 2026-09-12 Asia/Shanghai. R004 edf75c0 accepted and closed T002.
R00543fa056 approved T003 implementation/pairing. T003 all deliverables complete;
status NEEDS_REVIEW pending research lead acceptance. No active experiments.
Read T003_report.md for final interpretation and T003_handoff.md for recovery.
Run20260912-031123-taisp-t003-coco200 completed at03:45:51+08,exit0,2049.095s;
release20260912-031119-taisp-t003-full,source8d8913e; postprocessing724c04f.
200 images,6 cases,6 variants,7200 rows/72 evaluations. Final real suite34 passed;
R005 analysis tests2 passed. Do not rerun completed T003 based on stale queue TODO.
Soft weighting diffuse: overall top1 45.5%, entropy near98% max. Soft direction
and soft gate lack clear paired overall alignment gains. Oracle gates improve
effective alignment but AP is heterogeneous, including color-cast harm. No meta-
training; await next explicit task. Full paired stats and negative cases retained.
Raw120MB samples.jsonl remains local/remote; tracked lossless37.7MB gzip and hashes
under final run artifacts/study. See ARCHIVE.md after a fresh GitHub clone.
User authorizes direct execution and 15-minute heartbeat.
Remote root /home/liujianhua/wjq/TAISP using project .autodl/config.json.
T002 final source/run/report remain unchanged and valid; no meta-training.
