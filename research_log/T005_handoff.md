# T005 active experiment handoff

2026-09-12T06:49:43+08:00. T005 IN_PROGRESS, not DONE. R00724114d2 accepted/closed T004.

## Active run — do not duplicate

Run 20260912-064836-taisp-t005-coco200; release20260912-064831-taisp-t005-full; source9fb96c1.
Remote /home/liujianhua/wjq/TAISP/runs/20260912-064836-taisp-t005-coco200/artifacts/study.
Fresh full51-test suite before experiment, then200images x7cases x4variants =
5600rows,56adapted AP evaluations +7unadapted contemporary baselines (63total).
Same200IDs,sixcorruptions,clean separately,phi0=0,lr0.1,K3,JScoefficient1.
No tuning, meta-training, spatialISP or detector update. Only initial fixed
pseudo-support enters deployed objectives; oracle annotations stay in analysis.

## Completed gates and known numerical issue

Baseline11local tests6.38s and42real tests8.23s. FixedROI5tests4.67s.
Native objective/reset gate15tests81.40s. Smoke20260912-064017-taisp-t005-study-smoke,
sourcef64151a,release20260912-064013:50real tests87.80s,2images56rows63APevals,
27.37618s,exit0at06:42:23+08. Smoke receipts audited zero phi0, exact sharedgdet,
weights/normtargets; report pipeline passed, figure inspected after aspect fix.
Four focused report/norm/paired tests11.02s. No smoke-driven scientific changes.

Failed GPU repeated update test20260912-063551 retained:14pass1fail5.45s,
maxabs1.49682e-5 vs1e-6 tolerance. CUDA backward variability also recorded in
T003/T004. Same numerical repeat assertion/tolerance now runs realCPU; GPU
finite/nonzero gradients, state freeze and episode phi0 reset tested separately.
Initial empty-support unit assertion raw-image bitwise equality corrected to exact
ISP(phi0) output: existing identity arithmetic has roundoff, phi exactly unchanged.

## Completion actions

1. Use project .autodl/config.json and existing workflow scripts from
D:/work/claude-autodl/autodl-workflow-clean. Inspect active train.log; no duplicate.
2. After exit0 collect run into local research_log/remote_runs. Direct recursive
SCP worked but took several minutes even for smoke; packing requested run into
one tar.gz in project runs tree and fetching with Copy-FromAutodl is an appropriate
transfer option. Wait for transfer completion before postprocessing.
3. Local D:/anaconda3/python.exe -m scripts.report_t005
research_log/remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study.
4. Audit raw5600rows/28groupsx200,63metrics, exact fresh sharedgdet/loss/phi0,
normalized supportweights,normtargets,zero-updatefallbacks. Save receipt audit with
rawSHA. Preserve all raw predictions/logs/artifacts. If rawJSONL exceedsGitHub100MB,
retain raw locally/remotely and use tracked losslessgzip with checksum receipt.
5. Read full paired corrupted AND clean results; raw and norm-matched effects,
zero-coded cosine with valid-only descriptive values, support0/1-2/3-5/6+strata,
CE/JS components,AP1/3,clean phi norm,Taylor/saturation/latency. No family omitted.
6. Write T005_report.md and append full report to CODEX_TO_CHATGPT.md; mark
NEEDS_REVIEW only when all deliverables complete. Mirror final docs/generated
reports remote, commit/push. Do not edit CHATGPT_TO_CODEX.md. Await explicitT006.

## Protocol source

T005_plan.md locks deterministic greedy highestIoU then index matching and
view-specific fixed boxes. Stableweights arithmeticmean original scores;
pseudoCEbase,stablemeanCEbaseflip,stableJS+1. Empty support zero loss/update.
Full ROI91class logits incl background; frozen transform/backbone/ROI path bypass
RPN/NMS. No annotations/adapted predictions/family in the deployable adapter.
All seven per-image cases share ID clusters in appropriate report groups;
corrupted overall excludes clean. No zero-gradient sample dropped. Undefined
rawcosine=null; zero-coded for explicitly labeled aggregate paired measures.

Latest verification:51real tests passed85.36s;first2/200images26.0s. Run active.
