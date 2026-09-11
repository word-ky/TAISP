# T005 final report — detector-native self-supervision

**Status: NEEDS_REVIEW. All requested T005 deliverables are complete; research acceptance pending. No active experiment.**

Completed 2026-09-12 at07:33:51+08:00, exit0. Run `20260912-064836-taisp-t005-coco200`, release `20260912-064831-taisp-t005-full`, source/report helper `9fb96c1` (study implementation `f64151a`). The fixed study evaluated200 images, six corruptions plus clean, four objectives,5,600 observations,56 adapted and7 contemporaneous unadapted AP evaluations. Study elapsed2,619.637s (43.66min, excluding the preceding test/model startup).

## Research answer

Detector-native objectives provide a modest, supported initial alignment improvement over CLIP-global on the corrupted subset. Their positive paired mean-cosine differences survive the predeclared image-cluster intervals, and norm-matched one-step benefit rates improve by roughly3–4 percentage points. This signal is not explained solely by gradient magnitude. However, primary fixed-step beneficial-update rates remain near50%, their paired rate intervals cross zero, and AP remains family-dependent. Every detector-native objective beats CLIP atAP3 on gamma-s2, contrast-s2 and color-cast-s2 but loses on gamma-s1, contrast-s1 and color-cast-s1.

Stable filtering and JS do not establish an additional overall mechanism benefit over simple pseudo-confidence. Stable-minus-pseudo and JS-minus-stable cosine/benefit intervals cross zero; in the norm-matched mean-loss comparison stable is worse than pseudo. Clean subset AP is not damaged in this run, but clean ISP state changes are roughly2.4–2.7 times CLIP at3steps and are comparable to corrupted-image changes. This is not evidence of a clean-input identity mechanism.

**Recommendation to the research lead:** retain simple detector pseudo-confidence as the cheaper reference signal; do not claim stable-view consistency as the contribution or start meta-learning. The task decision rule remains unresolved for broad downstream utility: alignment improves, but AP is mixed and color-cast-s1 shows loss/AP disagreement. A next explicit task can test cross-detector transfer or an independent representation before committing to a final objective. No T006, tuning, spatial ISP or training was started.

## Exact implementation and protocol

- Reused existing8D global ISP, identity initialization, hard clamp, functional episodic SGD, frozen Faster R-CNN COCO_V1, CLIP-global baseline, COCO loader, corruption definitions, annotated oracle/evaluator and norm-matching helper. New files: `taisp/models/detector_signal.py`, `taisp/losses/detector_native.py`, `taisp/analysis/run_t005.py`, `configs/t005.yaml`, `scripts/report_t005.py`, and three corresponding test files. Existing detector/ISP/adaptation implementation was unchanged.
- Original base and horizontal-flip inference is eval/no_grad. Retain foreground detections with score>=0.5, descending stable-score top20 in each view. Flip mapping uses `(W-x2,y1,W-x1,y2)`. Same-class candidate matches require IoU>=0.5, sorted by descending IoU, then base index, then flip index; greedy unused-endpoint selection. This is deterministic greedy matching, not a maximum-cardinality assignment.
- Support retains each view's own predicted box. Boxes/classes/scores are detached and fixed for all steps. The enhanced flip view is the horizontal flip of the same ISP-enhanced base image. ROI logits use frozen transform/backbone, fixed boxes scaled by actual resized dimensions, box_roi_pool/box_head/box_predictor. RPN, proposal selection and NMS are bypassed in the differentiable path. Full91 logits include background; pseudo targets are original foreground predictions.
- `det_pseudo`: base-view CE weighted by normalized original base confidence. `det_stable`: mean base/flip CE, object weights from arithmetic mean of their original confidences and normalized to sum1. `det_stable_js`: stable CE plus weighted natural-log Jensen-Shannon divergence with fixed coefficient1.0. No box-regression objective. No CLIP contribution to these three objectives.
- Empty support returns differentiable zero loss: exact zero gradient and unchanged phi. No labels or replacement pseudo-targets are introduced, and fallback observations remain in every aggregate. The unchanged ISP(phi0) output can differ from raw input by identity-arithmetic roundoff; no-update is exact relative to that initial ISP output.
- Same200 IDs, seed20260912, GPU0, threads1, phi0=0, raw lr0.1,K3, external consistency/reg0. Gamma powers1.5/2.0; contrast factors0.6/0.3 around0.5; RGB cast gains[1.2,1.0,0.8]/[1.4,1.0,0.6]. Fixed order CLIP-global,pseudo,stable,stable+JS. No prompt/support/lr/JS tuning after smoke.
- One fresh annotated oracle gradient/loss per image/case is shared by all four variants, confined to analysis. Extra one-step norm match scales each nonzero self-gradient to the same-case CLIP norm, without labels; zero remains zero. Primary deployment results use ordinary fixed raw-phi lr.

The pre-run contract is [T005_plan.md](T005_plan.md). Corrupted overall keeps all six cases together when resampling200 image clusters; clean is separate. Intervals use2,000 percentile draws, seed20260912, exploratory without multiplicity adjustment. All1,200 corrupted episodes per variant stay in paired comparisons. Undefined zero-gradient cosine is stored asnull and explicitly zero-coded for aggregate cosine/positive-rate measures; valid-only cosine is separately reported. AP uses official COCO subset evaluation, not averaging per-image AP, and AP differences have no confidence intervals.

## Corrupted overall: raw and norm-matched mechanism

| Variant | Mean norm | Mean / median cosine* | Positive% | Raw benefit% | Norm-matched benefit% | Mean raw / matched detector-loss delta |
|---|---:|---:|---:|---:|---:|---:|
| global_generic | 0.12575 | 0.04532 / 0.05640 | 53.83 | 46.83 | 46.83 | +0.001969 / +0.001969 |
| det_pseudo | 0.48955 | 0.11757 / 0.19214 | 58.17 | 50.25 | 50.58 | -0.001557 / -0.001549 |
| det_stable | 0.39935 | 0.12431 / 0.18084 | 57.92 | 50.00 | 50.00 | -0.001655 / +0.000151 |
| det_stable_js | 0.42919 | 0.12271 / 0.18259 | 58.17 | 50.17 | 50.50 | -0.001944 / -0.000005 |

*Zero-coded aggregate cosine. Nonzero-gradient counts:1,200/1,196/1,194/1,194; valid-only means0.04532/0.11796/0.12494/0.12332. Four pseudo and six stable episodes have no support (the two stable variants share the same six episodes).

| Variant vs CLIP | Delta cosine [95%CI] | Delta positive pp [CI] | Delta raw benefit pp [CI] | Delta matched benefit pp [CI] | Delta raw mean loss [CI] | Delta matched mean loss [CI] |
|---|---:|---:|---:|---:|---:|---:|
| det_pseudo | 0.072 [0.008, 0.135] | 4.333 [-1.000, 9.333] | 3.417 [-0.252, 7.167] | 3.750 [0.500, 7.333] | -0.003526 [-0.006962, -0.000041] | -0.003517 [-0.005618, -0.001643] |
| det_stable | 0.079 [0.019, 0.139] | 4.083 [-1.000, 9.000] | 3.167 [-0.667, 6.917] | 3.167 [0.083, 6.333] | -0.003623 [-0.006931, -0.000245] | -0.001818 [-0.003617, -0.000049] |
| det_stable_js | 0.077 [0.017, 0.137] | 4.333 [-0.669, 9.167] | 3.333 [-0.250, 7.000] | 3.667 [0.331, 7.250] | -0.003912 [-0.007374, -0.000448] | -0.001974 [-0.003965, -0.000113] |

Native raw gradients are approximately3.2–3.9 times the CLIP mean norm. Norm matching retains modest benefit-frequency/mean-loss advantages over CLIP, but stable matched mean loss remains slightly positive (+0.000151) and stable+JS is essentially zero (-0.000005). Even the raw native mean-loss estimates have individual intervals crossing zero; paired differences relative to CLIP are better determined. Do not confuse superiority to a weak baseline with reliable improvement on each image.

## Stable filtering and JS ablations

| Contrast | Delta cosine [CI] | Delta raw benefit pp [CI] | Delta matched benefit pp [CI] | Delta matched mean loss [CI] |
|---|---:|---:|---:|---:|
| det_stable minus det_pseudo | 0.007 [-0.022, 0.037] | -0.250 [-3.500, 3.000] | -0.583 [-3.667, 2.419] | +0.001699 [+0.000599, +0.002798] |
| det_stable_js minus det_stable | -0.002 [-0.006, 0.003] | 0.167 [-2.250, 2.583] | 0.500 [-2.083, 3.167] | -0.000155 [-0.000873, +0.000531] |

The stable-versus-pseudo comparison changes both support selection and use of the flip view; it does not isolate either component alone. JS-versus-stable holds support and views fixed. The added JS term shows no supported overall advantage at the predeclared coefficient. No alternative coefficient was searched.

## All AP results and negative families

AP is in0–100 subset points. Before predictions were rerun in T005, including clean.

| Case | Before | CLIP1 /3 | Pseudo1 /3 | Stable1 /3 | Stable+JS1 /3 |
|---|---:|---:|---:|---:|---:|
| gamma_s1 | 38.145 | 38.349 / 38.680 | 37.976 / 38.067 | 38.001 / 38.352 | 38.080 / 38.069 |
| gamma_s2 | 36.446 | 36.582 / 36.409 | 36.866 / 36.819 | 36.727 / 36.734 | 36.785 / 36.717 |
| contrast_s1 | 36.175 | 36.369 / 36.498 | 36.546 / 36.307 | 36.819 / 36.473 | 36.844 / 36.430 |
| contrast_s2 | 31.175 | 31.115 / 31.277 | 31.854 / 31.989 | 31.393 / 31.323 | 30.957 / 31.603 |
| color_cast_s1 | 37.642 | 37.669 / 37.723 | 37.397 / 37.333 | 37.138 / 37.071 | 37.330 / 37.325 |
| color_cast_s2 | 35.908 | 35.846 / 35.528 | 35.795 / 35.864 | 36.097 / 36.161 | 36.094 / 36.116 |
| clean_s0 | 38.012 | 38.148 / 38.214 | 38.349 / 38.274 | 38.297 / 38.288 | 38.173 / 38.163 |

| Case | Pseudo deltaAP3 vs CLIP | Stable deltaAP3 vs CLIP | Stable+JS deltaAP3 vs CLIP |
|---|---:|---:|---:|
| gamma_s1 | -0.613 | -0.328 | -0.611 |
| gamma_s2 | +0.411 | +0.325 | +0.308 |
| contrast_s1 | -0.191 | -0.025 | -0.068 |
| contrast_s2 | +0.712 | +0.046 | +0.326 |
| color_cast_s1 | -0.390 | -0.652 | -0.398 |
| color_cast_s2 | +0.336 | +0.633 | +0.588 |
| clean_s0 | +0.060 | +0.074 | -0.051 |

The strongest pseudo-confidence case is contrast-s2: raw beneficial-step gain+12pp [2,21], cosine gain+0.203 [0.083,0.319], AP3+0.814 versus unadapted and+0.712 versus CLIP. This is a useful local signal, not an all-family conclusion. In color-cast-s1, stable confidence increases beneficial annotated-loss steps by+12.5pp [4,21] versus CLIP while AP3 falls0.652 points versus CLIP (0.571 versus unadapted). This direct loss/AP disagreement warrants the confirmation-bias concern in the task decision rule; it does not prove the underlying cause. Gamma-s1 also favors the CLIP baseline.

![T005 AP3 change vs unadapted](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/ap_change.png)

## Clean control

| Variant | Clean deltaAP1 /3 vs before | Mean raw phi norm1 /3 | Phi norm3 p95 | Saturation3% |
|---|---:|---:|---:|---:|
| global_generic | +0.136 / +0.202 | 0.01552 / 0.02877 | 0.05841 | 3.158 |
| det_pseudo | +0.338 / +0.262 | 0.04694 / 0.07799 | 0.21287 | 2.592 |
| det_stable | +0.285 / +0.276 | 0.03955 / 0.06956 | 0.17908 | 2.864 |
| det_stable_js | +0.162 / +0.151 | 0.04200 / 0.07306 | 0.19563 | 3.057 |

No clean AP degradation is observed at1 or3steps in this subset. This does not establish general safety: AP has no interval here, no margin was predeclared, and the same finite-step objective still changes clean images substantially relative to CLIP. All200 clean images have nonempty base/stable support.

## Support, confidence and no-update cases

Across corrupted images, original base/flip counts average7.160/7.232; stable count5.947. Mean stable/base fraction84.75%, symmetric match rate83.33% (ratios defined as0 for empty inputs). Selected base confidence mean0.8313, median0.8764, p05/p950.5431/0.9988. Stable mean confidence0.8659, median0.9121, p05/p950.6074/0.9989. Thus matching increases selected confidence but does not establish improved adaptation.

Pseudo has4/1200 no-update cases (0.33%); stable and stable+JS each6/1200 (0.50%). All16 variant observations have exact zero phi/gradient and zero measured oracle-loss change; none is excluded.

| Variant | Support | N | Mean cosine* | Raw benefit% | Matched benefit% | Mean raw loss delta |
|---|---|---:|---:|---:|---:|---:|
| det_pseudo | 0 | 4 | 0.00000 | 0.00 | 0.00 | +0.000000 |
| det_pseudo | 1-2 | 234 | 0.10989 | 48.29 | 49.15 | -0.000304 |
| det_pseudo | 3-5 | 416 | 0.08095 | 52.40 | 47.84 | -0.000689 |
| det_pseudo | 6+ | 546 | 0.14962 | 49.82 | 53.66 | -0.002767 |
| det_stable | 0 | 6 | 0.00000 | 0.00 | 0.00 | +0.000000 |
| det_stable | 1-2 | 330 | 0.15170 | 49.09 | 49.70 | +0.000374 |
| det_stable | 3-5 | 424 | 0.10193 | 49.29 | 49.53 | -0.002445 |
| det_stable | 6+ | 440 | 0.12703 | 52.05 | 51.36 | -0.002436 |
| det_stable_js | 0 | 6 | 0.00000 | 0.00 | 0.00 | +0.000000 |
| det_stable_js | 1-2 | 330 | 0.14733 | 50.91 | 52.73 | +0.000431 |
| det_stable_js | 3-5 | 424 | 0.10050 | 49.76 | 49.53 | -0.002844 |
| det_stable_js | 6+ | 440 | 0.12730 | 50.68 | 50.45 | -0.002884 |

Support strata are descriptive and select different episodes for pseudo versus stable; they are not causal matched subgroup contrasts. Per-family support counts/confidence distributions and strata, including clean, are preserved in the full analysis.

## Coordinate, loss and Taylor diagnostics

| Variant | Gamma% | Red% | Green% | Blue% | Contrast% | Brightness% | Tone% | Sharpen% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_generic | 3.41 | 29.00 | 26.18 | 12.30 | 19.14 | 8.55 | 0.82 | 0.60 |
| det_pseudo | 8.46 | 20.51 | 27.24 | 12.73 | 18.16 | 9.38 | 1.27 | 1.93 |
| det_stable | 8.66 | 21.60 | 25.96 | 12.49 | 18.40 | 9.25 | 1.25 | 1.89 |
| det_stable_js | 8.63 | 22.00 | 25.97 | 12.39 | 18.21 | 9.15 | 1.24 | 1.91 |

Energy is squared-coordinate fraction per episode, then averaged; zero-gradient episodes contribute0. RGB gains still carry about60% of native gradient energy versus67.5% for CLIP. This is descriptive, not a claim of eliminating coordinate conflict. Native own losses decrease on93.83%/95.08%/94.58% of corrupted episodes for pseudo/stable/JS, while beneficial annotated-loss steps stay near50%. Stable mean CE falls0.15444 to0.12960; adding JS lowers its mean JS from0.00804 to0.00635 but does not improve overall alignment or AP reliably.

| Variant | Mean Taylor prediction | Mean observed loss delta | Raw Taylor sign agreement% | Raw Spearman [CI] | Matched Spearman [CI] |
|---|---:|---:|---:|---:|---:|
| global_generic | -0.000068 | +0.001969 | 54.17 | 0.144 [0.083, 0.202] | 0.144 [0.083, 0.202] |
| det_pseudo | -0.008168 | -0.001557 | 59.08 | 0.325 [0.255, 0.391] | 0.165 [0.097, 0.230] |
| det_stable | -0.004427 | -0.001655 | 58.08 | 0.287 [0.221, 0.355] | 0.134 [0.062, 0.204] |
| det_stable_js | -0.004746 | -0.001944 | 58.17 | 0.283 [0.222, 0.348] | 0.146 [0.076, 0.214] |

Taylor uses -0.1 times the initial dot product with the shared annotated detector gradient. Native raw correlations improve but remain modest; norm-matched correlations return to roughly0.13–0.17. The annotated detector loss includes native proposal/matching/sampling behavior, so finite-step results are not guaranteed by the local derivative. No cached cross-run gradient comparisons are used.

## Costs and saturation

| Variant | Adapt / deployment seconds3 | Adapt peak MiB | Saturation3% |
|---|---:|---:|---:|
| global_generic | 0.1136 / 0.1136 | 841.7 | 3.470 |
| det_pseudo | 0.1444 / 0.1729 | 1855.2 | 3.488 |
| det_stable | 0.2806 / 0.3398 | 2653.7 | 2.951 |
| det_stable_js | 0.2795 / 0.3387 | 2653.7 | 2.949 |

Deployment adds original base setup28.48ms for pseudo and base plus flip/matching59.19ms for stable variants. Recorded mean setup peaks are1188.9MiB base and1192.1MiB flip/matching. Peaks are total allocated memory in the shared study process with both models resident, not isolated minimal deployment footprints. Adaptation timing includes existing diagnostics/synchronization; annotated oracle/AP/norm/extra CE-JS analyses are excluded. Fixed variant order and no optimization mean these are descriptive costs. Stable roughly doubles pseudo deployment latency without supported overall extra benefit.

## Validation, commands and retained failures

- Baseline:11 local ISP/adapt tests6.38s;42 real tests8.23s. Fixed ROI5tests4.67s, run20260912-063355-taisp-t005-roi-tests.
- Native loss gate20260912-063551-taisp-t005-loss-tests:14pass/1fail5.45s, GPU repeat discrepancy maxabs1.49682e-5 at unchanged1e-6 tolerance. Failed log retained. CUDA backward variability was already observed in T003/T004. The same numerical repeated-episode assertion now passes on realCPU; GPU finite nonzero phi gradients, frozen parameters/buffers and phi0 reset remain checked. Corrected gate15passed81.40s, run20260912-063641-taisp-t005-repeat-tests. No loss/optimizer/model change was made to improve outcomes.
- Empty-support unit tests initially assumed bitwise equality with raw input; phi was exactly0 but existing ISP identity arithmetic has roundoff. Corrected the test reference to exact unchanged ISP(phi0), with no tolerance widening or ISP change. An initial local command used a nonexistent test filename; corrected after inventory, no source failure.
- Real smoke20260912-064017-taisp-t005-study-smoke:50tests87.80s;2images56rows/all63APevaluations27.37618s,exit0. Four focused report/norm/paired tests11.02s. Figure label overlap was corrected by aspect auto; final figure visually inspected. No scientific change after smoke.
- Full run starts with51 real-model/regression tests passed85.36s. Tests cover flip geometry/matching ties, original-only no-grad support, label-free signatures, RPN bypass,91-class fixed ROI output, frozen model state, nonzero phi gradient, CE/JS equation including background, empty support, episode reset and zero-preserving aggregate statistics.
- Final raw audit passes5,600rows/28groupsx200/1,400image-cases/63metrics, exact sharedgdet/loss/support, zero phi0, finite gradients, normalized supportweights and nonzero norm-match targets. All fallback phi/gradient/loss deltas are exactly0. Raw bytes50,817,303; SHA256 `87554e26f6452d2063e92a50fb27d4cb9dcd5931dfd62942f48b6fac47e7bd87`.
- No full-study execution failure, tuning, dropped image, detector update, learned prompt/predictor, meta/source training, smooth clamp, spatial ISP or ViT³ mechanism. One fixed subset/seed; intervals reflect image resampling, not run-to-run numerical uncertainty.

Full launch (from the verified release, exact command also in run.sh/meta.json):

```bash
export TAISP_SOURCE_REVISION=9fb96c1 TAISP_REAL_MODELS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t005 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --output "$AUTODL_ARTIFACTS_DIR/study"
```

Local postprocessing:

```text
D:/anaconda3/python.exe -m scripts.report_t005 research_log/remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study
```

Environment: Python3.12.12, torch2.4.0, torchvision0.19.0+cu121, Transformers4.44.2, pycocotools2.0.10, NumPy1.26.4, Pillow12.3.0, CUDA12.1, RTX A6000. Local analysis Python3.12.7, NumPy1.26.4, SciPy1.13.1, Matplotlib3.9.2. Known nvidia-smi/NVML driver-library mismatch is present; CUDA experiment completed successfully without changing system drivers.

CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. Detector SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Annotation SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. Exact image IDs/hashes in subset.json. CLIP remains only the historical global baseline with the unchanged3 positive and6 negative prompts, listed verbatim in environment.json.

## Deliverables and recovery

- [Full per-case paired tables and support strata](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/results.md), [all statistics](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/analysis.json), [AP figure PDF](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/ap_change.pdf).
- [Raw5,600 observations](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/samples.jsonl), [all AP metrics](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/metrics.json), [receipt audit](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/receipt_audit.json), [environment/prompts](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/environment.json), [completion](remote_runs/20260912-064836-taisp-t005-coco200/artifacts/study/completion.json). All63 prediction files, raw logs, configs and run metadata retained.
- Raw JSONL fits the GitHub single-file limit and remains uncompressed. Remote original artifacts: `/home/liujianhua/wjq/TAISP/runs/20260912-064836-taisp-t005-coco200/artifacts/study`. Reports and recovery mirrored under remote project research_log.
- T005 NEEDS_REVIEW; await research acceptance/next explicit task. Heartbeat may execute new tasks directly, but must not rerun completed T005 or infer authorization for meta-learning from this modest signal.
