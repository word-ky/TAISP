# T007 final disjoint-set trust-radius report

Status: **NEEDS_REVIEW**. Full study completed without changing the predeclared protocol. No active experiment. T001–T006 accepted/closed; no T008 or meta-training started.

## Decision against the three requested endpoints

The disjoint subset partially replicates a finite-step scale benefit, but the three endpoints do not jointly establish a robust method. Hybrid improves FCOS beneficial-step frequency relative to raw pseudo and reduces average clean update magnitude. Its mean FCOS loss contrast against raw pseudo remains uncertain, and AP versus no adaptation remains mixed. The appropriate interpretation is **partial scale-benefit replication with unresolved loss-to-AP coupling**, not “hybrid confirmed,” complete failure out of sample, or a new direction/alignment mechanism.

1. **FCOS finite-step loss: partial support.** Across 1,200 corrupted episodes, raw pseudo is beneficial on 53.33% and hybrid on 56.67%; paired improvement +3.33 pp [1.33, 5.67]. Mean one-step loss change is +0.000471 for raw and -0.000423 for hybrid. Hybrid-minus-raw is -0.000894 [-0.002396, 0.000533], crossing zero. Hybrid's absolute mean change CI [-0.000753, -0.000083] is negative, and hybrid-minus-CLIP is -0.000958 [-0.001500, -0.000417]. These are distinct estimands: a supported absolute decrease and gain over CLIP do not imply a supported mean-loss gain over raw pseudo. K3 target mean changes are +0.000106 raw and -0.001047 hybrid; target benefit rates are 55.17% and 57.50% (descriptive K3 values).

2. **Hybrid versus no-adapt cross-detector AP: mixed.** At K3, hybrid FCOS AP exceeds raw pseudo on all six corrupted conditions, by +0.847/+0.221/+0.440/+0.211/+0.124/+0.002 points in gamma-s1, gamma-s2, contrast-s1, contrast-s2, color-s1, color-s2 order. But versus no adaptation it improves only four: +0.126/-0.302/-0.108/+0.294/+0.018/+0.132. The six-condition arithmetic mean gain is only +0.027 AP, not pooled COCO AP. At K1, target AP is lower than no adaptation on five of six corruptions. Source AP3 is lower on both gamma conditions (-0.231/-0.216); only contrast-s2 and the two color conditions improve on both detectors simultaneously at K3. Positive fixed-subset AP differences are descriptive, not statistically established AP improvements.

3. **Clean phi magnitude: supported average reduction, not identity preservation.** Mean clean phi1 decreases from 0.042264 raw to 0.015372 hybrid; phi3 decreases from 0.072558 to 0.038796, a 46.53% reduction in the ratio of means. Paired phi3 difference is -0.033762 [-0.043407, -0.024599]. Yet hybrid remains above CLIP phi3=0.029245 and adapts 199/200 clean inputs. Clean AP3 versus before is +0.228 source and only +0.011 FCOS; at K1 both decline (-0.091/-0.030). Clean hybrid source oracle loss3 increases +0.001705 while raw decreases -0.001518, despite the positive source AP3 difference. Smaller phi does not by itself guarantee universally safer finite-step behavior.

The hybrid direction is mathematically unchanged for a nonzero source gradient. Corrupted FCOS cosine is 0.130072 raw versus 0.130071 hybrid; source cosine is 0.129820 versus 0.129819. Same-forward collinearity error across all steps is at most 1.11e-15 when reconstructed in float64. Separate CUDA raw/hybrid forwards yield a maximum per-episode oracle-cosine difference of 0.001384; the average differences are around 1e-6, with unchanged positive-alignment rates. This is numerical variability, not higher cosine from norm matching.

Recommended next action: research review of this partial confirmation and the remaining loss/AP discrepancy. Preserve the signal as a diagnostic reference, but do not meta-train this inner objective or claim all three conditions passed. No controller, clipping, learned radius, prompt change or further experiment was added to improve the observed outcome.

## Sign flips, scale distribution and clean limitations

On corrupted inputs, raw harmful to hybrid beneficial FCOS loss1 flips occur in 7.92% [6.33, 9.75], while the reverse occurs in 4.58% [3.42, 5.83]. Both detectors jointly benefit on 30.50% hybrid versus 30.67% raw; paired -0.17 pp [-3.00, 2.58] does not support a joint-benefit increase. Thus the target benefit-rate gain does not imply a corresponding source or joint gain.

In the predeclared initial norm-ratio >=4 stratum (483 observations), favorable flips are 12.63% and reverse flips 5.80%. In the ratio <1 stratum (283, including nine zero-support cases), favorable/reverse flips are 8.13%/6.36%. Clean favorable/reverse flips are 7.00%/6.00%; in its ratio <1 stratum they are 8.47%/11.86%. These conditional descriptive results do not create a deployment gate.

The hybrid is not a cap: if CLIP's norm exceeds the source norm, it enlarges the source step. Initial scale exceeds one on 274/1,200 corrupted episodes (22.83%) and 58/200 clean images (29.00%). Hybrid phi3 exceeds raw phi3 on 289 corrupted episodes (24.08%) and 57 clean images (28.50%), despite lower means. This observed amplification is part of the specified equation and was left unchanged.

The initial detector/CLIP norm-ratio median is 2.889 corrupted and 2.422 clean. Scale-factor medians are 0.340 and 0.390, but means are 7.129 and 13.999 because small nonzero detector gradients produce large factors. P95 scale is 16.859 corrupted and 42.857 clean. The resulting hybrid norm remains tied to the CLIP norm; a large multiplicative factor is not itself a large absolute gradient. All four step distributions are retained below and in analysis.json. There were no zero CLIP norm cases.

## Frozen protocol and implementation

The deterministic new subset was committed before any new-subset model execution: algorithm/protocol `c5dc90a`, complete 200 IDs/JPEG hashes `55e83e0`. From all 5,000 COCO-val2017 IDs, exclude every historical ID, sort the remaining 4,800, sample 200 using Python random.Random(20260913), then sort the sample. Overlap is exactly zero; no images were replaced based on model outcomes. The annotation file is identical to the historical official file. Manifest SHA256: `1738d9d504233513893cc4a3f3b1e0a2f58d4665eb4df522e87255082fade1ea`.

Hybrid/driver implementation `4e87904`; smoke/report/full-run source `0df7e13`. The global eight-dimensional ISP starts at zero each episode, uses the unchanged hard clamp, lr=0.1 and K1/K3. Frozen Faster R-CNN COCO_V1 provides original-image detached score>=0.5 top20 support, fixed classes/boxes/confidence weights, and the unchanged T005 ROI CE. Frozen historical CLIP-global supplies only its gradient norm at each current phi. Update: g_det * norm(g_clip)/(norm(g_det)+1e-12). Empty support/zero source gradient gives exact no-update. No CLIP direction, target signal, annotation, synthetic severity or clean indicator enters the update.

FCOS COCO_V1 remains analysis/evaluation only, with the T006 pin unchanged: score0.2, NMS0.6, topk1000, max100 detections, center radius1.5, shorter-side800/max1333, size-divisible32, RGB mean [.485,.456,.406], std [.229,.224,.225]. Native annotated source/target gradients at phi0 and before losses are freshly computed once per image-case and shared by all variants in the analysis driver. Both detectors evaluate the same enhanced outputs. No-adapt predictions are a shared baseline, yielding 98 AP evaluations for seven conditions ×two detectors ×(one baseline+three variants×two steps).

Tests cover algebra including eps, exact zero source/CLIP norm behavior, current-phi recomputation, episodic reset, unchanged fixed support, source/CLIP frozen parameters and buffers, no accumulated model gradients, and no target/annotation adapter argument or analysis import. Real CPU repeat outputs are exactly equal; no bitwise CUDA repeat guarantee is claimed. Every step records both gradients/norms, scale, update, phi, saturation, support and timing. Step3 gradients are terminal diagnostics only, with no fourth update. Reported implementation time includes that terminal diagnostic work; it is not a claim for a minimal deployment implementation.

## Execution and evidence

Full run `20260912-105119-taisp-t007-coco200`, release `20260912-105102-taisp-t007-full`, source/report `0df7e13`. Started 2026-09-12T10:51:25+08; finished 11:35:47+08, exit0. Full suite: **59 real-model/regression tests passed in 127.00s**, eight known warnings. Study after tests/model setup: **2527.919981s**, 200 images, 4,200 adaptation rows, 1,400 image-cases, 21 groups ×200, 98 official AP evaluations.

Smoke `20260912-104530-taisp-t007-study-smoke`, release `20260912-104516-taisp-t007-smoke`, source `4e87904`: 58 real tests passed126.41s; two images,42rows,98APevals,35.857011s,exit0. Local algebra/selection/isolation four tests passed13.39s; report flip/ratio/unequal-cluster test passed1.19s. Smoke and full reports/figures were inspected. No experimental rerun or hyperparameter tuning was performed after results.

Full audit passes: exact committed manifest and zero historical overlap; complete pairing and counts; shared fixed support plus both annotated gradients/before losses; zero phi0; finite gradients; confidence weights; signed coordinate products; every-step norm-transfer and update equations; terminal step not applied; all prediction files. Each of raw and hybrid has nine corrupted fallback episodes (gamma1/2:1 each, contrast1/2:2 each, color1:1,color2:2), and one clean fallback. Their trajectories, source/target loss1/3 changes and self-gradients are exactly zero. All remain in denominators. Forty zero hybrid-gradient diagnostic steps correspond to ten empty episodes ×four recorded steps.

Operational failures: first full deployment `20260912-105051` encountered connection closure during upload and then extraction SSH failure (exit255); a fresh deployment succeeded before the single full run was started. First post-run packaging connection timed out (exit255); retry and complete download succeeded. Archive45,709,998 bytes, SHA256 `45312112a35ce567a0e93e136897f0d3de63324d74f802d3b020ec29cdae81d9`, verified locally. No scientific/data/model changes followed these failures. Known NVML mismatch warning remains; CUDA ran successfully without driver changes.

Raw samples.jsonl is 35,250,130 bytes, SHA256 `fd41b75af58e6b8eaffdc79405ee2cd604acf85505df9d402f4c9ea2c1d2f20a`. It is retained uncompressed along with all predictions, environment, subset, metrics, logs and receipt audit. Preserve every negative condition. Bootstrap uses 2,000 paired image-cluster percentile draws, seed20260912; selected conditions travel together, clean separately. Ratio-stratum fractions use cluster sums/counts for unequal case counts. Intervals are exploratory without multiplicity adjustment; AP has no fabricated per-image intervals. Source and target native loss units differ.

## Model pins, environment and commands

- clip_revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- clip_sha256: `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`
- detector_sha256: `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`
- annotation_sha256: `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`
- FCOS SHA256: `99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7`

Environment: Python3.12.12, torch2.4.0, torchvision0.19.0+cu121, Transformers4.44.2, pycocotools2.0.10, NumPy1.26.4, Pillow12.3.0, CUDA12.1, RTX A6000. Local analysis Python3.12.7, NumPy1.26.4, SciPy1.13.1, Matplotlib3.9.2. Exact prompts, package versions, model pins and evaluated IDs are in environment.json.

```bash
export TAISP_SOURCE_REVISION=0df7e13 TAISP_REAL_MODELS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t007 --data-root /home/liujianhua/wjq/TAISP/shared/coco200_t007 --output "$AUTODL_ARTIFACTS_DIR/study"
```

Local reporting: `PYTHONUTF8=1 D:/anaconda3/python.exe -m scripts.report_t007 research_log/remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study`.

## Absolute mechanism endpoints

Losses are after minus before (lower is better); beneficial means strictly negative. Undefined zero-gradient cosine is explicitly zero-coded for all-episode aggregates and separately retained as null in raw data.

| Group | Detector | Variant | Cosine | Benefit1 /3 % | Mean loss1 /3 | Both benefit1 % |
|---|---|---|---:|---:|---:|---:|
| corrupted_overall | source | global_generic | 0.055548 | 50.00 / 48.50 | -0.000088 / +0.001277 | 25.17 |
| corrupted_overall | target | global_generic | 0.038648 | 49.58 / 50.58 | +0.000535 / +0.000895 | 25.17 |
| corrupted_overall | source | det_pseudo | 0.129820 | 52.92 / 51.92 | -0.001257 / -0.001766 | 30.67 |
| corrupted_overall | target | det_pseudo | 0.130072 | 53.33 / 55.17 | +0.000471 / +0.000106 | 30.67 |
| corrupted_overall | source | det_pseudo_clip_radius | 0.129819 | 52.17 / 52.08 | -0.001354 / -0.001047 | 30.50 |
| corrupted_overall | target | det_pseudo_clip_radius | 0.130071 | 56.67 / 57.50 | -0.000423 / -0.001047 | 30.50 |
| severity_s1 | source | global_generic | 0.045217 | 48.67 / 48.33 | +0.000799 / +0.001446 | 24.67 |
| severity_s1 | target | global_generic | 0.033651 | 50.50 / 50.83 | +0.000402 / +0.000712 | 24.67 |
| severity_s1 | source | det_pseudo | 0.109690 | 52.50 / 49.83 | -0.000917 / -0.000914 | 30.17 |
| severity_s1 | target | det_pseudo | 0.114729 | 52.50 / 55.17 | +0.000434 / +0.000699 | 30.17 |
| severity_s1 | source | det_pseudo_clip_radius | 0.109693 | 49.50 / 52.83 | +0.000315 / -0.001026 | 29.33 |
| severity_s1 | target | det_pseudo_clip_radius | 0.114727 | 56.17 / 57.67 | -0.000438 / -0.000980 | 29.33 |
| severity_s2 | source | global_generic | 0.065880 | 51.33 / 48.67 | -0.000976 / +0.001107 | 25.67 |
| severity_s2 | target | global_generic | 0.043646 | 48.67 / 50.33 | +0.000668 / +0.001078 | 25.67 |
| severity_s2 | source | det_pseudo | 0.149950 | 53.33 / 54.00 | -0.001598 / -0.002617 | 31.17 |
| severity_s2 | target | det_pseudo | 0.145415 | 54.17 / 55.17 | +0.000507 / -0.000487 | 31.17 |
| severity_s2 | source | det_pseudo_clip_radius | 0.149945 | 54.83 / 51.33 | -0.003023 / -0.001068 | 31.67 |
| severity_s2 | target | det_pseudo_clip_radius | 0.145414 | 57.17 / 57.33 | -0.000408 / -0.001113 | 31.67 |
| clean_s0 | source | global_generic | 0.043354 | 53.00 / 53.50 | -0.001385 / -0.001226 | 27.50 |
| clean_s0 | target | global_generic | -0.033059 | 45.50 / 47.50 | -0.000187 / +0.000495 | 27.50 |
| clean_s0 | source | det_pseudo | 0.078017 | 52.00 / 58.50 | -0.002690 / -0.001518 | 30.00 |
| clean_s0 | target | det_pseudo | 0.149065 | 59.50 / 57.50 | -0.000940 / -0.000230 | 30.00 |
| clean_s0 | source | det_pseudo_clip_radius | 0.078021 | 50.00 / 46.00 | -0.001005 / +0.001705 | 26.50 |
| clean_s0 | target | det_pseudo_clip_radius | 0.149069 | 60.50 / 58.50 | -0.001199 / -0.002147 | 26.50 |

## Per-step norm-ratio and scale distributions

Scale is not bounded above by one; no clipping was added. Step3 is diagnostic only.

| Group | Step | Detector norm mean | CLIP norm mean | Hybrid norm mean | Ratio median /p95 | Scale mean /median /p95 |
|---|---:|---:|---:|---:|---:|---:|
| corrupted_overall | 0 | 0.457523 | 0.118464 | 0.117202 | 2.889 / 20.045 | 7.129 / 0.340 / 16.859 |
| corrupted_overall | 1 | 0.421240 | 0.113642 | 0.112380 | 2.763 / 20.904 | 7.097 / 0.355 / 18.233 |
| corrupted_overall | 2 | 0.379943 | 0.107036 | 0.105774 | 2.684 / 20.430 | 8.435 / 0.361 / 20.460 |
| corrupted_overall | 3 | 0.357166 | 0.102296 | 0.101034 | 2.529 / 19.645 | 8.182 / 0.387 / 25.460 |
| clean_s0 | 0 | 0.422641 | 0.154714 | 0.153723 | 2.422 / 14.875 | 13.999 / 0.390 / 42.857 |
| clean_s0 | 1 | 0.373122 | 0.137857 | 0.136866 | 2.272 / 17.521 | 13.975 / 0.429 / 42.425 |
| clean_s0 | 2 | 0.333646 | 0.127715 | 0.126724 | 2.010 / 15.261 | 16.118 / 0.489 / 43.126 |
| clean_s0 | 3 | 0.304035 | 0.113142 | 0.112151 | 2.063 / 14.640 | 18.536 / 0.472 / 71.175 |

Mean corrupted deployment time for CLIP/raw/hybrid is0.1147/0.2131/0.3455s per3step episode; peaks965.8/1960.2/2741.0MiB. All models are resident in the shared process, so these are measured implementation costs rather than isolated minimal footprints. Target analysis is excluded from adaptation/deployment timing.

## Artifacts and recovery

- [results.md](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/results.md)
- [analysis.json](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/analysis.json)
- [samples.jsonl](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/samples.jsonl)
- [metrics.json](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/metrics.json)
- [receipt_audit.json](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/receipt_audit.json)
- [environment.json](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/environment.json)
- [subset.json](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/subset.json)
- [completion.json](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/completion.json)
- [transfer_ap.pdf](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/transfer_ap.pdf)

- [Precommitted disjoint manifest](T007_subset.json)
- [Predeclared protocol](T007_plan.md)

Remote originals: `/home/liujianhua/wjq/TAISP/runs/20260912-105119-taisp-t007-coco200/artifacts/study`. Root project research_log mirrors recovery and final reports. No active run; await R009/T007 research review and an explicit subsequent task.

## Full per-condition AP, paired contrasts, clean control and ratio strata
# T007 disjoint-set trust-radius results

Source 0df7e13; images=200, rows=4200, smoke=None.

Frozen source pseudo direction, current-phi CLIP norm, eps1e-12, lr0.1,K1/3. Target FCOS/annotations analysis-only. Hybrid cosine is not a new direction. Both detectors evaluate the same enhanced images.

## Official subset AP / AP50 / AP75

| Detector | Case | Variant | Before AP /50 /75 | After1 AP /50 /75 | After3 AP /50 /75 | DeltaAP3 vs before / CLIP / raw |
|---|---|---|---:|---:|---:|---:|
| source | gamma_s1 | global_generic | 41.873 / 65.199 / 46.543 | 41.643 / 64.847 / 46.244 | 41.417 / 64.817 / 45.660 | -0.456 / +0.000 / -0.474 |
| source | gamma_s1 | det_pseudo | 41.873 / 65.199 / 46.543 | 41.438 / 65.143 / 45.974 | 41.891 / 65.325 / 46.802 | +0.018 / +0.474 / +0.000 |
| source | gamma_s1 | det_pseudo_clip_radius | 41.873 / 65.199 / 46.543 | 41.875 / 65.253 / 46.404 | 41.642 / 65.389 / 45.934 | -0.231 / +0.224 / -0.250 |
| target | gamma_s1 | global_generic | 44.262 / 62.926 / 48.480 | 44.023 / 62.778 / 48.591 | 44.156 / 62.304 / 48.419 | -0.106 / +0.000 / +0.615 |
| target | gamma_s1 | det_pseudo | 44.262 / 62.926 / 48.480 | 43.197 / 61.230 / 48.314 | 43.541 / 62.237 / 47.895 | -0.721 / -0.615 / +0.000 |
| target | gamma_s1 | det_pseudo_clip_radius | 44.262 / 62.926 / 48.480 | 44.184 / 62.714 / 48.314 | 44.388 / 62.607 / 49.967 | +0.126 / +0.232 / +0.847 |
| source | gamma_s2 | global_generic | 39.843 / 62.510 / 44.158 | 39.722 / 62.392 / 44.130 | 40.068 / 62.371 / 44.366 | +0.224 / +0.000 / +0.437 |
| source | gamma_s2 | det_pseudo | 39.843 / 62.510 / 44.158 | 39.475 / 61.953 / 43.321 | 39.630 / 62.350 / 43.350 | -0.213 / -0.437 / +0.000 |
| source | gamma_s2 | det_pseudo_clip_radius | 39.843 / 62.510 / 44.158 | 39.665 / 62.143 / 43.742 | 39.627 / 61.772 / 43.727 | -0.216 / -0.441 / -0.003 |
| target | gamma_s2 | global_generic | 41.704 / 59.672 / 46.389 | 41.485 / 59.646 / 45.489 | 41.471 / 59.833 / 45.201 | -0.233 / +0.000 / +0.290 |
| target | gamma_s2 | det_pseudo | 41.704 / 59.672 / 46.389 | 40.966 / 59.475 / 45.685 | 41.180 / 59.540 / 45.261 | -0.523 / -0.290 / +0.000 |
| target | gamma_s2 | det_pseudo_clip_radius | 41.704 / 59.672 / 46.389 | 41.529 / 59.740 / 45.798 | 41.401 / 59.708 / 45.611 | -0.302 / -0.070 / +0.221 |
| source | contrast_s1 | global_generic | 40.503 / 62.040 / 44.283 | 40.546 / 62.009 / 44.798 | 40.423 / 62.216 / 44.624 | -0.081 / +0.000 / -0.549 |
| source | contrast_s1 | det_pseudo | 40.503 / 62.040 / 44.283 | 41.066 / 62.395 / 44.860 | 40.972 / 62.713 / 44.174 | +0.469 / +0.549 / +0.000 |
| source | contrast_s1 | det_pseudo_clip_radius | 40.503 / 62.040 / 44.283 | 40.737 / 61.979 / 44.371 | 41.234 / 62.986 / 44.435 | +0.731 / +0.812 / +0.262 |
| target | contrast_s1 | global_generic | 42.150 / 59.554 / 46.317 | 41.918 / 59.395 / 45.871 | 42.056 / 59.357 / 46.157 | -0.094 / +0.000 / +0.454 |
| target | contrast_s1 | det_pseudo | 42.150 / 59.554 / 46.317 | 41.578 / 59.003 / 45.093 | 41.602 / 59.258 / 45.178 | -0.548 / -0.454 / +0.000 |
| target | contrast_s1 | det_pseudo_clip_radius | 42.150 / 59.554 / 46.317 | 41.978 / 59.432 / 45.895 | 42.042 / 59.473 / 46.258 | -0.108 / -0.014 / +0.440 |
| source | contrast_s2 | global_generic | 34.561 / 53.997 / 37.688 | 34.461 / 53.991 / 37.262 | 34.551 / 53.968 / 37.532 | -0.010 / +0.000 / +0.025 |
| source | contrast_s2 | det_pseudo | 34.561 / 53.997 / 37.688 | 34.447 / 53.811 / 36.833 | 34.526 / 54.185 / 37.199 | -0.035 / -0.025 / +0.000 |
| source | contrast_s2 | det_pseudo_clip_radius | 34.561 / 53.997 / 37.688 | 34.665 / 54.012 / 37.691 | 34.687 / 54.625 / 37.662 | +0.126 / +0.136 / +0.161 |
| target | contrast_s2 | global_generic | 35.109 / 50.755 / 37.931 | 34.808 / 50.323 / 37.473 | 34.830 / 50.352 / 37.651 | -0.279 / +0.000 / -0.362 |
| target | contrast_s2 | det_pseudo | 35.109 / 50.755 / 37.931 | 34.757 / 50.175 / 37.767 | 35.192 / 51.168 / 38.049 | +0.083 / +0.362 / +0.000 |
| target | contrast_s2 | det_pseudo_clip_radius | 35.109 / 50.755 / 37.931 | 35.079 / 50.868 / 38.022 | 35.403 / 51.063 / 38.753 | +0.294 / +0.573 / +0.211 |
| source | color_cast_s1 | global_generic | 42.293 / 64.347 / 45.831 | 42.388 / 64.732 / 45.684 | 42.326 / 64.478 / 45.356 | +0.034 / +0.000 / -0.257 |
| source | color_cast_s1 | det_pseudo | 42.293 / 64.347 / 45.831 | 42.922 / 65.377 / 47.071 | 42.584 / 65.134 / 46.269 | +0.291 / +0.257 / +0.000 |
| source | color_cast_s1 | det_pseudo_clip_radius | 42.293 / 64.347 / 45.831 | 42.448 / 64.798 / 45.921 | 42.711 / 64.882 / 46.175 | +0.419 / +0.385 / +0.128 |
| target | color_cast_s1 | global_generic | 42.066 / 61.410 / 45.547 | 42.211 / 61.481 / 45.821 | 42.125 / 61.435 / 45.684 | +0.059 / +0.000 / +0.165 |
| target | color_cast_s1 | det_pseudo | 42.066 / 61.410 / 45.547 | 41.995 / 61.550 / 45.518 | 41.960 / 61.450 / 45.557 | -0.106 / -0.165 / +0.000 |
| target | color_cast_s1 | det_pseudo_clip_radius | 42.066 / 61.410 / 45.547 | 42.020 / 61.415 / 45.576 | 42.084 / 61.608 / 45.653 | +0.018 / -0.041 / +0.124 |
| source | color_cast_s2 | global_generic | 40.102 / 61.634 / 43.847 | 40.246 / 61.662 / 43.968 | 40.441 / 62.231 / 44.328 | +0.340 / +0.000 / +0.122 |
| source | color_cast_s2 | det_pseudo | 40.102 / 61.634 / 43.847 | 40.309 / 61.753 / 43.839 | 40.319 / 61.706 / 44.243 | +0.218 / -0.122 / +0.000 |
| source | color_cast_s2 | det_pseudo_clip_radius | 40.102 / 61.634 / 43.847 | 40.095 / 61.702 / 43.948 | 40.211 / 61.869 / 43.615 | +0.110 / -0.230 / -0.108 |
| target | color_cast_s2 | global_generic | 40.051 / 58.234 / 43.306 | 39.852 / 58.185 / 43.262 | 39.863 / 58.275 / 43.425 | -0.188 / +0.000 / -0.318 |
| target | color_cast_s2 | det_pseudo | 40.051 / 58.234 / 43.306 | 40.249 / 58.307 / 43.609 | 40.182 / 58.293 / 43.615 | +0.130 / +0.318 / +0.000 |
| target | color_cast_s2 | det_pseudo_clip_radius | 40.051 / 58.234 / 43.306 | 40.121 / 58.357 / 43.451 | 40.183 / 58.201 / 43.353 | +0.132 / +0.320 / +0.002 |
| source | clean_s0 | global_generic | 42.898 / 66.355 / 45.985 | 42.677 / 66.469 / 46.177 | 42.630 / 66.519 / 45.758 | -0.268 / +0.000 / -0.318 |
| source | clean_s0 | det_pseudo | 42.898 / 66.355 / 45.985 | 42.793 / 66.348 / 45.834 | 42.948 / 66.339 / 45.744 | +0.050 / +0.318 / +0.000 |
| source | clean_s0 | det_pseudo_clip_radius | 42.898 / 66.355 / 45.985 | 42.807 / 66.405 / 46.375 | 43.126 / 66.588 / 45.877 | +0.228 / +0.496 / +0.178 |
| target | clean_s0 | global_generic | 44.023 / 63.552 / 48.743 | 44.021 / 63.716 / 48.557 | 43.925 / 63.655 / 48.991 | -0.098 / +0.000 / +0.232 |
| target | clean_s0 | det_pseudo | 44.023 / 63.552 / 48.743 | 43.925 / 62.987 / 48.588 | 43.693 / 62.947 / 48.282 | -0.330 / -0.232 / +0.000 |
| target | clean_s0 | det_pseudo_clip_radius | 44.023 / 63.552 / 48.743 | 43.993 / 63.661 / 48.354 | 44.034 / 63.450 / 48.547 | +0.011 / +0.108 / +0.341 |

All K1/K3 AP/AP50/AP75 deltas retained in analysis.json; no invented AP CIs.

## Paired one-step finite behavior

| Group | Detector | Contrast | Delta cosine [CI] (not a direction gain) | Delta benefit pp [CI] | Delta mean loss [CI] |
|---|---|---|---:|---:|---:|
| corrupted_overall | source | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | -0.750 [-4.333, 2.583] | -0.000097 [-0.002457, +0.002210] |
| corrupted_overall | target | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | 3.333 [1.333, 5.667] | -0.000894 [-0.002396, +0.000533] |
| corrupted_overall | source | det_pseudo_clip_radius-minus-global_generic | 0.074 [0.010, 0.143] | 2.167 [-1.333, 5.833] | -0.001266 [-0.002935, +0.000203] |
| corrupted_overall | target | det_pseudo_clip_radius-minus-global_generic | 0.091 [0.037, 0.147] | 7.083 [2.167, 11.917] | -0.000958 [-0.001500, -0.000417] |
| corrupted_overall | source | det_pseudo-minus-global_generic | 0.074 [0.010, 0.143] | 2.917 [-0.833, 6.667] | -0.001169 [-0.003839, +0.001308] |
| corrupted_overall | target | det_pseudo-minus-global_generic | 0.091 [0.037, 0.147] | 3.750 [-0.833, 8.583] | -0.000064 [-0.001698, +0.001664] |
| severity_s1 | source | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | -3.000 [-7.833, 1.671] | +0.001232 [-0.001729, +0.003983] |
| severity_s1 | target | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | 3.667 [1.000, 6.500] | -0.000873 [-0.002903, +0.001017] |
| severity_s1 | source | det_pseudo_clip_radius-minus-global_generic | 0.064 [-0.012, 0.144] | 0.833 [-3.837, 5.833] | -0.000484 [-0.002738, +0.001659] |
| severity_s1 | target | det_pseudo_clip_radius-minus-global_generic | 0.081 [0.012, 0.154] | 5.667 [-0.333, 12.000] | -0.000840 [-0.001612, -0.000162] |
| severity_s1 | source | det_pseudo-minus-global_generic | 0.064 [-0.012, 0.144] | 3.833 [-1.337, 9.167] | -0.001716 [-0.004861, +0.001486] |
| severity_s1 | target | det_pseudo-minus-global_generic | 0.081 [0.012, 0.154] | 2.000 [-3.833, 8.333] | +0.000033 [-0.002075, +0.002320] |
| severity_s2 | source | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, -0.000] | 1.500 [-3.167, 5.833] | -0.001426 [-0.005024, +0.002000] |
| severity_s2 | target | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | 3.000 [0.167, 6.167] | -0.000916 [-0.002933, +0.000933] |
| severity_s2 | source | det_pseudo_clip_radius-minus-global_generic | 0.084 [0.015, 0.152] | 3.500 [-1.167, 8.333] | -0.002048 [-0.004342, +0.000203] |
| severity_s2 | target | det_pseudo_clip_radius-minus-global_generic | 0.102 [0.036, 0.168] | 8.500 [2.833, 14.667] | -0.001076 [-0.001678, -0.000470] |
| severity_s2 | source | det_pseudo-minus-global_generic | 0.084 [0.015, 0.152] | 2.000 [-2.833, 6.833] | -0.000622 [-0.004444, +0.002999] |
| severity_s2 | target | det_pseudo-minus-global_generic | 0.102 [0.036, 0.168] | 5.500 [-0.167, 11.333] | -0.000161 [-0.002107, +0.002091] |
| gamma_s1 | source | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | -7.500 [-15.500, 0.000] | +0.000521 [-0.003867, +0.005225] |
| gamma_s1 | target | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | 3.500 [-1.000, 8.500] | -0.001798 [-0.004716, +0.000617] |
| gamma_s1 | source | det_pseudo_clip_radius-minus-global_generic | 0.114 [0.007, 0.224] | 1.000 [-7.000, 8.500] | -0.000962 [-0.004478, +0.002739] |
| gamma_s1 | target | det_pseudo_clip_radius-minus-global_generic | 0.105 [-0.000, 0.210] | 11.500 [2.500, 21.000] | -0.000984 [-0.001872, +0.000019] |
| gamma_s1 | source | det_pseudo-minus-global_generic | 0.114 [0.007, 0.224] | 8.500 [0.500, 16.500] | -0.001483 [-0.006614, +0.003304] |
| gamma_s1 | target | det_pseudo-minus-global_generic | 0.105 [-0.000, 0.210] | 8.000 [-1.512, 18.000] | +0.000814 [-0.001910, +0.003971] |
| gamma_s2 | source | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, -0.000] | 0.500 [-8.000, 9.000] | -0.004706 [-0.010272, +0.000364] |
| gamma_s2 | target | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | 3.000 [-2.500, 8.500] | -0.002614 [-0.005140, -0.000176] |
| gamma_s2 | source | det_pseudo_clip_radius-minus-global_generic | 0.115 [0.009, 0.217] | -2.000 [-10.000, 6.000] | -0.002137 [-0.006101, +0.001636] |
| gamma_s2 | target | det_pseudo_clip_radius-minus-global_generic | 0.070 [-0.032, 0.175] | 11.500 [1.500, 21.000] | -0.001265 [-0.002387, -0.000188] |
| gamma_s2 | source | det_pseudo-minus-global_generic | 0.115 [0.009, 0.217] | -2.500 [-10.500, 5.500] | +0.002569 [-0.002909, +0.008338] |
| gamma_s2 | target | det_pseudo-minus-global_generic | 0.070 [-0.032, 0.175] | 8.500 [-1.000, 17.500] | +0.001348 [-0.001478, +0.004212] |
| contrast_s1 | source | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | -8.500 [-16.500, -1.000] | +0.004884 [-0.000913, +0.010098] |
| contrast_s1 | target | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | 4.000 [-1.000, 9.000] | -0.000913 [-0.004831, +0.002697] |
| contrast_s1 | source | det_pseudo_clip_radius-minus-global_generic | 0.080 [-0.042, 0.207] | 1.000 [-7.000, 9.500] | -0.000116 [-0.004107, +0.003742] |
| contrast_s1 | target | det_pseudo_clip_radius-minus-global_generic | 0.074 [-0.039, 0.191] | 7.500 [-1.512, 17.000] | -0.001174 [-0.002775, +0.000304] |
| contrast_s1 | source | det_pseudo-minus-global_generic | 0.080 [-0.042, 0.207] | 9.500 [1.000, 18.500] | -0.005000 [-0.010751, +0.000908] |
| contrast_s1 | target | det_pseudo-minus-global_generic | 0.074 [-0.039, 0.191] | 3.500 [-6.000, 13.000] | -0.000261 [-0.004055, +0.003845] |
| contrast_s2 | source | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, -0.000] | -1.500 [-8.500, 5.000] | +0.000408 [-0.007734, +0.008942] |
| contrast_s2 | target | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | 6.500 [1.500, 11.500] | +0.000241 [-0.004200, +0.004545] |
| contrast_s2 | source | det_pseudo_clip_radius-minus-global_generic | 0.124 [0.010, 0.240] | 5.000 [-3.500, 13.500] | -0.001191 [-0.005446, +0.003218] |
| contrast_s2 | target | det_pseudo_clip_radius-minus-global_generic | 0.199 [0.086, 0.315] | 14.500 [5.500, 24.000] | -0.001859 [-0.003394, -0.000551] |
| contrast_s2 | source | det_pseudo-minus-global_generic | 0.124 [0.010, 0.240] | 6.500 [-1.512, 15.000] | -0.001600 [-0.010935, +0.006750] |
| contrast_s2 | target | det_pseudo-minus-global_generic | 0.199 [0.086, 0.315] | 8.000 [-1.500, 17.000] | -0.002099 [-0.006487, +0.002684] |
| color_cast_s1 | source | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | 7.000 [0.000, 14.000] | -0.001709 [-0.006630, +0.002949] |
| color_cast_s1 | target | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | 3.500 [-1.000, 8.000] | +0.000092 [-0.002843, +0.003063] |
| color_cast_s1 | source | det_pseudo_clip_radius-minus-global_generic | -0.001 [-0.105, 0.102] | 0.500 [-7.500, 8.500] | -0.000375 [-0.003917, +0.002890] |
| color_cast_s1 | target | det_pseudo_clip_radius-minus-global_generic | 0.064 [-0.048, 0.173] | -2.000 [-11.500, 7.000] | -0.000362 [-0.000846, +0.000075] |
| color_cast_s1 | source | det_pseudo-minus-global_generic | -0.001 [-0.105, 0.102] | -6.500 [-15.500, 2.000] | +0.001334 [-0.003838, +0.006304] |
| color_cast_s1 | target | det_pseudo-minus-global_generic | 0.064 [-0.048, 0.173] | -5.500 [-14.500, 3.500] | -0.000455 [-0.003574, +0.002687] |
| color_cast_s2 | source | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | 5.500 [-2.500, 13.500] | +0.000020 [-0.003905, +0.004089] |
| color_cast_s2 | target | det_pseudo_clip_radius-minus-det_pseudo | -0.000 [-0.000, 0.000] | -0.500 [-5.000, 4.500] | -0.000374 [-0.003057, +0.002293] |
| color_cast_s2 | source | det_pseudo_clip_radius-minus-global_generic | 0.013 [-0.095, 0.119] | 7.500 [-0.500, 16.000] | -0.002814 [-0.006218, +0.000647] |
| color_cast_s2 | target | det_pseudo_clip_radius-minus-global_generic | 0.036 [-0.065, 0.140] | -0.500 [-9.500, 9.000] | -0.000105 [-0.000494, +0.000263] |
| color_cast_s2 | source | det_pseudo-minus-global_generic | 0.013 [-0.095, 0.119] | 2.000 [-6.500, 10.500] | -0.002835 [-0.007320, +0.001585] |
| color_cast_s2 | target | det_pseudo-minus-global_generic | 0.036 [-0.065, 0.140] | 0.000 [-9.012, 9.500] | +0.000269 [-0.002420, +0.003067] |
| clean_s0 | source | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | -2.000 [-10.500, 6.500] | +0.001685 [-0.002700, +0.006267] |
| clean_s0 | target | det_pseudo_clip_radius-minus-det_pseudo | 0.000 [-0.000, 0.000] | 1.000 [-4.000, 6.000] | -0.000259 [-0.002163, +0.001380] |
| clean_s0 | source | det_pseudo_clip_radius-minus-global_generic | 0.035 [-0.074, 0.147] | -3.000 [-12.000, 5.500] | +0.000379 [-0.003741, +0.004691] |
| clean_s0 | target | det_pseudo_clip_radius-minus-global_generic | 0.182 [0.069, 0.292] | 15.000 [5.500, 24.000] | -0.001012 [-0.002081, +0.000160] |
| clean_s0 | source | det_pseudo-minus-global_generic | 0.035 [-0.074, 0.147] | -1.000 [-10.000, 8.000] | -0.001306 [-0.006535, +0.003679] |
| clean_s0 | target | det_pseudo-minus-global_generic | 0.182 [0.069, 0.292] | 14.000 [5.000, 23.000] | -0.000753 [-0.002661, +0.001417] |

## Clean and corrupted update magnitudes

| Group | Variant | Phi1 /3 | Saturation before /1 /3 % | Deploy sec3 | Peak MiB | Fallback% |
|---|---|---:|---:|---:|---:|---:|
| corrupted_overall | global_generic | 0.01185 / 0.02484 | 3.228 / 2.314 / 2.552 | 0.1147 | 965.8 | 0.00 |
| corrupted_overall | det_pseudo | 0.04575 / 0.08406 | 3.228 / 2.673 / 2.779 | 0.2131 | 1960.2 | 0.75 |
| corrupted_overall | det_pseudo_clip_radius | 0.01172 / 0.03130 | 3.228 / 2.132 / 2.311 | 0.3455 | 2741.0 | 0.75 |
| clean_s0 | global_generic | 0.01547 / 0.02925 | 2.218 / 2.015 / 2.582 | 0.1154 | 962.6 | 0.00 |
| clean_s0 | det_pseudo | 0.04226 / 0.07256 | 2.218 / 2.057 / 2.841 | 0.2130 | 1959.5 | 0.50 |
| clean_s0 | det_pseudo_clip_radius | 0.01537 / 0.03880 | 2.218 / 1.564 / 1.940 | 0.3414 | 2741.7 | 0.50 |

## Target loss sign flips by initial detector / CLIP norm ratio

Strict harmful>0 and beneficial<0; zeros separate. Strata only analyze outcomes, never control deployment.

| Group | Ratio | N | Raw harmful to hybrid beneficial% [CI] | Reverse% [CI] | Either zero% [CI] |
|---|---|---:|---:|---:|---:|
| corrupted_overall | all | 1200 | 7.917 [6.333, 9.750] | 4.583 [3.417, 5.833] | 0.750 [0.000, 2.000] |
| corrupted_overall | [0,1) | 283 | 8.127 [4.029, 12.617] | 6.360 [3.802, 9.059] | 3.180 [0.000, 8.511] |
| corrupted_overall | [1,2) | 173 | 1.734 [0.000, 3.933] | 0.578 [0.000, 1.786] | 0.000 [0.000, 0.000] |
| corrupted_overall | [2,4) | 261 | 3.065 [1.163, 5.357] | 3.065 [1.141, 5.200] | 0.000 [0.000, 0.000] |
| corrupted_overall | [4,inf) | 483 | 12.629 [9.829, 15.529] | 5.797 [3.788, 8.138] | 0.000 [0.000, 0.000] |
| clean_s0 | all | 200 | 7.000 [3.500, 10.500] | 6.000 [3.000, 9.500] | 0.500 [0.000, 1.500] |
| clean_s0 | [0,1) | 59 | 8.475 [1.695, 16.949] | 11.864 [5.085, 20.339] | 1.695 [0.000, 5.085] |
| clean_s0 | [1,2) | 31 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| clean_s0 | [2,4) | 43 | 4.651 [0.000, 11.628] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| clean_s0 | [4,inf) | 67 | 10.448 [4.478, 17.910] | 7.463 [1.493, 14.925] | 0.000 [0.000, 0.000] |

Step0/1/2/3 detector/CLIP/update norms, ratios/scales, collinearity, support, source/target loss1/3 and joint fractions are saved in analysis.json. Step3 gradients are terminal diagnostics, not an additional update. Raw observations retain every trajectory and failure/fallback.

![Hybrid target and source AP3 deltas](remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study/transfer_ap.png)
