# T006 final cross-detector transfer report

Status: **NEEDS_REVIEW**. Full study completed; no active experiment. T001–T005 accepted/closed.

## Interpretation for R008

T006 supports a modest transferable direction signal relative to CLIP, with mixed downstream performance. It does not support a pure source-only explanation, but it also does not establish a broadly beneficial detector-agnostic restoration method. No meta-training, T007, learned gate, new prompt, spatial ISP, smooth clamp or outcome-driven tuning was started.

Across six corruptions, pseudo-confidence improves FCOS cosine from -0.01162 to 0.08159: paired gain +0.093 [0.036, 0.152]. Positive target alignment rises from 48.33% to 56.75%. Target beneficial-step rate rises from 48.92% to 54.33%, paired +5.42 pp [0.58, 10.25]; after matching to the same-case CLIP gradient norm it is 56.17%, paired +7.25 pp [2.58, 12.00]. These exploratory image-cluster intervals support some transfer beyond the source detector and beyond a gradient-norm increase alone.

However, ordinary pseudo target mean loss change is +0.001474 (worse), versus CLIP +0.000718. Their paired difference +0.000756 has CI [-0.001843, 0.003888], so there is no supported improvement in raw mean target loss. A larger beneficial fraction can coexist with larger harmful changes. Norm matching gives target mean change -0.000434; its paired difference versus CLIP is -0.001152 [-0.001803, -0.000516]. The absolute matched mean CI [-0.000920, 0.000038] still crosses zero. This is evidence about direction versus update magnitude, not an evaluated deployment controller; norm-matched AP was not run or claimed.

At K=3, FCOS pseudo AP exceeds CLIP on 5/6 corruptions, but exceeds no adaptation on only 3/6. Gamma-s1/s2 and contrast-s1 improve versus no adaptation; contrast-s2 and both color-cast levels degrade. At K=1 it loses to CLIP on 4/6 conditions. The strong source contrast-s2 gain (+0.909 AP at K3 versus before) coincides with a target decline (-0.274 AP). Thus favorable oracle alignment and useful AP are distinct findings. No AP confidence intervals or full-COCO claims are made.

The source s2-versus-s1 split repeats: pseudo-minus-CLIP AP3 macro is +0.563 on s2 and -0.324 on s1. It does not transfer as the same severity-only rule: FCOS pseudo-minus-CLIP AP3 macro is +0.205 for both severities; target alignment and norm-matched benefit improve in both s1 and s2. Target AP3 versus no adaptation macro is only +0.044 / +0.089. These are arithmetic means of the three official condition AP values, not pooled COCO AP. A severity-only explanation is not supported by the target results.

Annotated source-target directions have mean cosine 0.366 [0.325, 0.405] and positive agreement 75.17%; about 24.83% are nonpositive. The detectors share substantial but incomplete low-dimensional direction agreement. Both losses decrease for 31.00% of ordinary pseudo episodes, versus 23.67% for CLIP: paired +7.33 pp [3.33, 11.00]. Matched pseudo jointly benefits 29.67%, paired +6.00 pp [2.50, 9.42]. This is partial joint transfer, not universal agreement.

Clean AP3 improves by +0.310 on source and +0.804 on FCOS versus before. Nevertheless clean mean phi3 norm is 0.07801 versus CLIP 0.02877 (about 2.71x), with support on all 200 clean images. The current signal has no measured identity/no-adaptation behavior on clean inputs. Clean AP improvement on this fixed subset does not remove that limitation.

Recommended research judgment: retain simple source pseudo-confidence as a modest cross-detector reference; do not call it pure detector-loss gaming, robust restoration, or severity-conditional transfer. The mismatch between raw/matched losses and between source/target AP warrants research review of update magnitude and loss-to-AP coupling. A second architecture with the same ResNet-50 backbone family and COCO training is one transfer test, not independence from training distribution or proof across detector families. Await the research lead's next explicit task.

## Implementation and isolation

FCOS_ResNet50_FPN_Weights.COCO_V1 is analysis/evaluation only. Version, checkpoint hash and inference defaults were fixed before smoke and full results: score 0.2, NMS 0.6, topk 1000, max 100 detections, center radius 1.5; shorter-side 800/max 1333, size-divisible 32, RGB mean [.485,.456,.406], std [.229,.224,.225]. Installed FCOS.forward was inspected: the oracle changes only the top-level training flag to select native classification+bbox_regression+bbox_ctrness; children remain eval and the top flag is restored. Normal eval inference supplies AP.

The source remains frozen Faster R-CNN COCO_V1 with exactly the T005 base-view fixed pseudo-box/class/confidence ROI CE. Original support is detached, score >=0.5, top20; confidence weights sum to 1. There is no target in the adaptation signature or objective. Independent annotated source/target gradients are computed once per image-case at fresh zero 8D phi and shared across variants only in the analysis driver. Both models receive the exact same enhanced tensors at K1/3. No target or annotation influences support, update, step size or CLIP norm matching.

The real isolation test compares source adaptation before the target object exists against adaptation after target construction, with target.forward trapped: CPU phi and enhanced image are exactly equal. FCOS native GPU phi gradients are finite/nonzero, all parameters/buffers remain unchanged, no parameter gradients accumulate, and CUDA RNG is unchanged. Existing source/deploy label-isolation tests and all regressions pass.

## Runtime, completeness and reproducibility

Full run `20260912-092401-taisp-t006-coco200`, release `20260912-092347-taisp-t006-full`, source/report `d7d0510`; driver `8c0f480`, target isolation `6429337`. Started 09:24:05+08; finished 2026-09-12T09:51:22+08, exit 0. Study 1527.182541 seconds after tests/model setup, 200 images, 2,800 rows, 1,400 paired image-cases, 14 groups ×200, 70 official AP evaluations. Full real-model suite: 54 passed, 7 known warnings, 102.35 seconds.

Baseline 51 real tests in 84.76s; target/isolation run 20260912-090851: 4 passed in 20.60s. Smoke 20260912-091109: 53 real tests in 100.40s, 2 images / 28 rows / 70 AP evaluations in 16.280468s, exit 0. Local report/projection/joint/Taylor test passed in 4.67s. Smoke and full-run raw receipts retained; the full figure was inspected.

Audit passes: same 200 IDs as T005; both oracle gradients/initial losses and fixed source support shared across paired variants; fresh phi0 zero; finite 8D gradients; score-normalized weights; CLIP-only norm reference; signed products/Taylor calculations; all 70 prediction files; FCOS metadata agrees with all corresponding predeclared fields. The pin file additionally records its cache path. Four pseudo fallback cases (contrast-s2: 2, color-cast-s1: 1, color-cast-s2: 1) have exact-zero phi, self-gradient, source/target raw/matched loss deltas. All remain in denominators. None occurred on clean.

Observed numerical reproducibility limit: contemporaneous source AP/benefit values differ slightly from T005 despite the unchanged scientific protocol (e.g. source gamma-s1 pseudo AP3: 38.185 versus 38.067). This study reports fresh paired source/target measurements; it does not substitute historical numbers. The prior CUDA backward repeatability discrepancy and CPU exact-isolation test are documented; no bitwise CUDA repeat guarantee or result-driven rerun is claimed.

Observed operational failures: initial FCOS Python download failed certificate verification; using the system CA bundle resolved it without disabling TLS verification. One download SSH timeout was retried. Smoke SFTP transfer stalled; only its matching scp process was stopped and the workflow's legacy SCP retry succeeded with matching archive hash. Full receipt packaging's first SSH connection closed (exit 255); retry succeeded. Full archive 31,081,757 bytes, SHA256 `1c121d5904a0df5dd5b1a72d3e658752dbbfa9343a8d69be83c3c95d75071553`, verified locally. No experiment/model/data changes followed these failures. Known NVML mismatch warning remains; CUDA completed successfully and system drivers were not changed.

Environment: Python 3.12.12, torch 2.4.0, torchvision 0.19.0+cu121, Transformers 4.44.2, pycocotools 2.0.10, NumPy 1.26.4, Pillow 12.3.0, CUDA 12.1, RTX A6000. Local reporting Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2.

Raw samples.jsonl: 23,544,440 bytes; SHA256 `eec9ff8b8a8c14442d0b4a5d25cd4fb896caca63b3e92892bd9f41a960f9ee70`. Retain uncompressed locally, remotely and in GitHub. No incomplete result or negative family excluded.

## Commands and pins

```bash
export TAISP_SOURCE_REVISION=d7d0510 TAISP_REAL_MODELS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t006 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --output "$AUTODL_ARTIFACTS_DIR/study"
```

Local report: `D:/anaconda3/python.exe -m scripts.report_t006 research_log/remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study`.

- clip_revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- clip_sha256: `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`
- detector_sha256: `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`
- annotation_sha256: `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`
- FCOS SHA256: `99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7`

## Mean oracle-loss deltas

After minus before; lower is better. Loss units differ across models. Paired pseudo-minus-CLIP intervals use2000 image-cluster draws, seed20260912; all selected cases travel together, exploratory without multiplicity adjustment.

| Group | Detector | CLIP loss1 /3 | Pseudo loss1 /3 | Pseudo matched loss1 | Paired raw loss1 [CI] | Paired matched loss1 [CI] |
|---|---|---:|---:|---:|---:|---:|
| corrupted_overall | source | +0.001947 / +0.003362 | -0.001573 / -0.003171 | -0.001188 | -0.003519 [-0.006977, +0.000069] | -0.003135 [-0.005183, -0.001272] |
| corrupted_overall | target | +0.000718 / +0.001609 | +0.001474 / +0.001357 | -0.000434 | +0.000756 [-0.001843, +0.003888] | -0.001152 [-0.001803, -0.000516] |
| severity_s1 | source | +0.000275 / +0.000743 | -0.001153 / -0.001400 | -0.000895 | -0.001428 [-0.004815, +0.001985] | -0.001170 [-0.003162, +0.000776] |
| severity_s1 | target | +0.000576 / +0.000901 | +0.001264 / +0.002500 | -0.000464 | +0.000689 [-0.001214, +0.002624] | -0.001040 [-0.001669, -0.000488] |
| severity_s2 | source | +0.003618 / +0.005981 | -0.001993 / -0.004943 | -0.001482 | -0.005611 [-0.010634, -0.000177] | -0.005100 [-0.008328, -0.002075] |
| severity_s2 | target | +0.000860 / +0.002317 | +0.001684 / +0.000213 | -0.000404 | +0.000824 [-0.003158, +0.005601] | -0.001264 [-0.002254, -0.000301] |
| clean_s0 | source | -0.000354 / -0.002588 | -0.003936 / -0.000924 | -0.000292 | -0.003583 [-0.008827, +0.002035] | +0.000062 [-0.003741, +0.003902] |
| clean_s0 | target | +0.000447 / +0.000441 | +0.001180 / +0.001413 | -0.000287 | +0.000733 [-0.002208, +0.003941] | -0.000733 [-0.001864, +0.000237] |

## Taylor and coordinate diagnostics

Prediction: -0.1 times self/oracle gradient dot product at phi0. Hard clamp, finite updates and detector nonlinearities remain unchanged.

| Detector / step | Predicted mean [CI] | Observed mean [CI] | Pearson [CI] | Sign agreement% [CI] |
|---|---:|---:|---:|---:|
| source / raw_taylor | -0.008168 [-0.012576, -0.004383] | -0.001573 [-0.004721, +0.001690] | 0.445 [0.315, 0.578] | 59.500 [56.417, 62.502] |
| source / matched_taylor | -0.001494 [-0.002176, -0.000873] | -0.001188 [-0.002721, +0.000305] | 0.347 [0.199, 0.477] | 56.250 [53.165, 59.333] |
| target / raw_taylor | -0.002920 [-0.005483, -0.000641] | +0.001474 [-0.001070, +0.004583] | 0.377 [0.267, 0.608] | 81.083 [78.833, 83.333] |
| target / matched_taylor | -0.000534 [-0.000893, -0.000199] | -0.000434 [-0.000920, +0.000038] | 0.620 [0.478, 0.801] | 87.083 [85.083, 89.000] |

Signed mean self-gradient × oracle-gradient products; positive predicts decrease. Cross-model loss units differ. No coordinate removed/reweighted.

| Coordinate | Source raw dot | FCOS raw dot | Source matched dot | FCOS matched dot |
|---|---:|---:|---:|---:|
| gamma | +0.00825676 | +0.00369666 | +0.00211127 | +0.00104817 |
| red_gain | +0.01432041 | +0.00848020 | +0.00218911 | +0.00125309 |
| green_gain | +0.02264708 | +0.00673138 | +0.00325713 | +0.00108613 |
| blue_gain | +0.01245762 | +0.00184613 | +0.00237059 | +0.00029132 |
| contrast | +0.01636288 | +0.00569074 | +0.00367262 | +0.00115522 |
| brightness | +0.00640894 | +0.00239942 | +0.00108894 | +0.00041613 |
| tone | +0.00061673 | +0.00036509 | +0.00012805 | +0.00008945 |
| sharpening | +0.00061323 | -0.00000600 | +0.00012657 | -0.00000218 |

All source mean dots are positive. FCOS sharpening is slightly negative while other means are positive; this small descriptive difference does not establish a coordinate-specific mechanism. Red/green gains and contrast contribute appreciably to the respective positive sums. Raw per-coordinate values remain available.

## Artifacts and recovery

- [results.md](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/results.md)
- [analysis.json](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/analysis.json)
- [samples.jsonl](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/samples.jsonl)
- [metrics.json](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/metrics.json)
- [receipt_audit.json](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/receipt_audit.json)
- [environment.json](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/environment.json)
- [completion.json](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/completion.json)
- [transfer_ap.pdf](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/transfer_ap.pdf)

Remote originals: `/home/liujianhua/wjq/TAISP/runs/20260912-092401-taisp-t006-coco200/artifacts/study`. No active run; await research review/new task. Recovery: T006_handoff.md and project_state.md.

## Complete per-case report
# T006 cross-detector transfer results

Source d7d0510; 200 images, 2800 variant observations, smoke=None.

Source-only frozen Faster R-CNN fixed ROI pseudo-confidence. Target FCOS is analysis/evaluation only. Both detectors evaluate identical enhanced images. Global8D ISP,lr0.1,K3,identity initialization, hard clamp. No target in support/update and no oracle information in norm matching.

## Source / target official AP1 / AP3 (0–100 points)

| Detector | Case | Before AP / AP50 / AP75 | CLIP AP1 /3 | Pseudo AP1 /3 | Pseudo AP50 1 /3 | Pseudo AP75 1 /3 |
|---|---|---:|---:|---:|---:|---:|
| source | gamma_s1 | 38.145 / 59.245 / 39.676 | 38.350 / 38.674 | 37.971 / 38.185 | 58.937 / 59.480 | 39.882 / 40.545 |
| source | gamma_s2 | 36.446 / 57.604 / 39.152 | 36.598 / 36.380 | 36.859 / 36.808 | 57.823 / 57.288 | 39.201 / 39.064 |
| source | contrast_s1 | 36.175 / 57.631 / 37.976 | 36.368 / 36.501 | 36.550 / 36.332 | 57.109 / 57.324 | 38.748 / 38.314 |
| source | contrast_s2 | 31.175 / 49.548 / 32.807 | 31.115 / 31.275 | 31.850 / 32.084 | 50.468 / 51.695 | 33.139 / 33.819 |
| source | color_cast_s1 | 37.642 / 58.493 / 41.783 | 37.668 / 37.722 | 37.423 / 37.409 | 58.011 / 58.051 | 41.159 / 41.384 |
| source | color_cast_s2 | 35.908 / 55.616 / 39.212 | 35.846 / 35.528 | 35.793 / 35.982 | 55.464 / 55.470 | 38.649 / 38.664 |
| source | clean_s0 | 38.012 / 59.424 / 40.742 | 38.139 / 38.213 | 38.390 / 38.322 | 59.457 / 59.755 | 41.360 / 40.928 |
| target | gamma_s1 | 42.233 / 61.389 / 44.866 | 42.080 / 42.279 | 42.119 / 42.548 | 61.425 / 61.948 | 44.701 / 44.822 |
| target | gamma_s2 | 40.219 / 58.698 / 41.879 | 40.553 / 40.256 | 40.662 / 40.954 | 58.836 / 59.190 | 42.755 / 42.366 |
| target | contrast_s1 | 41.892 / 61.002 / 43.654 | 42.300 / 41.853 | 41.909 / 42.070 | 60.948 / 60.877 | 43.708 / 43.932 |
| target | contrast_s2 | 35.307 / 51.443 / 37.148 | 35.297 / 35.191 | 34.894 / 35.033 | 51.859 / 51.777 | 36.751 / 36.646 |
| target | color_cast_s1 | 41.642 / 60.590 / 43.958 | 41.408 / 41.150 | 41.366 / 41.280 | 60.258 / 59.938 | 43.701 / 43.814 |
| target | color_cast_s2 | 39.386 / 57.186 / 42.878 | 39.385 / 39.118 | 39.263 / 39.191 | 57.150 / 57.268 | 42.180 / 41.553 |
| target | clean_s0 | 43.270 / 62.668 / 46.317 | 43.625 / 43.614 | 43.535 / 44.073 | 62.908 / 63.480 | 46.658 / 47.189 |

CLIP AP50/AP75 and all before/CLIP deltas are retained in analysis.json; official aggregate AP has no invented per-image intervals.

## Paired mechanism vs CLIP

All episodes included. *Undefined cosine zero-coded here; valid-only distributions retained separately.95% intervals use2000 paired image-cluster draws.

| Group | Detector | Variant | Cosine* | Positive% | Raw benefit% | Matched benefit% | Delta cosine [CI] | Delta raw benefit pp [CI] | Delta matched benefit pp [CI] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| corrupted_overall | source | global_generic | 0.04531 | 53.83 | 47.33 | 47.33 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| corrupted_overall | source | det_pseudo | 0.11757 | 58.17 | 51.17 | 50.58 | 0.072 [0.009, 0.135] | 3.833 [-0.083, 7.667] | 3.250 [0.331, 6.333] |
| corrupted_overall | target | global_generic | -0.01162 | 48.33 | 48.92 | 48.92 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| corrupted_overall | target | det_pseudo | 0.08159 | 56.75 | 54.33 | 56.17 | 0.093 [0.036, 0.152] | 5.417 [0.583, 10.250] | 7.250 [2.583, 12.000] |
| severity_s1 | source | global_generic | 0.06156 | 56.67 | 49.17 | 49.17 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| severity_s1 | source | det_pseudo | 0.07817 | 55.33 | 53.83 | 49.33 | 0.017 [-0.059, 0.094] | 4.667 [-1.000, 10.000] | 0.167 [-4.500, 4.833] |
| severity_s1 | target | global_generic | -0.01275 | 48.67 | 50.17 | 50.17 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| severity_s1 | target | det_pseudo | 0.08054 | 57.50 | 54.83 | 56.83 | 0.093 [0.023, 0.167] | 4.667 [-1.333, 10.833] | 6.667 [0.333, 13.000] |
| severity_s2 | source | global_generic | 0.02905 | 51.00 | 45.50 | 45.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| severity_s2 | source | det_pseudo | 0.15697 | 61.00 | 48.50 | 51.83 | 0.128 [0.055, 0.203] | 3.000 [-2.333, 8.167] | 6.333 [1.833, 10.671] |
| severity_s2 | target | global_generic | -0.01048 | 48.00 | 47.67 | 47.67 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| severity_s2 | target | det_pseudo | 0.08263 | 56.00 | 53.83 | 55.50 | 0.093 [0.025, 0.156] | 6.167 [0.333, 11.833] | 7.833 [2.333, 13.167] |
| gamma_s1 | source | global_generic | 0.02272 | 53.50 | 48.50 | 48.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| gamma_s1 | source | det_pseudo | 0.09152 | 54.50 | 49.00 | 44.00 | 0.069 [-0.037, 0.174] | 0.500 [-8.500, 9.500] | -4.500 [-12.500, 3.000] |
| gamma_s1 | target | global_generic | -0.06115 | 44.00 | 44.50 | 44.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| gamma_s1 | target | det_pseudo | 0.09756 | 59.50 | 55.50 | 56.00 | 0.159 [0.057, 0.262] | 11.000 [0.500, 21.500] | 11.500 [2.000, 21.500] |
| gamma_s2 | source | global_generic | 0.00171 | 51.00 | 45.50 | 45.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| gamma_s2 | source | det_pseudo | 0.14609 | 62.50 | 46.00 | 50.00 | 0.144 [0.037, 0.256] | 0.500 [-8.000, 9.000] | 4.500 [-2.512, 12.000] |
| gamma_s2 | target | global_generic | -0.01435 | 45.50 | 42.50 | 42.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| gamma_s2 | target | det_pseudo | 0.06411 | 51.50 | 47.00 | 50.50 | 0.078 [-0.027, 0.191] | 4.500 [-4.500, 14.012] | 8.000 [-1.000, 17.500] |
| contrast_s1 | source | global_generic | 0.11737 | 60.50 | 52.50 | 52.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| contrast_s1 | source | det_pseudo | 0.07089 | 55.50 | 59.50 | 51.50 | -0.046 [-0.170, 0.075] | 7.000 [-1.000, 15.500] | -1.000 [-9.000, 6.512] |
| contrast_s1 | target | global_generic | 0.01877 | 51.50 | 53.00 | 53.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| contrast_s1 | target | det_pseudo | 0.08143 | 57.50 | 55.00 | 57.50 | 0.063 [-0.052, 0.175] | 2.000 [-7.012, 12.000] | 4.500 [-4.512, 14.000] |
| contrast_s2 | source | global_generic | 0.02743 | 49.50 | 44.50 | 44.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| contrast_s2 | source | det_pseudo | 0.23024 | 66.00 | 52.50 | 51.50 | 0.203 [0.083, 0.319] | 8.000 [-2.000, 17.500] | 7.000 [-1.500, 15.500] |
| contrast_s2 | target | global_generic | -0.05844 | 46.50 | 47.00 | 47.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| contrast_s2 | target | det_pseudo | 0.14291 | 60.50 | 60.50 | 60.50 | 0.201 [0.082, 0.312] | 13.500 [3.000, 23.500] | 13.500 [3.500, 23.500] |
| color_cast_s1 | source | global_generic | 0.04461 | 56.00 | 46.50 | 46.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| color_cast_s1 | source | det_pseudo | 0.07209 | 56.00 | 53.00 | 52.50 | 0.027 [-0.071, 0.129] | 6.500 [-2.500, 15.012] | 6.000 [-1.500, 13.500] |
| color_cast_s1 | target | global_generic | 0.00413 | 50.50 | 53.00 | 53.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| color_cast_s1 | target | det_pseudo | 0.06264 | 55.50 | 54.00 | 57.00 | 0.059 [-0.045, 0.166] | 1.000 [-8.012, 10.500] | 4.000 [-5.500, 13.000] |
| color_cast_s2 | source | global_generic | 0.05802 | 52.50 | 46.50 | 46.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| color_cast_s2 | source | det_pseudo | 0.09460 | 54.50 | 47.00 | 54.00 | 0.037 [-0.076, 0.153] | 0.500 [-7.500, 9.000] | 7.500 [-0.500, 15.500] |
| color_cast_s2 | target | global_generic | 0.04133 | 52.00 | 53.50 | 53.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| color_cast_s2 | target | det_pseudo | 0.04086 | 56.00 | 54.00 | 55.50 | -0.000 [-0.099, 0.096] | 0.500 [-9.000, 9.500] | 2.000 [-6.512, 10.500] |
| clean_s0 | source | global_generic | 0.04171 | 54.00 | 48.50 | 48.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| clean_s0 | source | det_pseudo | 0.05014 | 54.00 | 51.50 | 52.50 | 0.008 [-0.108, 0.117] | 3.000 [-5.500, 12.000] | 4.000 [-4.000, 12.012] |
| clean_s0 | target | global_generic | -0.04640 | 48.50 | 49.00 | 49.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| clean_s0 | target | det_pseudo | 0.00687 | 50.50 | 51.50 | 53.00 | 0.053 [-0.061, 0.166] | 2.500 [-6.512, 12.000] | 4.000 [-5.500, 13.000] |

## Detector agreement and joint pseudo-step outcomes

| Group | Source-target cosine [CI] | Agreement positive% | Raw both / source-only / target-only / neither% | Matched both / source-only / target-only / neither% |
|---|---:|---:|---:|---:|
| corrupted_overall | 0.366 [0.325, 0.405] | 75.17 | 31.00 / 20.17 / 23.33 / 25.50 | 29.67 / 20.92 / 26.50 / 22.92 |
| severity_s1 | 0.345 [0.293, 0.392] | 74.50 | 32.17 / 21.67 / 22.67 / 23.50 | 28.67 / 20.67 / 28.17 / 22.50 |
| severity_s2 | 0.387 [0.342, 0.431] | 75.83 | 29.83 / 18.67 / 24.00 / 27.50 | 30.67 / 21.17 / 24.83 / 23.33 |
| gamma_s1 | 0.360 [0.287, 0.427] | 76.00 | 30.50 / 18.50 / 25.00 / 26.00 | 25.50 / 18.50 / 30.50 / 25.50 |
| gamma_s2 | 0.402 [0.329, 0.474] | 75.00 | 25.00 / 21.00 / 22.00 / 32.00 | 28.00 / 22.00 / 22.50 / 27.50 |
| contrast_s1 | 0.360 [0.279, 0.436] | 75.00 | 34.50 / 25.00 / 20.50 / 20.00 | 31.00 / 20.50 / 26.50 / 22.00 |
| contrast_s2 | 0.417 [0.344, 0.493] | 76.00 | 37.50 / 15.00 / 23.00 / 24.50 | 33.00 / 18.50 / 27.50 / 21.00 |
| color_cast_s1 | 0.315 [0.242, 0.382] | 72.50 | 31.50 / 21.50 / 22.50 / 24.50 | 29.50 / 23.00 / 27.50 / 20.00 |
| color_cast_s2 | 0.340 [0.272, 0.402] | 76.50 | 27.00 / 20.00 / 27.00 / 26.00 | 31.00 / 23.00 / 24.50 / 21.50 |
| clean_s0 | 0.291 [0.221, 0.361] | 70.50 | 27.00 / 24.50 / 24.50 / 24.00 | 27.50 / 25.00 / 25.50 / 22.00 |

## Severity: macro-average of3 condition AP deltas for pseudo

Descriptive arithmetic means of official condition AP; not pooled detections or new COCO AP.

| Severity | Detector | DeltaAP1 /3 vs before | DeltaAP1 /3 vs CLIP |
|---|---|---:|---:|
| s1 | source | -0.006 / -0.012 | -0.147 / -0.324 |
| s1 | target | -0.124 / +0.044 | -0.131 / +0.205 |
| s2 | source | +0.324 / +0.448 | +0.314 / +0.563 |
| s2 | target | -0.031 / +0.089 | -0.139 / +0.205 |

## Clean control and resource costs

| Variant | Clean phi norm1 /3 | Clean source deltaAP3 | Clean target deltaAP3 | Corrupted adapt / deploy sec3 | Adapt peak MiB | Fallback% |
|---|---:|---:|---:|---:|---:|---:|
| global_generic | 0.01552 / 0.02877 | +0.202 | +0.345 | 0.1143 / 0.1143 | 968.0 | 0.00 |
| det_pseudo | 0.04694 / 0.07801 | +0.310 | +0.804 | 0.1470 / 0.1759 | 1981.9 | 0.33 |

Full signed-coordinate dot products, source-target agreement, Taylor correlations/predictions, raw/matched paired mean-loss intervals, joint-outcome paired intervals, support/confidence, phi trajectories and saturation are in analysis.json. Both models and CLIP are resident during memory measurements; target evaluation is excluded from deployment timing. No family is dropped.

![Pseudo AP3 changes on source and target](remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study/transfer_ap.png)
