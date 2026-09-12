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

![Pseudo AP3 changes on source and target](transfer_ap.png)
