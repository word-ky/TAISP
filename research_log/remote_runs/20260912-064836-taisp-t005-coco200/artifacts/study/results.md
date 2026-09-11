# T005 detector-native signal screen

Source 9fb96c1; 200 images, 5600 observations, smoke=None.

Frozen detector, fixed original base/flip ROI support; global8D ISP, lr0.1,K3, JS coefficient1.0. Fresh within-run annotated oracle only for analysis. Clean reported separately. Empty-support episodes retained with exact no update from ISP(phi0); undefined cosine zero-coded only for all-observation summaries and paired contrasts. Valid-only cosine in analysis.json.

## AP1 / AP3 (0–100 subset points)

| Case | Before | CLIP global | Pseudo confidence | Stable confidence | Stable + JS |
|---|---:|---:|---:|---:|---:|
| gamma_s1 | 38.145 | 38.349 / 38.680 | 37.976 / 38.067 | 38.001 / 38.352 | 38.080 / 38.069 |
| gamma_s2 | 36.446 | 36.582 / 36.409 | 36.866 / 36.819 | 36.727 / 36.734 | 36.785 / 36.717 |
| contrast_s1 | 36.175 | 36.369 / 36.498 | 36.546 / 36.307 | 36.819 / 36.473 | 36.844 / 36.430 |
| contrast_s2 | 31.175 | 31.115 / 31.277 | 31.854 / 31.989 | 31.393 / 31.323 | 30.957 / 31.603 |
| color_cast_s1 | 37.642 | 37.669 / 37.723 | 37.397 / 37.333 | 37.138 / 37.071 | 37.330 / 37.325 |
| color_cast_s2 | 35.908 | 35.846 / 35.528 | 35.795 / 35.864 | 36.097 / 36.161 | 36.094 / 36.116 |
| clean_s0 | 38.012 | 38.148 / 38.214 | 38.349 / 38.274 | 38.297 / 38.288 | 38.173 / 38.163 |

## Mechanism and paired95% intervals

Intervals use2000 paired image-cluster draws; no multiplicity adjustment. Frequencies include fallbacks.

| Group | Variant | Mean cosine* | Positive% | Benefit% | Matched benefit% | Δcos vs CLIP [CI] | Δbenefit pp [CI] | Matched Δbenefit pp [CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corrupted_overall | global_generic | 0.04532 | 53.83 | 46.83 | 46.83 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| corrupted_overall | det_pseudo | 0.11757 | 58.17 | 50.25 | 50.58 | 0.072 [0.008, 0.135] | 3.417 [-0.252, 7.167] | 3.750 [0.500, 7.333] |
| corrupted_overall | det_stable | 0.12431 | 57.92 | 50.00 | 50.00 | 0.079 [0.019, 0.139] | 3.167 [-0.667, 6.917] | 3.167 [0.083, 6.333] |
| corrupted_overall | det_stable_js | 0.12271 | 58.17 | 50.17 | 50.50 | 0.077 [0.017, 0.137] | 3.333 [-0.250, 7.000] | 3.667 [0.331, 7.250] |
| gamma_s1 | global_generic | 0.02270 | 53.50 | 48.00 | 48.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| gamma_s1 | det_pseudo | 0.09150 | 54.50 | 47.50 | 44.00 | 0.069 [-0.037, 0.174] | -0.500 [-9.000, 8.012] | -4.000 [-11.500, 4.500] |
| gamma_s1 | det_stable | 0.12901 | 59.00 | 47.50 | 46.50 | 0.106 [-0.003, 0.214] | -0.500 [-9.000, 7.500] | -1.500 [-9.500, 6.500] |
| gamma_s1 | det_stable_js | 0.13203 | 60.50 | 51.50 | 48.50 | 0.109 [0.002, 0.213] | 3.500 [-5.000, 12.000] | 0.500 [-7.500, 8.500] |
| gamma_s2 | global_generic | 0.00175 | 51.00 | 45.00 | 45.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| gamma_s2 | det_pseudo | 0.14611 | 62.50 | 46.50 | 50.00 | 0.144 [0.037, 0.256] | 1.500 [-7.000, 10.000] | 5.000 [-2.500, 13.000] |
| gamma_s2 | det_stable | 0.15110 | 60.00 | 45.50 | 44.50 | 0.149 [0.043, 0.265] | 0.500 [-7.500, 9.000] | -0.500 [-8.500, 7.500] |
| gamma_s2 | det_stable_js | 0.14765 | 59.00 | 47.00 | 47.50 | 0.146 [0.040, 0.264] | 2.000 [-6.000, 10.500] | 2.500 [-5.000, 10.000] |
| contrast_s1 | global_generic | 0.11739 | 60.50 | 53.00 | 53.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| contrast_s1 | det_pseudo | 0.07091 | 55.50 | 57.00 | 55.00 | -0.046 [-0.170, 0.075] | 4.000 [-4.500, 12.500] | 2.000 [-6.000, 10.000] |
| contrast_s1 | det_stable | 0.08109 | 53.50 | 50.00 | 56.50 | -0.036 [-0.150, 0.078] | -3.000 [-11.000, 5.000] | 3.500 [-3.500, 11.000] |
| contrast_s1 | det_stable_js | 0.07658 | 55.00 | 51.00 | 53.50 | -0.041 [-0.156, 0.073] | -2.000 [-10.012, 6.000] | 0.500 [-6.500, 8.500] |
| contrast_s2 | global_generic | 0.02743 | 49.50 | 42.00 | 42.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| contrast_s2 | det_pseudo | 0.23022 | 66.00 | 54.00 | 51.00 | 0.203 [0.083, 0.319] | 12.000 [2.000, 21.000] | 9.000 [0.500, 17.500] |
| contrast_s2 | det_stable | 0.19816 | 62.50 | 50.50 | 54.00 | 0.171 [0.059, 0.285] | 8.500 [-0.512, 17.500] | 12.000 [3.500, 20.500] |
| contrast_s2 | det_stable_js | 0.18981 | 62.00 | 48.50 | 51.00 | 0.162 [0.051, 0.275] | 6.500 [-2.512, 15.500] | 9.000 [0.000, 18.000] |
| color_cast_s1 | global_generic | 0.04463 | 56.00 | 44.50 | 44.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| color_cast_s1 | det_pseudo | 0.07210 | 56.00 | 48.00 | 50.50 | 0.027 [-0.071, 0.129] | 3.500 [-6.000, 12.500] | 6.000 [-2.500, 14.000] |
| color_cast_s1 | det_stable | 0.05572 | 56.00 | 57.00 | 50.00 | 0.011 [-0.099, 0.122] | 12.500 [4.000, 21.000] | 5.500 [-3.000, 14.000] |
| color_cast_s1 | det_stable_js | 0.05433 | 55.00 | 51.00 | 50.50 | 0.010 [-0.101, 0.118] | 6.500 [-2.500, 15.000] | 6.000 [-2.500, 14.500] |
| color_cast_s2 | global_generic | 0.05803 | 52.50 | 48.50 | 48.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| color_cast_s2 | det_pseudo | 0.09456 | 54.50 | 48.50 | 53.00 | 0.037 [-0.076, 0.153] | 0.000 [-8.000, 8.500] | 4.500 [-3.500, 12.500] |
| color_cast_s2 | det_stable | 0.13078 | 56.50 | 49.50 | 48.50 | 0.073 [-0.036, 0.178] | 1.000 [-7.500, 9.500] | 0.000 [-7.500, 8.000] |
| color_cast_s2 | det_stable_js | 0.13582 | 57.50 | 52.00 | 52.00 | 0.078 [-0.031, 0.183] | 3.500 [-5.500, 12.000] | 3.500 [-5.000, 11.512] |
| clean_s0 | global_generic | 0.04172 | 54.00 | 50.00 | 50.00 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| clean_s0 | det_pseudo | 0.05014 | 54.00 | 53.00 | 54.50 | 0.008 [-0.108, 0.117] | 3.000 [-6.000, 12.000] | 4.500 [-4.000, 13.000] |
| clean_s0 | det_stable | 0.05323 | 53.50 | 51.00 | 52.00 | 0.012 [-0.106, 0.123] | 1.000 [-7.012, 10.000] | 2.000 [-7.000, 11.000] |
| clean_s0 | det_stable_js | 0.05460 | 54.00 | 53.50 | 56.00 | 0.013 [-0.105, 0.125] | 3.500 [-5.000, 12.500] | 6.000 [-1.500, 14.500] |

*Undefined zero-gradient cosine contributes0 here; valid-only distributions and exact counts retained.

## Clean control and deployment costs

| Variant | Clean ΔAP1 / ΔAP3 | Clean mean phi norm1 /3 | Corrupted adapt / deploy seconds3 | Adapt peak MiB |
|---|---:|---:|---:|---:|
| global_generic | +0.136 / +0.202 | 0.01552 / 0.02877 | 0.1136 / 0.1136 | 841.7 |
| det_pseudo | +0.338 / +0.262 | 0.04694 / 0.07799 | 0.1444 / 0.1729 | 1855.2 |
| det_stable | +0.285 / +0.276 | 0.03955 / 0.06956 | 0.2806 / 0.3398 | 2653.7 |
| det_stable_js | +0.162 / +0.151 | 0.04200 / 0.07306 | 0.2795 / 0.3387 | 2653.7 |

Setup peaks are separately retained in analysis.json; timings include unchanged diagnostics, fixed variant order.

## Support and fallback

| Group | Variant | Base / stable count | Stable/base | Symmetric match | Fallback% |
|---|---|---:|---:|---:|---:|
| corrupted_overall | det_pseudo | 7.16 / 5.95 | 0.848 | 0.833 | 0.33 |
| corrupted_overall | det_stable | 7.16 / 5.95 | 0.848 | 0.833 | 0.50 |
| corrupted_overall | det_stable_js | 7.16 / 5.95 | 0.848 | 0.833 | 0.50 |
| gamma_s1 | det_pseudo | 7.63 / 6.39 | 0.849 | 0.838 | 0.00 |
| gamma_s1 | det_stable | 7.63 / 6.39 | 0.849 | 0.838 | 0.50 |
| gamma_s1 | det_stable_js | 7.63 / 6.39 | 0.849 | 0.838 | 0.50 |
| gamma_s2 | det_pseudo | 7.37 / 6.06 | 0.842 | 0.827 | 0.00 |
| gamma_s2 | det_stable | 7.37 / 6.06 | 0.842 | 0.827 | 0.00 |
| gamma_s2 | det_stable_js | 7.37 / 6.06 | 0.842 | 0.827 | 0.00 |
| contrast_s1 | det_pseudo | 7.16 / 5.96 | 0.859 | 0.845 | 0.00 |
| contrast_s1 | det_stable | 7.16 / 5.96 | 0.859 | 0.845 | 0.00 |
| contrast_s1 | det_stable_js | 7.16 / 5.96 | 0.859 | 0.845 | 0.00 |
| contrast_s2 | det_pseudo | 5.83 / 4.82 | 0.842 | 0.819 | 1.00 |
| contrast_s2 | det_stable | 5.83 / 4.82 | 0.842 | 0.819 | 1.50 |
| contrast_s2 | det_stable_js | 5.83 / 4.82 | 0.842 | 0.819 | 1.50 |
| color_cast_s1 | det_pseudo | 7.62 / 6.38 | 0.855 | 0.842 | 0.50 |
| color_cast_s1 | det_stable | 7.62 / 6.38 | 0.855 | 0.842 | 0.50 |
| color_cast_s1 | det_stable_js | 7.62 / 6.38 | 0.855 | 0.842 | 0.50 |
| color_cast_s2 | det_pseudo | 7.37 / 6.07 | 0.839 | 0.829 | 0.50 |
| color_cast_s2 | det_stable | 7.37 / 6.07 | 0.839 | 0.829 | 0.50 |
| color_cast_s2 | det_stable_js | 7.37 / 6.07 | 0.839 | 0.829 | 0.50 |
| clean_s0 | det_pseudo | 7.76 / 6.53 | 0.860 | 0.848 | 0.00 |
| clean_s0 | det_stable | 7.76 / 6.53 | 0.860 | 0.848 | 0.00 |
| clean_s0 | det_stable_js | 7.76 / 6.53 | 0.860 | 0.848 | 0.00 |

## Support-size outcomes

| Group | Variant | Support | N | Mean cosine* | Benefit% | Matched benefit% | Mean loss delta1 |
|---|---|---|---:|---:|---:|---:|---:|
| corrupted_overall | det_pseudo | 0 | 4 | 0.00000 | 0.00 | 0.00 | 0 |
| corrupted_overall | det_pseudo | 1-2 | 234 | 0.10989 | 48.29 | 49.15 | -0.000303816 |
| corrupted_overall | det_pseudo | 3-5 | 416 | 0.08095 | 52.40 | 47.84 | -0.000689302 |
| corrupted_overall | det_pseudo | 6+ | 546 | 0.14962 | 49.82 | 53.66 | -0.00276656 |
| corrupted_overall | det_stable | 0 | 6 | 0.00000 | 0.00 | 0.00 | 0 |
| corrupted_overall | det_stable | 1-2 | 330 | 0.15170 | 49.09 | 49.70 | 0.000373716 |
| corrupted_overall | det_stable | 3-5 | 424 | 0.10193 | 49.29 | 49.53 | -0.0024454 |
| corrupted_overall | det_stable | 6+ | 440 | 0.12703 | 52.05 | 51.36 | -0.00243619 |
| corrupted_overall | det_stable_js | 0 | 6 | 0.00000 | 0.00 | 0.00 | 0 |
| corrupted_overall | det_stable_js | 1-2 | 330 | 0.14733 | 50.91 | 52.73 | 0.000431197 |
| corrupted_overall | det_stable_js | 3-5 | 424 | 0.10050 | 49.76 | 49.53 | -0.00284361 |
| corrupted_overall | det_stable_js | 6+ | 440 | 0.12730 | 50.68 | 50.45 | -0.00288382 |
| gamma_s1 | det_pseudo | 1-2 | 35 | 0.09075 | 42.86 | 37.14 | 0.00170497 |
| gamma_s1 | det_pseudo | 3-5 | 72 | 0.05723 | 44.44 | 45.83 | 0.00405517 |
| gamma_s1 | det_pseudo | 6+ | 93 | 0.11831 | 51.61 | 45.16 | 0.00113035 |
| gamma_s1 | det_stable | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| gamma_s1 | det_stable | 1-2 | 53 | 0.09247 | 41.51 | 52.83 | 0.000550562 |
| gamma_s1 | det_stable | 3-5 | 68 | 0.13927 | 45.59 | 42.65 | -0.0030384 |
| gamma_s1 | det_stable | 6+ | 78 | 0.14654 | 53.85 | 46.15 | -0.00047286 |
| gamma_s1 | det_stable_js | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| gamma_s1 | det_stable_js | 1-2 | 53 | 0.09516 | 56.60 | 52.83 | -0.000142136 |
| gamma_s1 | det_stable_js | 3-5 | 68 | 0.13994 | 51.47 | 44.12 | -0.00545603 |
| gamma_s1 | det_stable_js | 6+ | 78 | 0.15189 | 48.72 | 50.00 | 0.00107298 |
| gamma_s2 | det_pseudo | 1-2 | 35 | 0.04355 | 51.43 | 48.57 | -0.000353125 |
| gamma_s2 | det_pseudo | 3-5 | 74 | 0.08757 | 50.00 | 48.65 | -0.00622704 |
| gamma_s2 | det_pseudo | 6+ | 91 | 0.23316 | 41.76 | 51.65 | 0.00263944 |
| gamma_s2 | det_stable | 1-2 | 52 | 0.15150 | 46.15 | 50.00 | 0.00104254 |
| gamma_s2 | det_stable | 3-5 | 74 | 0.11262 | 45.95 | 41.89 | 0.00605546 |
| gamma_s2 | det_stable | 6+ | 74 | 0.18931 | 44.59 | 43.24 | 0.00600391 |
| gamma_s2 | det_stable_js | 1-2 | 52 | 0.15892 | 48.08 | 53.85 | 0.000390999 |
| gamma_s2 | det_stable_js | 3-5 | 74 | 0.10609 | 47.30 | 50.00 | 0.00229017 |
| gamma_s2 | det_stable_js | 6+ | 74 | 0.18130 | 45.95 | 40.54 | 0.00973408 |
| contrast_s1 | det_pseudo | 1-2 | 43 | 0.05063 | 46.51 | 44.19 | -0.00277628 |
| contrast_s1 | det_pseudo | 3-5 | 62 | 0.01033 | 53.23 | 48.39 | 0.00199149 |
| contrast_s1 | det_pseudo | 6+ | 95 | 0.11963 | 64.21 | 64.21 | -0.00837618 |
| contrast_s1 | det_stable | 1-2 | 56 | 0.17097 | 50.00 | 53.57 | -0.00273593 |
| contrast_s1 | det_stable | 3-5 | 65 | 0.01703 | 36.92 | 50.77 | 0.00491583 |
| contrast_s1 | det_stable | 6+ | 79 | 0.07009 | 60.76 | 63.29 | -0.0115861 |
| contrast_s1 | det_stable_js | 1-2 | 56 | 0.15720 | 44.64 | 50.00 | -0.00243456 |
| contrast_s1 | det_stable_js | 3-5 | 65 | 0.02336 | 41.54 | 44.62 | 0.0041036 |
| contrast_s1 | det_stable_js | 6+ | 79 | 0.06322 | 63.29 | 63.29 | -0.0141197 |
| contrast_s2 | det_pseudo | 0 | 2 | 0.00000 | 0.00 | 0.00 | 0 |
| contrast_s2 | det_pseudo | 1-2 | 53 | 0.09107 | 49.06 | 41.51 | 0.00106849 |
| contrast_s2 | det_pseudo | 3-5 | 76 | 0.28995 | 57.89 | 50.00 | 0.00582377 |
| contrast_s2 | det_pseudo | 6+ | 69 | 0.27798 | 55.07 | 60.87 | -0.0238458 |
| contrast_s2 | det_stable | 0 | 3 | 0.00000 | 0.00 | 0.00 | 0 |
| contrast_s2 | det_stable | 1-2 | 72 | 0.04853 | 41.67 | 44.44 | 0.00682158 |
| contrast_s2 | det_stable | 3-5 | 70 | 0.29709 | 61.43 | 61.43 | -0.0113505 |
| contrast_s2 | det_stable | 6+ | 55 | 0.27894 | 50.91 | 60.00 | -0.0125705 |
| contrast_s2 | det_stable_js | 0 | 3 | 0.00000 | 0.00 | 0.00 | 0 |
| contrast_s2 | det_stable_js | 1-2 | 72 | 0.03499 | 44.44 | 43.06 | 0.00744582 |
| contrast_s2 | det_stable_js | 3-5 | 70 | 0.29874 | 55.71 | 67.14 | -0.00804407 |
| contrast_s2 | det_stable_js | 6+ | 55 | 0.26420 | 47.27 | 43.64 | -0.0109757 |
| color_cast_s1 | det_pseudo | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| color_cast_s1 | det_pseudo | 1-2 | 34 | 0.11126 | 52.94 | 61.76 | 0.00191783 |
| color_cast_s1 | det_pseudo | 3-5 | 62 | 0.03010 | 59.68 | 45.16 | -0.00705667 |
| color_cast_s1 | det_pseudo | 6+ | 103 | 0.08515 | 39.81 | 50.49 | 0.0035037 |
| color_cast_s1 | det_stable | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| color_cast_s1 | det_stable | 1-2 | 46 | 0.21042 | 63.04 | 50.00 | -0.00199762 |
| color_cast_s1 | det_stable | 3-5 | 72 | -0.05483 | 52.78 | 51.39 | -0.00176025 |
| color_cast_s1 | det_stable | 6+ | 81 | 0.06683 | 58.02 | 49.38 | 0.0001285 |
| color_cast_s1 | det_stable_js | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| color_cast_s1 | det_stable_js | 1-2 | 46 | 0.20986 | 58.70 | 56.52 | -0.00219523 |
| color_cast_s1 | det_stable_js | 3-5 | 72 | -0.06193 | 48.61 | 44.44 | 0.000646522 |
| color_cast_s1 | det_stable_js | 6+ | 81 | 0.07002 | 49.38 | 53.09 | -0.00115675 |
| color_cast_s2 | det_pseudo | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| color_cast_s2 | det_pseudo | 1-2 | 34 | 0.30083 | 47.06 | 67.65 | -0.00355481 |
| color_cast_s2 | det_pseudo | 3-5 | 70 | -0.02100 | 50.00 | 48.57 | -0.00352123 |
| color_cast_s2 | det_pseudo | 6+ | 95 | 0.10688 | 48.42 | 51.58 | 0.00236173 |
| color_cast_s2 | det_stable | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| color_cast_s2 | det_stable | 1-2 | 51 | 0.28498 | 56.86 | 49.02 | -0.00404151 |
| color_cast_s2 | det_stable | 3-5 | 75 | 0.09945 | 52.00 | 49.33 | -0.00902133 |
| color_cast_s2 | det_stable | 6+ | 73 | 0.05704 | 42.47 | 47.95 | 0.00160187 |
| color_cast_s2 | det_stable_js | 0 | 1 | 0.00000 | 0.00 | 0.00 | 0 |
| color_cast_s2 | det_stable_js | 1-2 | 51 | 0.28112 | 56.86 | 64.71 | -0.00331934 |
| color_cast_s2 | det_stable_js | 3-5 | 75 | 0.09701 | 53.33 | 46.67 | -0.010058 |
| color_cast_s2 | det_stable_js | 6+ | 73 | 0.07605 | 47.95 | 49.32 | -0.00356275 |
| clean_s0 | det_pseudo | 1-2 | 35 | 0.04579 | 51.43 | 60.00 | -0.00411297 |
| clean_s0 | det_pseudo | 3-5 | 66 | -0.05319 | 48.48 | 56.06 | -0.00195875 |
| clean_s0 | det_pseudo | 6+ | 99 | 0.12056 | 56.57 | 51.52 | -0.00459064 |
| clean_s0 | det_stable | 1-2 | 52 | 0.04400 | 50.00 | 59.62 | -0.00420466 |
| clean_s0 | det_stable | 3-5 | 65 | -0.03149 | 47.69 | 46.15 | -0.00127444 |
| clean_s0 | det_stable | 6+ | 83 | 0.12536 | 54.22 | 51.81 | -0.00607141 |
| clean_s0 | det_stable_js | 1-2 | 52 | 0.05156 | 48.08 | 59.62 | -0.00271563 |
| clean_s0 | det_stable_js | 3-5 | 65 | -0.03367 | 47.69 | 49.23 | -1.20741e-05 |
| clean_s0 | det_stable_js | 6+ | 83 | 0.12564 | 61.45 | 59.04 | -0.0111895 |

Full per-case gradient norms/energy, confidence distributions, CE/JS components, Taylor predictions/correlations, mean-loss paired CIs, stable-minus-pseudo and JS-minus-stable paired effects, saturation and phi trajectories are in analysis.json. Negative families are retained.

![Subset AP change at3steps](ap_change.png)
