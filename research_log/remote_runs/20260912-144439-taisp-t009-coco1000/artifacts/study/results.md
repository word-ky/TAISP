# T009 fixed-method external validation

Images: 1000; variant rows: 21000. K=3 only. No AP confidence intervals.

## Independent-target replication

| Detector | Reference | Aggregate macro AP delta | Block deltas (AP points) | Positive blocks |
| --- | --- | ---: | --- | ---: |
| target | no_adapt | +0.0703 | block_1: +0.0935 / block_2: +0.1621 / block_3: -0.0508 / block_4: +0.0982 / block_5: +0.0807 | 4/5 |
| target | det_pseudo | +0.0892 | block_1: -0.0478 / block_2: +0.1579 / block_3: +0.0023 / block_4: +0.2105 / block_5: -0.0746 | 3/5 |
| ssd | no_adapt | +0.0322 | block_1: -0.0046 / block_2: +0.0554 / block_3: +0.0839 / block_4: +0.0316 / block_5: +0.0244 | 4/5 |
| ssd | det_pseudo | +0.0331 | block_1: -0.1290 / block_2: -0.1360 / block_3: +0.1345 / block_4: +0.1530 / block_5: +0.0633 | 3/5 |

Full 1,000-image/five-block coverage: True. External AP criterion: True. Clean outcomes still require review.

## Aggregate AP / AP50 / AP75

| Detector | Condition | Method | AP / AP50 / AP75 | AP delta no-adapt / CLIP / raw |
| --- | --- | --- | ---: | ---: |
| source | gamma_s1 | no_adapt | 36.893 / 58.460 / 40.203 | +0.000 / +0.161 / -0.128 |
| source | gamma_s2 | no_adapt | 35.296 / 56.239 / 37.034 | +0.000 / +0.416 / +0.098 |
| source | contrast_s1 | no_adapt | 36.010 / 57.102 / 38.897 | +0.000 / -0.096 / +0.088 |
| source | contrast_s2 | no_adapt | 31.148 / 50.386 / 32.872 | +0.000 / +0.120 / -0.016 |
| source | color_cast_s1 | no_adapt | 36.533 / 58.208 / 39.990 | +0.000 / -0.114 / -0.134 |
| source | color_cast_s2 | no_adapt | 35.658 / 56.848 / 37.999 | +0.000 / -0.019 / +0.648 |
| source | clean_s0 | no_adapt | 37.578 / 59.431 / 40.282 | +0.000 / +0.076 / +0.249 |
| source | gamma_s1 | global_generic | 36.732 / 58.129 / 39.945 | -0.161 / +0.000 / -0.289 |
| source | gamma_s2 | global_generic | 34.881 / 55.892 / 36.535 | -0.416 / +0.000 / -0.318 |
| source | contrast_s1 | global_generic | 36.106 / 57.132 / 38.962 | +0.096 / +0.000 / +0.184 |
| source | contrast_s2 | global_generic | 31.029 / 50.313 / 32.772 | -0.120 / +0.000 / -0.135 |
| source | color_cast_s1 | global_generic | 36.648 / 58.170 / 39.850 | +0.114 / +0.000 / -0.019 |
| source | color_cast_s2 | global_generic | 35.677 / 56.822 / 38.053 | +0.019 / +0.000 / +0.667 |
| source | clean_s0 | global_generic | 37.502 / 59.293 / 40.285 | -0.076 / +0.000 / +0.173 |
| source | gamma_s1 | det_pseudo | 37.021 / 58.434 / 40.281 | +0.128 / +0.289 / +0.000 |
| source | gamma_s2 | det_pseudo | 35.199 / 55.987 / 38.142 | -0.098 / +0.318 / +0.000 |
| source | contrast_s1 | det_pseudo | 35.922 / 56.920 / 38.716 | -0.088 / -0.184 / +0.000 |
| source | contrast_s2 | det_pseudo | 31.164 / 50.451 / 32.926 | +0.016 / +0.135 / +0.000 |
| source | color_cast_s1 | det_pseudo | 36.667 / 58.004 / 39.895 | +0.134 / +0.019 / +0.000 |
| source | color_cast_s2 | det_pseudo | 35.010 / 55.950 / 37.125 | -0.648 / -0.667 / +0.000 |
| source | clean_s0 | det_pseudo | 37.329 / 59.003 / 40.081 | -0.249 / -0.173 / +0.000 |
| source | gamma_s1 | det_pseudo_clip_radius | 37.118 / 58.641 / 40.385 | +0.225 / +0.386 / +0.097 |
| source | gamma_s2 | det_pseudo_clip_radius | 35.278 / 56.183 / 37.928 | -0.018 / +0.397 / +0.080 |
| source | contrast_s1 | det_pseudo_clip_radius | 35.978 / 57.168 / 38.960 | -0.032 / -0.128 / +0.056 |
| source | contrast_s2 | det_pseudo_clip_radius | 31.261 / 50.645 / 33.158 | +0.113 / +0.233 / +0.097 |
| source | color_cast_s1 | det_pseudo_clip_radius | 36.561 / 58.271 / 39.750 | +0.027 / -0.087 / -0.106 |
| source | color_cast_s2 | det_pseudo_clip_radius | 35.747 / 56.977 / 38.178 | +0.089 / +0.069 / +0.737 |
| source | clean_s0 | det_pseudo_clip_radius | 37.560 / 59.330 / 40.222 | -0.018 / +0.058 / +0.231 |
| target | gamma_s1 | no_adapt | 38.193 / 57.489 / 39.484 | +0.000 / +0.077 / +0.016 |
| target | gamma_s2 | no_adapt | 36.554 / 55.337 / 37.671 | +0.000 / +0.374 / +0.238 |
| target | contrast_s1 | no_adapt | 37.612 / 56.510 / 38.420 | +0.000 / +0.030 / -0.111 |
| target | contrast_s2 | no_adapt | 33.101 / 50.168 / 33.595 | +0.000 / -0.025 / -0.130 |
| target | color_cast_s1 | no_adapt | 38.152 / 57.136 / 40.417 | +0.000 / -0.037 / -0.039 |
| target | color_cast_s2 | no_adapt | 36.873 / 55.313 / 38.923 | +0.000 / -0.058 / +0.139 |
| target | clean_s0 | no_adapt | 39.316 / 58.771 / 41.725 | +0.000 / +0.022 / +0.108 |
| target | gamma_s1 | global_generic | 38.116 / 57.387 / 39.365 | -0.077 / +0.000 / -0.061 |
| target | gamma_s2 | global_generic | 36.180 / 54.995 / 37.310 | -0.374 / +0.000 / -0.136 |
| target | contrast_s1 | global_generic | 37.582 / 56.458 / 38.535 | -0.030 / +0.000 / -0.141 |
| target | contrast_s2 | global_generic | 33.126 / 50.054 / 33.857 | +0.025 / +0.000 / -0.105 |
| target | color_cast_s1 | global_generic | 38.189 / 57.215 / 40.360 | +0.037 / +0.000 / -0.001 |
| target | color_cast_s2 | global_generic | 36.931 / 55.389 / 38.946 | +0.058 / +0.000 / +0.196 |
| target | clean_s0 | global_generic | 39.294 / 58.747 / 41.584 | -0.022 / +0.000 / +0.086 |
| target | gamma_s1 | det_pseudo | 38.177 / 57.733 / 39.191 | -0.016 / +0.061 / +0.000 |
| target | gamma_s2 | det_pseudo | 36.316 / 55.159 / 37.205 | -0.238 / +0.136 / +0.000 |
| target | contrast_s1 | det_pseudo | 37.723 / 56.463 / 38.603 | +0.111 / +0.141 / +0.000 |
| target | contrast_s2 | det_pseudo | 33.230 / 50.292 / 33.831 | +0.130 / +0.105 / +0.000 |
| target | color_cast_s1 | det_pseudo | 38.191 / 57.221 / 40.333 | +0.039 / +0.001 / +0.000 |
| target | color_cast_s2 | det_pseudo | 36.735 / 55.233 / 38.864 | -0.139 / -0.196 / +0.000 |
| target | clean_s0 | det_pseudo | 39.208 / 58.460 / 41.784 | -0.108 / -0.086 / +0.000 |
| target | gamma_s1 | det_pseudo_clip_radius | 38.283 / 57.548 / 39.526 | +0.090 / +0.167 / +0.106 |
| target | gamma_s2 | det_pseudo_clip_radius | 36.449 / 55.367 / 37.420 | -0.105 / +0.269 / +0.133 |
| target | contrast_s1 | det_pseudo_clip_radius | 37.730 / 56.506 / 38.708 | +0.118 / +0.149 / +0.008 |
| target | contrast_s2 | det_pseudo_clip_radius | 33.296 / 50.401 / 33.970 | +0.195 / +0.170 / +0.066 |
| target | color_cast_s1 | det_pseudo_clip_radius | 38.185 / 57.219 / 40.326 | +0.033 / -0.004 / -0.005 |
| target | color_cast_s2 | det_pseudo_clip_radius | 36.964 / 55.381 / 39.040 | +0.090 / +0.032 / +0.229 |
| target | clean_s0 | det_pseudo_clip_radius | 39.377 / 58.811 / 41.972 | +0.061 / +0.083 / +0.169 |
| ssd | gamma_s1 | no_adapt | 24.178 / 40.364 / 25.066 | +0.000 / +0.069 / +0.027 |
| ssd | gamma_s2 | no_adapt | 23.258 / 39.142 / 23.839 | +0.000 / -0.071 / -0.192 |
| ssd | contrast_s1 | no_adapt | 24.673 / 41.404 / 25.244 | +0.000 / +0.018 / +0.012 |
| ssd | contrast_s2 | no_adapt | 22.892 / 38.827 / 22.643 | +0.000 / -0.048 / +0.162 |
| ssd | color_cast_s1 | no_adapt | 24.239 / 40.693 / 24.562 | +0.000 / -0.024 / -0.036 |
| ssd | color_cast_s2 | no_adapt | 23.668 / 39.686 / 24.038 | +0.000 / +0.037 / +0.032 |
| ssd | clean_s0 | no_adapt | 24.836 / 41.581 / 25.589 | +0.000 / -0.015 / +0.024 |
| ssd | gamma_s1 | global_generic | 24.109 / 40.264 / 25.196 | -0.069 / +0.000 / -0.042 |
| ssd | gamma_s2 | global_generic | 23.329 / 39.332 / 23.888 | +0.071 / +0.000 / -0.121 |
| ssd | contrast_s1 | global_generic | 24.655 / 41.386 / 25.090 | -0.018 / +0.000 / -0.006 |
| ssd | contrast_s2 | global_generic | 22.941 / 38.849 / 22.872 | +0.048 / +0.000 / +0.211 |
| ssd | color_cast_s1 | global_generic | 24.263 / 40.701 / 24.529 | +0.024 / +0.000 / -0.012 |
| ssd | color_cast_s2 | global_generic | 23.630 / 39.684 / 23.963 | -0.037 / +0.000 / -0.006 |
| ssd | clean_s0 | global_generic | 24.851 / 41.487 / 25.679 | +0.015 / +0.000 / +0.039 |
| ssd | gamma_s1 | det_pseudo | 24.152 / 40.613 / 25.085 | -0.027 / +0.042 / +0.000 |
| ssd | gamma_s2 | det_pseudo | 23.450 / 39.345 / 23.980 | +0.192 / +0.121 / +0.000 |
| ssd | contrast_s1 | det_pseudo | 24.661 / 41.430 / 25.249 | -0.012 / +0.006 / +0.000 |
| ssd | contrast_s2 | det_pseudo | 22.730 / 38.309 / 22.860 | -0.162 / -0.211 / +0.000 |
| ssd | color_cast_s1 | det_pseudo | 24.274 / 40.762 / 24.592 | +0.036 / +0.012 / +0.000 |
| ssd | color_cast_s2 | det_pseudo | 23.636 / 39.760 / 23.916 | -0.032 / +0.006 / +0.000 |
| ssd | clean_s0 | det_pseudo | 24.812 / 41.621 / 25.608 | -0.024 / -0.039 / +0.000 |
| ssd | gamma_s1 | det_pseudo_clip_radius | 24.180 / 40.447 / 25.124 | +0.001 / +0.070 / +0.028 |
| ssd | gamma_s2 | det_pseudo_clip_radius | 23.376 / 39.360 / 23.779 | +0.118 / +0.047 / -0.074 |
| ssd | contrast_s1 | det_pseudo_clip_radius | 24.722 / 41.463 / 25.240 | +0.049 / +0.067 / +0.060 |
| ssd | contrast_s2 | det_pseudo_clip_radius | 22.916 / 38.939 / 22.398 | +0.024 / -0.024 / +0.186 |
| ssd | color_cast_s1 | det_pseudo_clip_radius | 24.287 / 40.726 / 24.616 | +0.048 / +0.024 / +0.012 |
| ssd | color_cast_s2 | det_pseudo_clip_radius | 23.621 / 39.655 / 23.952 | -0.047 / -0.010 / -0.015 |
| ssd | clean_s0 | det_pseudo_clip_radius | 24.832 / 41.529 / 25.582 | -0.004 / -0.019 / +0.020 |

## Every method and replication block: macro AP deltas

| Group | Detector | Method | Macro delta no-adapt / CLIP / raw | Positive conditions vs no-adapt / raw | Clean delta vs no-adapt |
| --- | --- | --- | ---: | ---: | ---: |
| aggregate | source | no_adapt | +0.0000 / +0.0778 / +0.0929 | 0/6 / 3/6 | +0.0000 |
| aggregate | source | global_generic | -0.0778 / +0.0000 / +0.0150 | 3/6 / 2/6 | -0.0756 |
| aggregate | source | det_pseudo | -0.0929 / -0.0150 / +0.0000 | 3/6 / 0/6 | -0.2491 |
| aggregate | source | det_pseudo_clip_radius | +0.0672 / +0.1450 / +0.1600 | 4/6 / 5/6 | -0.0179 |
| aggregate | target | no_adapt | +0.0000 / +0.0603 / +0.0189 | 0/6 / 3/6 | +0.0000 |
| aggregate | target | global_generic | -0.0603 / +0.0000 / -0.0413 | 3/6 / 1/6 | -0.0219 |
| aggregate | target | det_pseudo | -0.0189 / +0.0413 / +0.0000 | 3/6 / 0/6 | -0.1076 |
| aggregate | target | det_pseudo_clip_radius | +0.0703 / +0.1305 / +0.0892 | 5/6 / 5/6 | +0.0609 |
| aggregate | ssd | no_adapt | +0.0000 / -0.0031 / +0.0009 | 0/6 / 4/6 | +0.0000 |
| aggregate | ssd | global_generic | +0.0031 / +0.0000 / +0.0041 | 3/6 / 1/6 | +0.0151 |
| aggregate | ssd | det_pseudo | -0.0009 / -0.0041 / +0.0000 | 2/6 / 0/6 | -0.0239 |
| aggregate | ssd | det_pseudo_clip_radius | +0.0322 / +0.0290 / +0.0331 | 5/6 / 4/6 | -0.0041 |
| block_1 | source | no_adapt | +0.0000 / +0.0418 / +0.0798 | 0/6 / 3/6 | +0.0000 |
| block_1 | source | global_generic | -0.0418 / +0.0000 / +0.0380 | 2/6 / 4/6 | +0.0541 |
| block_1 | source | det_pseudo | -0.0798 / -0.0380 / +0.0000 | 3/6 / 0/6 | -0.2071 |
| block_1 | source | det_pseudo_clip_radius | -0.0543 / -0.0125 / +0.0255 | 1/6 / 3/6 | -0.1937 |
| block_1 | target | no_adapt | +0.0000 / +0.0044 / -0.1413 | 0/6 / 3/6 | +0.0000 |
| block_1 | target | global_generic | -0.0044 / +0.0000 / -0.1458 | 2/6 / 3/6 | -0.0568 |
| block_1 | target | det_pseudo | +0.1413 / +0.1458 / +0.0000 | 3/6 / 0/6 | -0.0957 |
| block_1 | target | det_pseudo_clip_radius | +0.0935 / +0.0980 / -0.0478 | 3/6 / 3/6 | +0.0676 |
| block_1 | ssd | no_adapt | +0.0000 / -0.0198 / -0.1244 | 0/6 / 1/6 | +0.0000 |
| block_1 | ssd | global_generic | +0.0198 / +0.0000 / -0.1046 | 2/6 / 2/6 | +0.0277 |
| block_1 | ssd | det_pseudo | +0.1244 / +0.1046 / +0.0000 | 5/6 / 0/6 | -0.1122 |
| block_1 | ssd | det_pseudo_clip_radius | -0.0046 / -0.0244 / -0.1290 | 2/6 / 2/6 | -0.0023 |
| block_2 | source | no_adapt | +0.0000 / +0.1024 / +0.0205 | 0/6 / 4/6 | +0.0000 |
| block_2 | source | global_generic | -0.1024 / +0.0000 / -0.0819 | 2/6 / 2/6 | -0.2134 |
| block_2 | source | det_pseudo | -0.0205 / +0.0819 / +0.0000 | 2/6 / 0/6 | -0.4991 |
| block_2 | source | det_pseudo_clip_radius | +0.0782 / +0.1807 / +0.0988 | 5/6 / 3/6 | -0.3379 |
| block_2 | target | no_adapt | +0.0000 / -0.0537 / -0.0041 | 0/6 / 3/6 | +0.0000 |
| block_2 | target | global_generic | +0.0537 / +0.0000 / +0.0496 | 3/6 / 3/6 | -0.1914 |
| block_2 | target | det_pseudo | +0.0041 / -0.0496 / +0.0000 | 3/6 / 0/6 | -0.2759 |
| block_2 | target | det_pseudo_clip_radius | +0.1621 / +0.1083 / +0.1579 | 4/6 / 4/6 | -0.1850 |
| block_2 | ssd | no_adapt | +0.0000 / -0.0095 / -0.1914 | 0/6 / 0/6 | +0.0000 |
| block_2 | ssd | global_generic | +0.0095 / +0.0000 / -0.1819 | 4/6 / 0/6 | -0.0246 |
| block_2 | ssd | det_pseudo | +0.1914 / +0.1819 / +0.0000 | 6/6 / 0/6 | +0.0003 |
| block_2 | ssd | det_pseudo_clip_radius | +0.0554 / +0.0459 / -0.1360 | 4/6 / 1/6 | -0.0162 |
| block_3 | source | no_adapt | +0.0000 / +0.1371 / +0.0323 | 0/6 / 4/6 | +0.0000 |
| block_3 | source | global_generic | -0.1371 / +0.0000 / -0.1048 | 1/6 / 4/6 | -0.2449 |
| block_3 | source | det_pseudo | -0.0323 / +0.1048 / +0.0000 | 2/6 / 0/6 | -0.5684 |
| block_3 | source | det_pseudo_clip_radius | +0.0164 / +0.1535 / +0.0487 | 3/6 / 4/6 | +0.0500 |
| block_3 | target | no_adapt | +0.0000 / -0.0149 / +0.0531 | 0/6 / 4/6 | +0.0000 |
| block_3 | target | global_generic | +0.0149 / +0.0000 / +0.0680 | 2/6 / 3/6 | +0.1104 |
| block_3 | target | det_pseudo | -0.0531 / -0.0680 / +0.0000 | 2/6 / 0/6 | -0.4133 |
| block_3 | target | det_pseudo_clip_radius | -0.0508 / -0.0657 / +0.0023 | 3/6 / 4/6 | -0.0990 |
| block_3 | ssd | no_adapt | +0.0000 / +0.0294 / +0.0506 | 0/6 / 3/6 | +0.0000 |
| block_3 | ssd | global_generic | -0.0294 / +0.0000 / +0.0212 | 2/6 / 3/6 | +0.0342 |
| block_3 | ssd | det_pseudo | -0.0506 / -0.0212 / +0.0000 | 3/6 / 0/6 | -0.1611 |
| block_3 | ssd | det_pseudo_clip_radius | +0.0839 / +0.1132 / +0.1345 | 4/6 / 5/6 | +0.0813 |
| block_4 | source | no_adapt | +0.0000 / +0.1338 / +0.3961 | 0/6 / 6/6 | +0.0000 |
| block_4 | source | global_generic | -0.1338 / +0.0000 / +0.2623 | 1/6 / 4/6 | +0.1176 |
| block_4 | source | det_pseudo | -0.3961 / -0.2623 / +0.0000 | 0/6 / 0/6 | -0.6077 |
| block_4 | source | det_pseudo_clip_radius | +0.0204 / +0.1542 / +0.4165 | 3/6 / 5/6 | +0.0884 |
| block_4 | target | no_adapt | +0.0000 / +0.0197 / +0.1123 | 0/6 / 3/6 | +0.0000 |
| block_4 | target | global_generic | -0.0197 / +0.0000 / +0.0926 | 3/6 / 3/6 | +0.0232 |
| block_4 | target | det_pseudo | -0.1123 / -0.0926 / +0.0000 | 3/6 / 0/6 | +0.1114 |
| block_4 | target | det_pseudo_clip_radius | +0.0982 / +0.1179 / +0.2105 | 4/6 / 5/6 | +0.1399 |
| block_4 | ssd | no_adapt | +0.0000 / -0.0202 / +0.1214 | 0/6 / 4/6 | +0.0000 |
| block_4 | ssd | global_generic | +0.0202 / +0.0000 / +0.1416 | 3/6 / 5/6 | -0.1223 |
| block_4 | ssd | det_pseudo | -0.1214 / -0.1416 / +0.0000 | 2/6 / 0/6 | -0.0349 |
| block_4 | ssd | det_pseudo_clip_radius | +0.0316 / +0.0114 / +0.1530 | 3/6 / 5/6 | -0.1940 |
| block_5 | source | no_adapt | +0.0000 / +0.2306 / +0.0736 | 0/6 / 2/6 | +0.0000 |
| block_5 | source | global_generic | -0.2306 / +0.0000 / -0.1570 | 1/6 / 2/6 | -0.3225 |
| block_5 | source | det_pseudo | -0.0736 / +0.1570 / +0.0000 | 4/6 / 0/6 | +0.5972 |
| block_5 | source | det_pseudo_clip_radius | +0.0315 / +0.2621 / +0.1051 | 4/6 / 4/6 | +0.0360 |
| block_5 | target | no_adapt | +0.0000 / +0.0096 / -0.1553 | 0/6 / 2/6 | +0.0000 |
| block_5 | target | global_generic | -0.0096 / +0.0000 / -0.1648 | 3/6 / 3/6 | +0.1358 |
| block_5 | target | det_pseudo | +0.1553 / +0.1648 / +0.0000 | 4/6 / 0/6 | +0.0752 |
| block_5 | target | det_pseudo_clip_radius | +0.0807 / +0.0903 / -0.0746 | 4/6 / 2/6 | +0.2701 |
| block_5 | ssd | no_adapt | +0.0000 / +0.0656 / +0.0388 | 0/6 / 3/6 | +0.0000 |
| block_5 | ssd | global_generic | -0.0656 / +0.0000 / -0.0267 | 1/6 / 2/6 | +0.0976 |
| block_5 | ssd | det_pseudo | -0.0388 / +0.0267 / +0.0000 | 3/6 / 0/6 | -0.1369 |
| block_5 | ssd | det_pseudo_clip_radius | +0.0244 / +0.0900 / +0.0633 | 4/6 / 4/6 | -0.0778 |

## Fixed safety/runtime panel

Shared-process peak allocated GPU memory includes source, FCOS, SSD and CLIP residency. Adaptation latency excludes target evaluation; native variants include original source/support setup. Existing terminal diagnostics retained.

| Group | Method | Mean phi3 | Mean saturation before / after % | Support mean | Fallback % | Deploy sec | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| corrupted_overall | global_generic | 0.024611 | 3.600 / 3.169 | 7.35 | 0.00 | 0.1157 | 1102.6 |
| corrupted_overall | det_pseudo | 0.087728 | 3.600 / 2.850 | 7.35 | 1.05 | 0.1724 | 2103.8 |
| corrupted_overall | det_pseudo_clip_radius | 0.030878 | 3.600 / 2.382 | 7.35 | 1.05 | 0.3142 | 2893.0 |
| gamma_s1 | global_generic | 0.027574 | 2.518 / 4.948 | 7.74 | 0.00 | 0.1152 | 1102.7 |
| gamma_s1 | det_pseudo | 0.085968 | 2.518 / 3.660 | 7.74 | 0.50 | 0.1762 | 2109.3 |
| gamma_s1 | det_pseudo_clip_radius | 0.035406 | 2.518 / 2.881 | 7.74 | 0.50 | 0.3148 | 2902.4 |
| gamma_s2 | global_generic | 0.028458 | 3.625 / 7.922 | 7.40 | 0.00 | 0.1153 | 1102.6 |
| gamma_s2 | det_pseudo | 0.100937 | 3.625 / 4.119 | 7.40 | 0.40 | 0.1726 | 2110.7 |
| gamma_s2 | det_pseudo_clip_radius | 0.035300 | 3.625 / 4.075 | 7.40 | 0.40 | 0.3145 | 2904.8 |
| contrast_s1 | global_generic | 0.025992 | 0.000 / 0.000 | 7.38 | 0.00 | 0.1176 | 1102.6 |
| contrast_s1 | det_pseudo | 0.080406 | 0.000 / 0.044 | 7.38 | 1.30 | 0.1715 | 2101.1 |
| contrast_s1 | det_pseudo_clip_radius | 0.035307 | 0.000 / 0.000 | 7.38 | 1.30 | 0.3142 | 2888.4 |
| contrast_s2 | global_generic | 0.022726 | 0.000 / 0.000 | 6.13 | 0.00 | 0.1154 | 1102.6 |
| contrast_s2 | det_pseudo | 0.106842 | 0.000 / 0.000 | 6.13 | 2.70 | 0.1696 | 2086.9 |
| contrast_s2 | det_pseudo_clip_radius | 0.033961 | 0.000 / 0.000 | 6.13 | 2.70 | 0.3117 | 2863.6 |
| color_cast_s1 | global_generic | 0.022134 | 5.918 / 2.935 | 7.79 | 0.00 | 0.1156 | 1102.6 |
| color_cast_s1 | det_pseudo | 0.075293 | 5.918 / 3.698 | 7.79 | 0.60 | 0.1723 | 2108.5 |
| color_cast_s1 | det_pseudo_clip_radius | 0.023812 | 5.918 / 3.006 | 7.79 | 0.60 | 0.3165 | 2901.0 |
| color_cast_s2 | global_generic | 0.020783 | 9.539 / 3.209 | 7.63 | 0.00 | 0.1152 | 1102.6 |
| color_cast_s2 | det_pseudo | 0.076924 | 9.539 / 5.580 | 7.63 | 0.80 | 0.1721 | 2106.6 |
| color_cast_s2 | det_pseudo_clip_radius | 0.021483 | 9.539 / 4.327 | 7.63 | 0.80 | 0.3135 | 2897.8 |
| clean_s0 | global_generic | 0.027339 | 2.518 / 2.655 | 7.94 | 0.00 | 0.1151 | 1099.4 |
| clean_s0 | det_pseudo | 0.080144 | 2.518 / 2.376 | 7.94 | 0.70 | 0.1722 | 2103.9 |
| clean_s0 | det_pseudo_clip_radius | 0.036655 | 2.518 / 1.864 | 7.94 | 0.70 | 0.3138 | 2895.2 |

Full per-block AP/AP50/AP75 and all paired deltas: AP_tables.csv. All phi/ratio/scale distributions and step diagnostics: analysis.json and samples.jsonl.
