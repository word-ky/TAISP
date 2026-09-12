# T009 fixed-method external validation

Images: 2; variant rows: 42. K=3 only. No AP confidence intervals.

## Independent-target replication

| Detector | Reference | Aggregate macro AP delta | Block deltas (AP points) | Positive blocks |
| --- | --- | ---: | --- | ---: |
| target | no_adapt | +1.8081 | block_1: -0.1331 / block_2: +3.7500 | 1/2 |
| target | det_pseudo | -0.9636 | block_1: +0.0797 / block_2: -2.0058 | 1/2 |
| ssd | no_adapt | -2.1890 | block_1: -0.1864 / block_2: -4.1916 | 0/2 |
| ssd | det_pseudo | +1.6280 | block_1: -0.0580 / block_2: +3.3141 | 1/2 |

Full 1,000-image/five-block coverage: False. External AP criterion: not assessed on a partial/smoke cohort. Clean outcomes still require review.

## Aggregate AP / AP50 / AP75

| Detector | Condition | Method | AP / AP50 / AP75 | AP delta no-adapt / CLIP / raw |
| --- | --- | --- | ---: | ---: |
| source | gamma_s1 | no_adapt | 23.519 / 36.052 / 25.652 | +0.000 / +0.140 / +1.778 |
| source | gamma_s2 | no_adapt | 20.508 / 36.132 / 16.584 | +0.000 / +1.002 / -0.551 |
| source | contrast_s1 | no_adapt | 23.215 / 47.969 / 18.936 | +0.000 / +1.010 / +3.733 |
| source | contrast_s2 | no_adapt | 26.763 / 65.790 / 7.013 | +0.000 / +0.000 / +3.342 |
| source | color_cast_s1 | no_adapt | 20.915 / 35.396 / 23.597 | +0.000 / -1.926 / -3.060 |
| source | color_cast_s2 | no_adapt | 19.880 / 36.056 / 22.896 | +0.000 / +0.302 / +0.302 |
| source | clean_s0 | no_adapt | 23.838 / 35.176 / 28.864 | +0.000 / +0.926 / +0.388 |
| source | gamma_s1 | global_generic | 23.379 / 36.169 / 24.158 | -0.140 / +0.000 / +1.638 |
| source | gamma_s2 | global_generic | 19.506 / 36.794 / 17.879 | -1.002 / +0.000 / -1.553 |
| source | contrast_s1 | global_generic | 22.205 / 42.919 / 18.936 | -1.010 / +0.000 / +2.723 |
| source | contrast_s2 | global_generic | 26.763 / 65.790 / 7.013 | +0.000 / +0.000 / +3.342 |
| source | color_cast_s1 | global_generic | 22.841 / 38.236 / 23.597 | +1.926 / +0.000 / -1.134 |
| source | color_cast_s2 | global_generic | 19.578 / 35.679 / 22.896 | -0.302 / +0.000 / +0.000 |
| source | clean_s0 | global_generic | 22.912 / 34.208 / 27.104 | -0.926 / +0.000 / -0.539 |
| source | gamma_s1 | det_pseudo | 21.741 / 35.956 / 18.478 | -1.778 / -1.638 / +0.000 |
| source | gamma_s2 | det_pseudo | 21.060 / 36.132 / 17.636 | +0.551 / +1.553 / +0.000 |
| source | contrast_s1 | det_pseudo | 19.482 / 30.419 / 27.104 | -3.733 / -2.723 / +0.000 |
| source | contrast_s2 | det_pseudo | 23.421 / 65.790 / 7.013 | -3.342 / -3.342 / +0.000 |
| source | color_cast_s1 | det_pseudo | 23.975 / 36.840 / 24.649 | +3.060 / +1.134 / +0.000 |
| source | color_cast_s2 | det_pseudo | 19.578 / 35.679 / 22.896 | -0.302 / +0.000 / +0.000 |
| source | clean_s0 | det_pseudo | 23.450 / 37.129 / 28.864 | -0.388 / +0.539 / +0.000 |
| source | gamma_s1 | det_pseudo_clip_radius | 22.312 / 35.956 / 18.369 | -1.207 / -1.067 / +0.571 |
| source | gamma_s2 | det_pseudo_clip_radius | 21.095 / 36.132 / 17.987 | +0.587 / +1.588 / +0.035 |
| source | contrast_s1 | det_pseudo_clip_radius | 20.281 / 35.469 / 27.104 | -2.933 / -1.923 / +0.800 |
| source | contrast_s2 | det_pseudo_clip_radius | 26.387 / 66.354 / 9.818 | -0.375 / -0.375 / +2.966 |
| source | color_cast_s1 | det_pseudo_clip_radius | 23.134 / 36.840 / 24.649 | +2.219 / +0.293 / -0.842 |
| source | color_cast_s2 | det_pseudo_clip_radius | 19.578 / 35.679 / 22.896 | -0.302 / +0.000 / +0.000 |
| source | clean_s0 | det_pseudo_clip_radius | 23.724 / 37.309 / 29.084 | -0.114 / +0.812 / +0.274 |
| target | gamma_s1 | no_adapt | 33.052 / 60.787 / 27.136 | +0.000 / -0.404 / -9.163 |
| target | gamma_s2 | no_adapt | 33.557 / 59.514 / 27.049 | +0.000 / -1.126 / +0.406 |
| target | contrast_s1 | no_adapt | 42.341 / 57.295 / 39.102 | +0.000 / +0.132 / -0.180 |
| target | contrast_s2 | no_adapt | 28.314 / 57.983 / 22.851 | +0.000 / -1.169 / -10.048 |
| target | color_cast_s1 | no_adapt | 47.482 / 64.671 / 46.917 | +0.000 / +1.091 / +0.984 |
| target | color_cast_s2 | no_adapt | 42.919 / 60.182 / 38.669 | +0.000 / +0.146 / +1.370 |
| target | clean_s0 | no_adapt | 48.489 / 64.960 / 49.444 | +0.000 / +1.424 / +3.110 |
| target | gamma_s1 | global_generic | 33.455 / 60.966 / 26.757 | +0.404 / +0.000 / -8.759 |
| target | gamma_s2 | global_generic | 34.683 / 61.592 / 26.255 | +1.126 / +0.000 / +1.532 |
| target | contrast_s1 | global_generic | 42.209 / 58.176 / 38.810 | -0.132 / +0.000 / -0.312 |
| target | contrast_s2 | global_generic | 29.484 / 60.626 / 22.803 | +1.169 / +0.000 / -8.878 |
| target | color_cast_s1 | global_generic | 46.391 / 61.497 / 46.376 | -1.091 / +0.000 / -0.107 |
| target | color_cast_s2 | global_generic | 42.772 / 60.182 / 38.415 | -0.146 / +0.000 / +1.224 |
| target | clean_s0 | global_generic | 47.065 / 65.341 / 44.934 | -1.424 / +0.000 / +1.686 |
| target | gamma_s1 | det_pseudo | 42.214 / 61.005 / 39.043 | +9.163 / +8.759 / +0.000 |
| target | gamma_s2 | det_pseudo | 33.151 / 59.454 / 26.025 | -0.406 / -1.532 / +0.000 |
| target | contrast_s1 | det_pseudo | 42.521 / 58.627 / 39.102 | +0.180 / +0.312 / +0.000 |
| target | contrast_s2 | det_pseudo | 38.362 / 54.968 / 29.929 | +10.048 / +8.878 / +0.000 |
| target | color_cast_s1 | det_pseudo | 46.497 / 63.475 / 45.925 | -0.984 / +0.107 / +0.000 |
| target | color_cast_s2 | det_pseudo | 41.548 / 60.204 / 38.669 | -1.370 / -1.224 / +0.000 |
| target | clean_s0 | det_pseudo | 45.379 / 64.960 / 46.515 | -3.110 / -1.686 / +0.000 |
| target | gamma_s1 | det_pseudo_clip_radius | 33.086 / 61.506 / 27.236 | +0.035 / -0.369 / -9.128 |
| target | gamma_s2 | det_pseudo_clip_radius | 33.418 / 59.469 / 26.888 | -0.139 / -1.265 / +0.267 |
| target | contrast_s1 | det_pseudo_clip_radius | 42.521 / 58.627 / 39.102 | +0.180 / +0.312 / +0.000 |
| target | contrast_s2 | det_pseudo_clip_radius | 39.646 / 58.360 / 35.371 | +11.332 / +10.163 / +1.284 |
| target | color_cast_s1 | det_pseudo_clip_radius | 46.915 / 63.500 / 45.925 | -0.566 / +0.525 / +0.418 |
| target | color_cast_s2 | det_pseudo_clip_radius | 42.925 / 60.204 / 38.669 | +0.006 / +0.153 / +1.377 |
| target | clean_s0 | det_pseudo_clip_radius | 48.149 / 64.976 / 49.444 | -0.340 / +1.084 / +2.770 |
| ssd | gamma_s1 | no_adapt | 23.074 / 48.631 / 16.708 | +0.000 / -0.086 / -0.616 |
| ssd | gamma_s2 | no_adapt | 22.636 / 46.383 / 12.500 | +0.000 / -0.468 / -0.074 |
| ssd | contrast_s1 | no_adapt | 11.277 / 36.595 / 4.370 | +0.000 / +0.205 / -0.184 |
| ssd | contrast_s2 | no_adapt | 21.825 / 42.999 / 12.500 | +0.000 / -3.786 / +0.507 |
| ssd | color_cast_s1 | no_adapt | 38.207 / 61.852 / 41.584 | +0.000 / +0.038 / +12.032 |
| ssd | color_cast_s2 | no_adapt | 33.910 / 57.711 / 41.584 | +0.000 / +0.001 / +11.239 |
| ssd | clean_s0 | no_adapt | 26.698 / 52.021 / 16.708 | +0.000 / +0.123 / +0.165 |
| ssd | gamma_s1 | global_generic | 23.161 / 49.125 / 16.708 | +0.086 / +0.000 / -0.530 |
| ssd | gamma_s2 | global_generic | 23.104 / 47.472 / 12.500 | +0.468 / +0.000 / +0.394 |
| ssd | contrast_s1 | global_generic | 11.072 / 37.372 / 4.208 | -0.205 / +0.000 / -0.390 |
| ssd | contrast_s2 | global_generic | 25.611 / 53.494 / 12.500 | +3.786 / +0.000 / +4.292 |
| ssd | color_cast_s1 | global_generic | 38.168 / 59.555 / 41.664 | -0.038 / +0.000 / +11.993 |
| ssd | color_cast_s2 | global_generic | 33.909 / 57.700 / 41.584 | -0.001 / +0.000 / +11.238 |
| ssd | clean_s0 | global_generic | 26.575 / 51.934 / 16.708 | -0.123 / +0.000 / +0.042 |
| ssd | gamma_s1 | det_pseudo | 23.691 / 47.640 / 16.708 | +0.616 / +0.530 / +0.000 |
| ssd | gamma_s2 | det_pseudo | 22.710 / 47.126 / 12.500 | +0.074 / -0.394 / +0.000 |
| ssd | contrast_s1 | det_pseudo | 11.462 / 36.557 / 4.208 | +0.184 / +0.390 / +0.000 |
| ssd | contrast_s2 | det_pseudo | 21.319 / 39.865 / 15.305 | -0.507 / -4.292 / +0.000 |
| ssd | color_cast_s1 | det_pseudo | 26.175 / 48.802 / 16.708 | -12.032 / -11.993 / +0.000 |
| ssd | color_cast_s2 | det_pseudo | 22.672 / 45.200 / 16.708 | -11.239 / -11.238 / +0.000 |
| ssd | clean_s0 | det_pseudo | 26.533 / 51.342 / 16.708 | -0.165 / -0.042 / +0.000 |
| ssd | gamma_s1 | det_pseudo_clip_radius | 22.740 / 47.594 / 16.708 | -0.334 / -0.420 / -0.950 |
| ssd | gamma_s2 | det_pseudo_clip_radius | 22.640 / 47.126 / 12.500 | +0.004 / -0.464 / -0.070 |
| ssd | contrast_s1 | det_pseudo_clip_radius | 11.223 / 36.565 / 4.364 | -0.054 / +0.151 / -0.239 |
| ssd | contrast_s2 | det_pseudo_clip_radius | 21.788 / 42.588 / 12.636 | -0.037 / -3.823 / +0.469 |
| ssd | color_cast_s1 | det_pseudo_clip_radius | 25.494 / 46.905 / 16.708 | -12.712 / -12.674 / -0.680 |
| ssd | color_cast_s2 | det_pseudo_clip_radius | 33.910 / 57.711 / 41.584 | +0.000 / +0.001 / +11.239 |
| ssd | clean_s0 | det_pseudo_clip_radius | 26.533 / 51.342 / 16.708 | -0.165 / -0.042 / +0.000 |

## Every method and replication block: macro AP deltas

| Group | Detector | Method | Macro delta no-adapt / CLIP / raw | Positive conditions vs no-adapt / raw | Clean delta vs no-adapt |
| --- | --- | --- | ---: | ---: | ---: |
| aggregate | source | no_adapt | +0.0000 / +0.0880 / +0.9238 | 0/6 / 4/6 | +0.0000 |
| aggregate | source | global_generic | -0.0880 / +0.0000 / +0.8358 | 1/6 / 3/6 | -0.9263 |
| aggregate | source | det_pseudo | -0.9238 / -0.8358 / +0.0000 | 2/6 / 0/6 | -0.3878 |
| aggregate | source | det_pseudo_clip_radius | -0.3354 / -0.2474 / +0.5884 | 2/6 / 4/6 | -0.1140 |
| aggregate | target | no_adapt | +0.0000 / -0.2217 / -2.7717 | 0/6 / 3/6 | +0.0000 |
| aggregate | target | global_generic | +0.2217 / +0.0000 / -2.5500 | 3/6 / 2/6 | -1.4240 |
| aggregate | target | det_pseudo | +2.7717 / +2.5500 / +0.0000 | 3/6 / 0/6 | -3.1101 |
| aggregate | target | det_pseudo_clip_radius | +1.8081 / +1.5864 / -0.9636 | 4/6 / 4/6 | -0.3404 |
| aggregate | ssd | no_adapt | +0.0000 / -0.6826 / +3.8170 | 0/6 / 3/6 | +0.0000 |
| aggregate | ssd | global_generic | +0.6826 / +0.0000 / +4.4996 | 3/6 / 4/6 | -0.1232 |
| aggregate | ssd | det_pseudo | -3.8170 / -4.4996 / +0.0000 | 3/6 / 0/6 | -0.1650 |
| aggregate | ssd | det_pseudo_clip_radius | -2.1890 / -2.8716 / +1.6280 | 1/6 / 2/6 | -0.1650 |
| block_1 | source | no_adapt | +0.0000 / +0.7538 / +0.4104 | 0/6 / 3/6 | +0.0000 |
| block_1 | source | global_generic | -0.7538 / +0.0000 / -0.3434 | 0/6 / 1/6 | -1.8526 |
| block_1 | source | det_pseudo | -0.4104 / +0.3434 / +0.0000 | 2/6 / 0/6 | +0.9076 |
| block_1 | source | det_pseudo_clip_radius | -0.4935 / +0.2603 / -0.0831 | 2/6 / 2/6 | +1.4552 |
| block_1 | target | no_adapt | +0.0000 / -0.1888 / +0.2128 | 0/6 / 2/6 | +0.0000 |
| block_1 | target | global_generic | +0.1888 / +0.0000 / +0.4015 | 4/6 / 2/6 | -2.0264 |
| block_1 | target | det_pseudo | -0.2128 / -0.4015 / +0.0000 | 4/6 / 0/6 | -0.5124 |
| block_1 | target | det_pseudo_clip_radius | -0.1331 / -0.3219 / +0.0797 | 4/6 / 3/6 | +0.1397 |
| block_1 | ssd | no_adapt | +0.0000 / -0.3189 / +0.1284 | 0/6 / 4/6 | +0.0000 |
| block_1 | ssd | global_generic | +0.3189 / +0.0000 / +0.4473 | 3/6 / 4/6 | -0.2464 |
| block_1 | ssd | det_pseudo | -0.1284 / -0.4473 / +0.0000 | 2/6 / 0/6 | -0.3300 |
| block_1 | ssd | det_pseudo_clip_radius | -0.1864 / -0.5053 / -0.0580 | 2/6 / 3/6 | -0.3300 |
| block_2 | source | no_adapt | +0.0000 / -0.6452 / +1.4381 | 0/6 / 3/6 | +0.0000 |
| block_2 | source | global_generic | +0.6452 / +0.0000 / +2.0833 | 1/6 / 3/6 | +0.0000 |
| block_2 | source | det_pseudo | -1.4381 / -2.0833 / +0.0000 | 1/6 / 0/6 | -1.6832 |
| block_2 | source | det_pseudo_clip_radius | -0.1774 / -0.8226 / +1.2607 | 1/6 / 3/6 | -1.6832 |
| block_2 | target | no_adapt | +0.0000 / -0.2666 / -5.7558 | 0/6 / 2/6 | +0.0000 |
| block_2 | target | global_generic | +0.2666 / +0.0000 / -5.4892 | 3/6 / 2/6 | -0.8168 |
| block_2 | target | det_pseudo | +5.7558 / +5.4892 / +0.0000 | 2/6 / 0/6 | -5.7046 |
| block_2 | target | det_pseudo_clip_radius | +3.7500 / +3.4834 / -2.0058 | 1/6 / 3/6 | -0.8168 |
| block_2 | ssd | no_adapt | +0.0000 / -1.0463 / +7.5056 | 0/6 / 3/6 | +0.0000 |
| block_2 | ssd | global_generic | +1.0463 / +0.0000 / +8.5520 | 1/6 / 3/6 | +0.0000 |
| block_2 | ssd | det_pseudo | -7.5056 / -8.5520 / +0.0000 | 2/6 / 0/6 | +0.0000 |
| block_2 | ssd | det_pseudo_clip_radius | -4.1916 / -5.2379 / +3.3141 | 0/6 / 2/6 | +0.0000 |

## Fixed safety/runtime panel

Shared-process peak allocated GPU memory includes source, FCOS, SSD and CLIP residency. Adaptation latency excludes target evaluation; native variants include original source/support setup. Existing terminal diagnostics retained.

| Group | Method | Mean phi3 | Mean saturation before / after % | Support mean | Fallback % | Deploy sec | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| corrupted_overall | global_generic | 0.027292 | 2.894 / 13.794 | 5.00 | 0.00 | 0.1161 | 1094.1 |
| corrupted_overall | det_pseudo | 0.061767 | 2.894 / 8.607 | 5.00 | 0.00 | 0.2006 | 2117.8 |
| corrupted_overall | det_pseudo_clip_radius | 0.025592 | 2.894 / 7.995 | 5.00 | 0.00 | 0.3270 | 2928.4 |
| gamma_s1 | global_generic | 0.025668 | 2.839 / 20.775 | 5.50 | 0.00 | 0.1302 | 1092.7 |
| gamma_s1 | det_pseudo | 0.059161 | 2.839 / 17.174 | 5.50 | 0.00 | 0.3412 | 2118.2 |
| gamma_s1 | det_pseudo_clip_radius | 0.021996 | 2.839 / 17.295 | 5.50 | 0.00 | 0.4204 | 2929.1 |
| gamma_s2 | global_generic | 0.040582 | 5.433 / 39.569 | 4.00 | 0.00 | 0.1113 | 1094.3 |
| gamma_s2 | det_pseudo | 0.041461 | 5.433 / 18.028 | 4.00 | 0.00 | 0.1731 | 2117.6 |
| gamma_s2 | det_pseudo_clip_radius | 0.043439 | 5.433 / 16.986 | 4.00 | 0.00 | 0.3144 | 2928.3 |
| contrast_s1 | global_generic | 0.013580 | 0.000 / 0.000 | 5.50 | 0.00 | 0.1166 | 1094.5 |
| contrast_s1 | det_pseudo | 0.039729 | 0.000 / 0.000 | 5.50 | 0.00 | 0.1727 | 2117.7 |
| contrast_s1 | det_pseudo_clip_radius | 0.013942 | 0.000 / 0.000 | 5.50 | 0.00 | 0.3176 | 2928.6 |
| contrast_s2 | global_generic | 0.024910 | 0.000 / 0.000 | 3.50 | 0.00 | 0.1119 | 1094.5 |
| contrast_s2 | det_pseudo | 0.093818 | 0.000 / 0.000 | 3.50 | 0.00 | 0.1729 | 2117.6 |
| contrast_s2 | det_pseudo_clip_radius | 0.017237 | 0.000 / 0.000 | 3.50 | 0.00 | 0.3177 | 2927.9 |
| color_cast_s1 | global_generic | 0.043027 | 3.064 / 12.787 | 5.50 | 0.00 | 0.1137 | 1094.5 |
| color_cast_s1 | det_pseudo | 0.067326 | 3.064 / 8.960 | 5.50 | 0.00 | 0.1729 | 2117.7 |
| color_cast_s1 | det_pseudo_clip_radius | 0.041201 | 3.064 / 10.340 | 5.50 | 0.00 | 0.3150 | 2927.9 |
| color_cast_s2 | global_generic | 0.015985 | 6.028 / 9.634 | 6.00 | 0.00 | 0.1127 | 1094.4 |
| color_cast_s2 | det_pseudo | 0.069111 | 6.028 / 7.481 | 6.00 | 0.00 | 0.1706 | 2117.7 |
| color_cast_s2 | det_pseudo_clip_radius | 0.015737 | 6.028 / 3.350 | 6.00 | 0.00 | 0.2768 | 2928.9 |
| clean_s0 | global_generic | 0.032984 | 2.839 / 9.353 | 6.00 | 0.00 | 0.0939 | 1091.8 |
| clean_s0 | det_pseudo | 0.051503 | 2.839 / 5.140 | 6.00 | 0.00 | 0.1709 | 2115.0 |
| clean_s0 | det_pseudo_clip_radius | 0.028837 | 2.839 / 3.856 | 6.00 | 0.00 | 0.3150 | 2926.4 |

Full per-block AP/AP50/AP75 and all paired deltas: AP_tables.csv. All phi/ratio/scale distributions and step diagnostics: analysis.json and samples.jsonl.
