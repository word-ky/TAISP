# T011 matched-random falsification

Images: 1000; configurations: 203; random draws: 200; AP rows: 25578.
Full cohort: True. Selection information beyond thinning: False.

Random selectors satisfying R012: 96/200 (unassessed on smoke).

All deltas are AP points. Percentiles are strict empirical ranks; corrected upper tail counts ties as >=. These are control distributions, not AP confidence intervals.

| Detector | Candidate macro AP | Random mean / median / p05 / p95 | Candidate percentile | Upper tail | Candidate > block medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | +0.078574 | +0.009333 / +0.014261 / -0.055059 / +0.066718 | 99.00% | 0.014925 | 5/5 |
| target | +0.036793 | +0.038260 / +0.037841 / +0.017211 / +0.058859 | 47.00% | 0.532338 | 2/5 |
| ssd | +0.032926 | +0.017373 / +0.019289 / -0.000565 / +0.032105 | 96.00% | 0.044776 | 4/5 |

## Every block

| Detector | Block | Candidate | Random median / p05 / p95 | Candidate minus median |
| --- | --- | ---: | ---: | ---: |
| source | block_1 | -0.009064 | -0.041739 / -0.106545 / +0.024827 | +0.032675 |
| source | block_2 | +0.025441 | +0.023703 / -0.064681 / +0.111462 | +0.001739 |
| source | block_3 | +0.031647 | +0.016061 / -0.095038 / +0.161281 | +0.015586 |
| source | block_4 | +0.071513 | -0.013804 / -0.088258 / +0.079734 | +0.085317 |
| source | block_5 | +0.085997 | +0.008984 / -0.079733 / +0.115624 | +0.077013 |
| target | block_1 | +0.046015 | +0.060170 / +0.007513 / +0.121738 | -0.014155 |
| target | block_2 | +0.071191 | +0.079141 / -0.012817 / +0.159454 | -0.007949 |
| target | block_3 | -0.032180 | -0.027437 / -0.082778 / +0.024627 | -0.004744 |
| target | block_4 | +0.068303 | +0.056616 / -0.003941 / +0.125028 | +0.011688 |
| target | block_5 | +0.072118 | +0.057068 / -0.000011 / +0.107924 | +0.015049 |
| ssd | block_1 | +0.011162 | +0.005753 / -0.032028 / +0.041059 | +0.005409 |
| ssd | block_2 | +0.068580 | +0.038050 / -0.023513 / +0.141986 | +0.030530 |
| ssd | block_3 | +0.086320 | +0.036298 / -0.016471 / +0.092529 | +0.050022 |
| ssd | block_4 | +0.005109 | +0.009776 / -0.037396 / +0.054093 | -0.004667 |
| ssd | block_5 | +0.104044 | +0.025709 / -0.046658 / +0.093913 | +0.078335 |

All 200 draws and three anchors: config_summary.csv. All condition/group AP/AP50/AP75 and both reference deltas: AP_tables.csv. All block macros and clean deltas: macro_tables.csv. Candidate-minus-control distributions for macro and clean AP/AP50/AP75, exact ties, all tail probabilities, and descriptive safety/cost are in analysis.json.

Controls exactly match selected counts within condition×block and are not deployment rules. The candidate was selected in T010; even a passing test remains developmental. No model rerun, threshold change, alternative candidate, new cohort or meta-training was performed.
