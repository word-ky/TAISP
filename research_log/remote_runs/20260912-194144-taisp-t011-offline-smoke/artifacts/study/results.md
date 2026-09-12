# T011 matched-random falsification

Images: 2; configurations: 203; random draws: 200; AP rows: 12789.
Full cohort: False. Selection information beyond thinning: None.

Random selectors satisfying R012: None/200 (unassessed on smoke).

All deltas are AP points. Percentiles are strict empirical ranks; corrected upper tail counts ties as >=. These are control distributions, not AP confidence intervals.

| Detector | Candidate macro AP | Random mean / median / p05 / p95 | Candidate percentile | Upper tail | Candidate > block medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | -0.463432 | -0.463432 / -0.463432 / -0.463432 / -0.463432 | 0.00% | 1.000000 | 0/2 |
| target | +1.817521 | +1.817521 / +1.817521 / +1.817521 / +1.817521 | 0.00% | 1.000000 | 0/2 |
| ssd | -0.087680 | -0.087680 / -0.087680 / -0.087680 / -0.087680 | 0.00% | 1.000000 | 0/2 |

## Every block

| Detector | Block | Candidate | Random median / p05 / p95 | Candidate minus median |
| --- | --- | ---: | ---: | ---: |
| source | block_1 | -0.748641 | -0.748641 / -0.748641 / -0.748641 | +0.000000 |
| source | block_2 | -0.181518 | -0.181518 / -0.181518 / -0.181518 | +0.000000 |
| target | block_1 | -0.114890 | -0.114890 / -0.114890 / -0.114890 | +0.000000 |
| target | block_2 | +3.750000 | +3.750000 / +3.750000 / +3.750000 | +0.000000 |
| ssd | block_1 | -0.175360 | -0.175360 / -0.175360 / -0.175360 | +0.000000 |
| ssd | block_2 | +0.000000 | +0.000000 / +0.000000 / +0.000000 | +0.000000 |

All 200 draws and three anchors: config_summary.csv. All condition/group AP/AP50/AP75 and both reference deltas: AP_tables.csv. All block macros and clean deltas: macro_tables.csv. Candidate-minus-control distributions for macro and clean AP/AP50/AP75, exact ties, all tail probabilities, and descriptive safety/cost are in analysis.json.

Controls exactly match selected counts within condition×block and are not deployment rules. The candidate was selected in T010; even a passing test remains developmental. No model rerun, threshold change, alternative candidate, new cohort or meta-training was performed.
