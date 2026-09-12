# T012 fixed half-dose results

Full cohort: True; supported: False.
Criteria: {'both_target_macros_positive': True, 'both_targets_four_positive_blocks': False, 'both_targets_above_control_medians': False, 'all_clean_AP_within_bound': True, 'clean_phi_at_most_65_percent': True}

| Detector | Macro delta no-adapt | Macro delta full | Random median | Positive blocks | Above random block median | Clean AP delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source | +0.059581 | -0.007602 | +0.014261 | 3/5 | 3/5 | -0.056870 |
| target | +0.061752 | -0.008520 | +0.037841 | 5/5 | 3/5 | +0.022727 |
| ssd | +0.016939 | -0.015221 | +0.019289 | 3/5 | 2/5 | -0.022981 |

Mean clean phi3 half/full: 0.019448760/0.036655454; ratio 0.5305829659031093.

All condition/block AP/AP50/AP75 and paired deltas: AP_tables.csv. All macro signs/clean deltas: macro_tables.csv. All paired dose ratios: paired_dose.csv. All safety/step distributions and random-control comparisons: analysis.json.

Fixed development cohort; no AP confidence intervals, no alpha search. Continuous half dose is not half compute. Smoke criteria are unassessed.
