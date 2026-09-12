# T012 frozen half-dose implementation and evidence plan

R014 (4afc07a, pointer cbbc8d8) accepts/closes T011 and authorizes one fixed
half-dose variant on the existing T009 cohort. This plan precedes new model execution.

## Contract

At each current phi compute the accepted source pseudo gradient gd and CLIP gc,
h = gd * (||gc|| / (||gd|| + 1e-12)); apply exactly .5*h with lr .1 and K=3.
Identity initialization, same global8D ISP/hardclamp, original detached source
score>=.5/top20 support, all models/prompts/preprocessing/seeds remain unchanged.
Empty support remains exact zero. No other dose, gate, cap, objective or search.

Reuse exactly T009 1000 IDs, six corruptions/clean, five fixed200-image blocks.
Only half-dose model predictions are newly evaluated; T009 no-adapt/full-hybrid
endpoints are authoritative, T011's 200 random results are fixed comparison data.
Verify input IDs/subset, model/prompt/config pins and reference file hashes before
new study. Smoke uses the first two existing T009 images and matching old smoke
reference, not a new cohort; it yields no scientific conclusion.

## Reuse and increments

| Increment | Existing owner / change | Focused check |
| --- | --- | --- |
| Baseline | T007 trust_radius and T009 replication; same repository code | Fresh remote4 passed1 skipped in1.56s (real-model opt-in skipped). |
| Half-dose algebra | Share existing trust_radius loop behind unchanged full public signature; new fixed half-dose wrapper | Same-phi half algebra, full regression, freeze/reset/fixedsupport/empty/diagnostics and isolation tests. |
| Experiment | Extend existing run_t009 driver with half-dose variant and reference-only endpoints mode; same loaders/corruption/prediction_records/replication_ap | Reference pin/ID/config comparisons; two-image real-model smoke. |
| Reporting | Existing T009 summaries and T011 distributions; new T012 comparison | Hand-computed decision fixtures, smoke AP and paired norm ratios; full report. |

Record raw detector norm, CLIP norm, pre-attenuation hybrid norm, applied half norm,
phi/physical state/saturation and timings at steps0..3 (step3 terminal diagnostic).
Compare each image/case against saved full-hybrid phi and actual step0..2 update
norms. Report ratios only with positive full denominator; explicitly count both-zero
and nonzero/zero cases, rather than silently adding an epsilon to descriptive ratios.
Step0 attenuation is exactly .5; subsequent trajectory ratios need not stay .5.
Timing includes the same support setup and terminal diagnostic conventions as T009;
all four frozen models resident; target evaluation excluded. Continuous half-dose
is not claimed to halve inference time or FLOPs like discrete thinning.

## Predeclared result

AP/AP50/AP75 for three detectors x seven conditions x six groups:126 new half-dose
AP rows plus252 reused endpoints,378 total rows. All negatives and five blocks stay.
Macro deltas vs no-adapt/full, clean deltas, phi/saturation/nonzero/timing distributions,
paired phi and update norm ratios (clean/corrupted and conditions/blocks) retained.
Compare aggregate and block AP deltas against T011 frozen control medians; retain
percentiles/tails descriptively with the same strict rank/linear quantile conventions.

R014 passes iff both target macros>0; both>=4/5 positive blocks; both above T011
aggregate random medians and both>=4/5 above block random medians; all three clean
AP deltas>=-.10; mean clean phi_half<=.65*mean clean phi_full. No parameter responds
to smoke or full results. If failure, stop scalar gating and dose sweeps; if success,
still developmental. Await review before T013/newcohort/alpha sweep/cap/meta.
