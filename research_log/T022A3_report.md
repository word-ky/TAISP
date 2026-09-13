# T022-A3 BLOCKED — prospective output reproducibility confirmation failed

R037/3a3c598; plan70df971; analysis code9b27c49; frozen support receipts5c35f4c.
2026-09-14 +08. **All80episodes completed; the precommitted reproducibility
criterion fails. Zero official AP evaluations. No formal200-image study.**
This is an engineering confirmation result, not evidence of AP benefit/harm.

## Frozen provenance and execution

Before new model calls, plan70df971 fixed exactly the first four ordered T022-A
cohort IDs160585,114830,449844,332316; conditions clean then contrast_s2;
5independent K=3 repeats per current/spatial method; and all four R037 criteria.
Relative L2 is ||y_i-y_j|| / max(||y_i||,1e-12) for i<j, float64 comparisons of
final float32 images, maximum over all10pairs. The same convention is used for
both methods; no alternative normalization or threshold was selected later.

Exactly8original-view teacher support sets were generated once in preparation,
saved with support/mask hashes and committed/pushed as5c35f4c **before repeated
adaptation**. Both methods and all repeats reused these supports. No teacher
recomputation, gradient/state averaging, shared trajectory or GT was used.
Support file SHA2560d7115a0acfae5cd81eaf6cc55be39fadf36bf2203fdcbcb88237349efc9eeea.
Cohort SHA256589878ad7af1d86bf73514d63dd58a4e40d58797d119ea4981be9e4292772690 unchanged.

Preparation run `20260914-024325-taisp-t022a3-supports`, exit0 at02:43:52,
8teacher calls, zero adaptation/AP. The earlier024219 launcher failed on SSH
before its remote directory existed; read-only checks confirmed no duplicate
run before retry. No library/driver/environment changes were made.
Focused unit increment17passed5.21s; run focused **17passed5.17s**;
full **192passed10skipped4warnings10.09s**.

Confirmation run `20260914-024501-taisp-t022a3-confirmation80`,
02:45:06–02:47:28 +08, shell exit0, result status **blocked**.
Both runs used release `20260914-024158-taisp-t022a3-confirmation`, code9b27c49;
confirmation explicitly selected that release. All model/gradient/adaptation
work used A6000 CUDA. Deterministic-algorithms mode stayed off. Known NVML
warning did not prevent CUDA. Offline tables used CPU.

## All eight prospective output comparisons

Each value is maximum final-image relative-L2 dispersion over all10repeat pairs.
Ratios omit the additive floor and are descriptive; decisions include1e-6.

| ID | Condition | Current d | Spatial d | Ratio | Within2+floor | Above5+floor |
| --- | --- | --- | --- | --- | --- | --- |
| 160585 | clean_s0 | 0.00042534277012 | 0.00203434393693 | 4.782834174783982 | False | False |
| 160585 | contrast_s2 | 0.00415652879695 | 0.00184905862115 | 0.44485644427909377 | True | False |
| 114830 | clean_s0 | 0.000695633163196 | 0.000685488011403 | 0.9854159457453723 | True | False |
| 114830 | contrast_s2 | 7.31457936681e-05 | 8.07827356886e-05 | 1.104407124970579 | True | False |
| 449844 | clean_s0 | 0.000849768315773 | 0.00304512364005 | 3.583475146729563 | False | False |
| 449844 | contrast_s2 | 0.000198266680021 | 0.000121317091272 | 0.6118884487244847 | True | False |
| 332316 | clean_s0 | 0.000161073270697 | 0.000119420233806 | 0.741403171915498 | True | False |
| 332316 | contrast_s2 | 0.000168634435414 | 0.000900430059227 | 5.3395384935188295 | False | True |

## Apply the unchanged conjunction

| Precommitted criterion | Observed | Result |
| --- | --- | --- |
| All80episodes finite; reset/isolation/support/mask/model checks pass | 80/80 | PASS |
| At least7/8 tuples: d_sp <=2*d_cur+1e-6 | 5/8 | FAIL |
| Median(d_sp) <=1.25*median(d_cur)+1e-6 | 0.000792959035315 >0.000390755906339 | FAIL |
| No tuple: d_sp >5*d_cur+1e-6 | 332316/contrast_s2 exceeds bound | FAIL |

Median current dispersion0.000311804725071; median spatial0.000792959035315.
The two other failures of the2x rule are160585/clean and449844/clean.
The largest descriptive ratio is5.3395384935 for332316/contrast_s2.
No post-outcome tolerance change, new repeat, sample replacement or AP was used.

All80final images/states, per-step states and three update vectors are saved as
lossless compressed tensors; all160within-method pair comparisons are saved
for every field. All80episode frozen/eval/grad-none/model-state checks, support
and mask hashes, identity reset, finite state/image/update checks pass. Raw
per-step diagnostics retain loss/gradient/state details. Actual release hashes
match23prior modules,2shared R036 analysis/test modules and2new R037 modules
(27 LF pins, raw hashes also retained). Current Ours, spatial runtime/formula,
ISP, detector/CLIP, masks, rho.5,K3,LR.1 and previous receipts are unchanged.

No zero-update episodes occurred:0/40current and0/40spatial. Mean synchronized
adaptation time is0.298330976s current and0.597325252s spatial. Timing includes
terminal diagnostics (four gradient evaluations / three updates), excludes
teacher preparation, hashing, tensor saving and pairwise comparisons. Complete
per-tuple latency, mask area/support count, state/update dispersion and all
pairwise max-abs/relative-L2/cosine summaries are in T022A3/complete_tables.md.
State/update metrics do not override the primary output criterion.

## Interpretation and stop

The preceding one-image R036 attribution established inherited upstream
variation and exact fixed-input JVP/dose. It did not guarantee acceptable
output variability on additional tuples. This prospective four-image check
fails the research-defined relative-to-baseline requirement despite passing
finite/reset/isolation checks. The spatial candidate is **not cleared for AP**.
The five original AP performance criteria are **NOT EVALUATED**; the sixth
criterion, absence of a reproducibility blocker, is **not satisfied**. No scientific AP
PASS/FAIL or candidate promotion is claimed.

Per R037, stop **BLOCKED** with all receipts. No200image run, official COCO AP,
FCOS/SSD/val, newcohort, rho/mask/threshold/topk/K/LR/CLIP/ISP/dtype tuning,
objective change, deterministic-mode performance run or training was started.
No active TAISP job remains. Await the next explicit research decision.

Both raw archives were fetched and SHA256-verified: preparation016981867c1d2168006b830528a5b3e4994d978777c14a97576567de216b9fdf; confirmationfb9296465cc53ebb42df92e5594c04361f4d70034342424b03484cb5bc25c2ab. 127 rawfiles / 191,292,120bytes retained. Artifact manifests, commands, full environment, every repeat and original support hashes remain under the two named research_log/remote_runs directories.
