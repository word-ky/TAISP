# T018-B pre-outcome plan — fixed source500 confirmation

R030 / research pointer 7902e3a accepts T018-A and authorizes only an unchanged-candidate
source confirmation. T018-A report 293cfe0 remains developmental; T017 blocker is preserved.
This plan and the exact 500-image cohort must be committed before any new model outcomes.

## Fixed candidate and reuse

Keep NativePseudoTargetLoss, adapt_clip_radius/current Ours, detector loading, ISP bounds,
teacher selection, CLIP model/prompts/preprocessing, oracle native loss and corruptions byte
unchanged from the accepted T018-A code. Reuse the existing run_t018a driver with a narrow
cohort-count/task-summary extension; the model/gradient/episode loop is unchanged.
Original condition teacher is evaluated once; existing positive-class score>=0.5 stable
top20 filter; detached boxes/int labels only, no score weights. Native objective is the
unit sum of classifier, box_reg, objectness and rpn_box_reg losses. Same seeded sampling
20260912 at each loss call. Global8D identity phi, K3/LR0.1/EPS1e-12, current-phi CLIP-norm
transfer with CLIP direction discarded. Four diagnostic evaluations apply three updates.
Empty support retains exact zero phi and existing identity-ISP semantics. Both models frozen.

Reuse map: prepare_t018a for annotation eligibility/JPEG hashing/exclusions; run_t018a for
all model execution, saved predictions and official replication_ap; new small summary
function only for R030's five-block confirmation rule and AP50/AP75 diagnostics. Existing
renderer supplies table formatting. No alternative objective, fallback method or new optimizer.

## Precommitted source500 cohort

Use the same readable train2017 pool and disclosed non-crowd positive-box/positive-area
eligibility as T018-A. Prior-source exclusion union includes H32, prior4, all later reused
H subsets and all100 T018-A IDs:136unique prior-source IDs. Exclude all5000val IDs and the
existing evaluation-ID ledger. No source image replacement or performance-based selection.

New selection key fixed now: (sha256('20260919:'+decimal image ID), image ID); take first500
eligible records. Five consecutive100-image blocks block0..block4 in selected order.
Manifest includes ordered IDs, paths, JPEG SHA256s, annotation SHA256, prior-source ledger,
allval exclusion count/hash, prior manifest hashes, selection hashes and block assignment.
Seed20260919 controls cohort selection only; model/sampling seed remains20260912.

Same six corruption cases followed by clean, in existing order. Exactly no_adapt,
current_ours,nativePT_ours. Source-only FasterRCNN; no FCOS/SSD/COCOval. Annotation eligibility
is used during cohort construction; the adaptation loader receives only JPEG/image metadata.
All predictions are saved before loading annotations for official COCO AP in the formal run.

## Tests, launch and receipts

Unchanged accepted release153309 baseline: native_pseudo_target+detector_native+trust_radius
10passed2skipped1.74s onA6000 environment. One bounded extension to preparation/count/summary;
focused tests cover 5-block and +0.10 AP/clean/isolation rules and AP75 diagnostic-only behavior,
plus existing native/current tests. Then full regression and two-image CUDA K3 smoke with0AP.
No settings change from smoke. Formal launch only after passing tests/smoke and precommit.

A6000cuda:0,float32,threads1,normal CUDA settings,cudnn.benchmarkFalse. Configt018b.yaml.
Exactly500x7=3500teacherforwards,7000adaptiveepisodes,21predictionfiles and126official
evaluations (aggregate plus five blocks for sevenconditions x threemethods). Save perstep
phi/gradients/native four-loss histories/supports/counts/latency. Keep before-after source
and CLIP hashes plus per-episode source/frozen/eval/gradNone/ISP/support/finite checks.
Synchronized latency convention unchanged: adaptation includes terminal diagnostic; teacher-
inclusive adds original teacher forward, excludes final enhanced inference/state checks/AP.
Report clean/corrupt phi mean/median/update fraction, supports, paired counts and peak memory.

## Frozen source-confirmation rule

Macro=simple mean of six corrupted condition APs, reported in0–100 AP points. All six:
1. Native-current macro AP >=+0.10.
2. At least4/5 positive fixed-block native-current macro AP deltas.
3. At least4/6 positive corrupted-condition AP deltas.
4. Native corruption macro AP >no_adapt.
5. Native-current clean AP >=-0.10.
6. No implementation/isolation/reproducibility blocker.

AP50/AP75 macro/deltas are mandatory diagnostics, including whether AP75 macro delta is
nonnegative, but are not additional confirmation gates. Keep all negative blocks/conditions.
If all pass: source-confirmed, NEEDS_REVIEW; otherwise not-confirmed, with exact results.
No tuning, no follow-on target evaluation or deployment change in this task.

Persist raw runs/commands/environment/predictions/samples/tests, complete tables, report,
hash manifest, state/handoff/progress; append CODEX mailbox,commit/push,mirrorA6000.
Stop for research review. Every15minute heartbeat monitors the existing run without duplication.

Cohort prepared before model outcomes:500selected from13662eligible;136prior-source/5000val excluded;5blocks100. Manifest SHA256 e7a771126ae2fee9844dde456648b50f4c01ebcdc404318c21f9a33260da7b17. Method pins saved in T018B_method_pins.json; protected-module git diff against293cfe0 is empty.
