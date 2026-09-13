# T019-A pre-outcome plan — fixed native component family

R031 / pointer d60e72c accepts the negative T018-B confirmation and closes full-native.
Only the four named component subsets below are authorized. No full-native fifth candidate.
Precommit this plan/config/cohort/method hashes before real-model outcomes or AP evaluation.

## Fixed family and method

| Candidate | Unit-weight active loss components, in sum order |
| --- | --- |
| native_cls | loss_classifier |
| native_conf | loss_classifier, loss_objectness |
| native_roi | loss_classifier, loss_box_reg |
| native_conf_roi | loss_classifier, loss_objectness, loss_box_reg |

loss_rpn_box_reg is absent from every candidate. The detector native forward still computes
its usual four losses, but only the named subset contributes to the ISP gradient. No weighting,
normalization, component clipping, teacher threshold change or alternative teacher is introduced.
Default/full NativePseudoTargetLoss behavior remains backward compatible with T018-A/B.
Original condition teacher evaluated once; detached boxes/int classes, existing score>=.5 stable
top20 selection. Fixed sampling seed20260912 on each native loss call. Global8D identity ISP,
K3/LR.1/EPS1e-12, existing current-phi CLIP gradient norm transfer; discard CLIP direction.
Four diagnostic evaluations apply exactly three updates. Empty supports preserve exactzero phi.
Source/CLIP frozen, no optimizer or accumulated gradients. Only functional episodic phi changes.

Reuse accepted native pseudo-target oracle/teacher/ISP/trust_radius/CLIP/corruptions and source
driver. Minimal edits: named native component sets; driver methods from config and active-key
histories; four-candidate summary/selection. Current Ours, detector loading, CLIP, oracle, ISP,
corruptions and adaptation code remain unchanged. Baseline on accepted release161116:
native_pseudo_target/native_confirmation/detector_native/trust_radius12passed2skipped1.57s.

## Fixed cohort and data boundary

Same readable train2017 pool and non-crowd positive-box/positive-area eligibility as T018-A/B.
Exclude B500 plus B's136 prior-source ledger (A100 and earlier36):636unique prior source IDs.
Also exclude all5000val IDs and the existing prior-evaluation ledger. The prior B manifest
and inherited H/A/ledger hashes are recorded, with every excluded prior source ID.
New fixed selection key: (sha256('20260920:'+decimalID),ID), first200 eligible images.
Split selected order into four consecutive50-image blocks before outcomes; no replacement.
Selection seed20260920 is independent of model/native sampling seed20260912, which is unchanged.
Persist ordered IDs,JPEG hashes,annotation hash,selection hashes,block assignments,exclusions.
Annotation eligibility is used during preparation only. Adaptation gets JPEG/image metadata
and detached teacher predictions. Load GT for official COCO evaluation after saving all predictions.

## Run and tests

Exactly no_adapt,current_ours,native_cls,native_conf,native_roi,native_conf_roi in that order.
Same gamma_s1/s2,contrast_s1/s2,color_cast_s1/s2,clean_s0; full200images, no early AP stop.
Expected1400teacherforwards,7000adaptiveepisodes,42predictionfiles and210officialevaluations
(sixmethods xsevenconditions xaggregate/fourblocks). A6000cuda:0,float32,threads1,
cudnn.benchmarkFalse; inherited model/weight/prompt/environment pins.
Tests: exact named keys/sums/gradients,detached unchanged supports,frozen state,onlyphi updates,
empty no-update,oldfull behavior,all existing relevant tests; selection thresholds/tie-breakers.
Focused tests then full regression then two-image CUDAK3 smoke withzeroAP. Smoke covers all
four candidates; compare returned totals with recorded active-component sum under float32
summation rounding. No scientific setting changes from smoke. Then one formal200 run.
Before/after source/CLIP hashes, per-episode source/frozen/eval/gradNone/ISP/support/finite checks
remain; save active perstep loss components,phi/gradients/norms,supports,pairedcounts andlatency.
Synchronized adaptation latency includes terminal diagnostics, excludes final prediction/state
checks/AP; teacher-inclusive adds original teacher forward. Retain same instrumentation convention.

## Frozen selection

Primary: six-corruption macro AP, mean of six official condition APs on0–100scale.
For each candidate all six eligibility criteria must pass: macro-current >=+.10AP;
>=3/4positive block macros; >=4/6positive conditions; macro>no_adapt;
clean-current >=-.10AP; no isolation/reproducibility blocker.
Report clean-vs-raw too. AP50/AP75 macros/deltas and all negative blocks/conditions retained.
If multiple eligible: take those within0.01AP of the highest eligible macro AP, then highest
macro AP75, then most positive blocks. These are the only performance selection criteria.
No per-condition candidate choices or extra coefficient search. Selected result is developmental,
not confirmation. If none qualify, close this component family pending research review.

Persist all code/config/plan/cohort/method pins/tests/raw command/environment/predictions/histories,
complete tables/report/hash manifest; append CODEX mailbox,commit/push,mirrorA6000. StopNEEDS_REVIEW.
No FCOS/SSD/COCOval,spatialISP,T017forensics,meta-training,predictor/gate/dose/mask changes or
deployment replacement. Every15minuteheartbeat monitors one existing run without duplication.

Cohort prepared before models:200from13162eligible,636prior-source/all5000val excluded,four50blocks. SHA256 534b17343ccba995beb9d112d552eefd7f1b2116479c2132dd99265b1d6735b4. Method pins include10unchanged modules andLF-normalizedhashes for3authorized new/modified modules.
