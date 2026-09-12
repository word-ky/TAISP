# T008 predeclared offline proxy diagnosis

R010 accepts/closes T007. Analyze only frozen T007 run20260912-105119,
source0df7e13, report066197d. No model inference/adaptation or deployment edits.
Initial GitHub fetch timed out; retry synced0677233/2a8be77. Latest task is in
the main research mailbox, not a new continuation file.

Predictions: saved post-NMS, native-threshold, max100 outputs only. FCOS has
score floor0.2, so FP@0.05 cannot recover discarded candidates; report it on
retained outputs only. All other best-IoU/score proxies also refer to retained
outputs, not pre-NMS candidates. Absolute cross-detector FP levels are not
comparable because output policies differ. Do not rerun for missing candidates.

GT: official annotations from existing shared/coco200_t007; verify fullhash,
persist subset annotations for all200 IDs. Positive-area noncrowd GT as in the
T007 oracle loader; annotation ID ascending. Crowds excluded from denominator.
Unmatched detections are counted as proxy FPs, including possible crowd overlap;
these are not official COCOeval FP flags or perimageAP. Recordcrowdcounts.

At IoU>=.50 and>=.75 independently: stable score-descending prediction order,
ties retain JSONorder; greedily match highest-IoU unmatched same-class GT,
ties lowestGTannotationID. Unmatched detection is a duplicate if it overlaps
an already matched same-class GT at >=.50 (evaluated in the .50 pass).
FP counts among unmatched predictions at score>=.05 and>=.50, inclusive.

PerGT save class-aware bestIoU, class-agnostic bestIoU, highest same-class score
regardlessofIoU (thus class-presence confidence, not localized confidence),
zero if no eligibleprediction. Recall atboththresholds, TPscorelists atboth.
Image scalars: recall50/75,bestIoUmean/median,agnosticIoUmean,
sameclassscoremean,TPscoremean50/75,FP05/50,duplicates. NoGT => geometry/score
scalars undefined(null); noTP => TPscoremean null. Raw arrays and validcounts
retained; do not invent zero matchedscores or drop undefined cases silently.

K1/3 variantsCLIP/raw/hybrid vs sharednoadapt; hybridvsraw additional primary
contrast. For each, delta=new-reference. Positive direction for recalls/IoU/
scores, negative forFP/duplicates. For target andsource separately join matching
oraclelossdelta(new-reference) and classify strictlossimprove<0/worsen>0 against
strictorientedproxychange>0/<0. Ties andundefined separate. Also report loss
improves/proxyworsens fraction conditional onlossimprovement andunconditional.
Failure families: geometry(anyrecall50/75,bestIoUmean worsening),correct-class
score,FP(anythreshold),duplicates. Categories overlap,notforcedexclusive.
Matched-score changes are selection-sensitive; not evidence of calibration
or full dataset ranking alone. Same-image cross-detector coincident failures
use the samecontrast/K/imagecase and require each detector's ownlossimprovement.

Groups: allcorrupted plus each6condition andclean separately; allnegative
conditions remainvisible. Scale strata reuseT007pre-updatehybridratio bins
[0,1),[1,2),[2,4),[4,inf),clipzero onallcorrupted andclean,analysisonly.
Bootstrap2000 draws seed20260912,95percentile exploratory: resampleimageclusters,
allselectedcases travelwithimage; means/fractions weightedbyobservationcounts.
Conditionalfraction denominator lossimproved validobservations; undefined if0.
No AP CIs. APfromT007 reused asfixedofficial aggregate forcontext.

Determine failure branch from signed proxychanges, discordance frequencies,
pairedintervals andsame-image source/target agreement; don't picklargestnumber
acrossdifferentunits. Mixed/unclear is allowed. No rankingcausality claim from
these proxies alone, no newthreshold/cap/gate/localizationloss/T009/meta.
Synthetic matchingtests before fulloffline processing; alloutputs projectlocal.
