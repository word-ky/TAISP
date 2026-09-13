# T018-A pre-outcome full native pseudo-target performance plan

R029/53f8662 preserves T017-A numericalblocker and explicitly pivots toperformance.
Onefixedcandidate; no further T017forensics. Commit candidate/config/cohort before any
performance outcomes. Use implementation-from-baselines andexistingAutoDLworkflow.

## Frozen candidate and accepted baseline

Reuse select_predictions unchanged: positiveclass,score>=.5,stable descending score,
top20. One originalcondition-image teacherforward; both adaptive methods receive
the same detachedsupport. Candidate clones onlyboxes andintegerlabels, no scoreweight.
NativePseudoTargetLoss calls unchanged analysis.oracle.detector_task_loss onenhanced
image with these fixed pseudo targets. Exactlyunit-weight sum ofloss_classifier,
loss_box_reg,loss_objectness,loss_rpn_box_reg; nointercept/componentdrop/reweight.
SeedednativeRPN/ROI sampling uses20260912 on eachcall, sameexperimentseed asT009;
oraclefork_rng restoresoutsideRNG andevalflags. Allmodelparams frozen andbuffersunchanged.
GTannotations are never passed toteacher/support/adaptation; loaded onlyforCOCOeval.

Bothmethods call unchangedadapt_clip_radius: global8DISP,identityphi0,hardclamp/bounds
unchanged,K3,lr.1,EPS1e-12,gdet*norm(gclip)/(norm(gdet)+EPS),discardCLIPdirection.
Currentguidance staysDetectorNativeLoss(det_pseudo) withoriginal score-normalizedCE.
OnlyexperimentalnativePTloss isnew; no deployedmodule isreplaced. Emptycandidate
support returnsexactzero loss,phiunchangedandoutputexistingISP(phi0), matchingcurrent
semantics (numericalimageidentity,notbitwise rawpixelidentity). Fourterminaldiagnostic
evaluations atsteps0..3,exactly3updates, asauthoritativeT009; retainfourcomponentlosses.
Adaptation isepisodicfunctionalphi; ISP-ownedphi remainszero. CLIP unchanged/frozen.

SourceweightsSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384,
loadedstateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf.
CLIPopenai/clip-vit-base-patch32 revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268,
weightsSHAa63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f.
Original3positive/6negativegenericprompts andtensorpreprocessing unchanged; recordexact
environment_metadata and comparepins toT009receipt. No targetdetectors loaded.

## New developmental source cohort

Exactly100newCOCOtrain2017 images; sourceassets/annotationpath reusedfromT013-H.
Selectfirst100 bysha256('20260918:'+decimalID),ID amongreadablevalidsourceimages after
excludingall5000valIDs,ledgerallpriorvalidationIDs,andunionofprior4source+H32+A16.
T014/15/16/17reuseHsubset; exclusionmanifest preservesallpriorcohorthashes and36unique
sourceIDs. Eligibilitymatchesprevioussourcecollection:noncrowdpositivebox/area.
Persist T018A_train_cohort.json withorderedIDs,JPEGhashes,selectionhash,paths,
annotationsHA,disjointness and4consecutive25imageblocks beforemodeloutcomes.
No image replacement, resampling orcase-specificselection.
Conditions exactlyexistingCASES thenclean:gamma_s1/s2,contrast_s1/s2,color_cast_s1/s2,
clean_s0. Originalcorrupt() unchanged. Thisissource-developmentonly, notconfirmation.

## Implementation, tests and one run

Reuseexistinglossoracle,trust_radius,teacherfilter,CLIPloader,corrupt(),prediction_records
andreplication_ap. Smallnewsource-PILloaderdoesnotloadannotations; officialCOCO object
constructed for evaluationafterpredictionsarepersisted. Driveradaptation receivesonly
imageanddetachedsupport. No annotation-fed componentloss anywhere inadaptation.
Baseline onunchangedT017release: detector_native+trust_radius6passed2skipped1.40s.
Focusedtests: detachedteacher targets, exactfourkeys/unitsum/finite gradients,empty
support,onlyphiupdates,unchangedsourceflags/hash,unmodifiedcurrentcalls. CUDAK3
integration requiredbeforefull100. Fullregression beforeformalrun. A2image/7condition
runtime smoke may beused withoutAPaggregation; do notalter anyscientificsettingfromit.
UseA6000cuda:0 peruserpreference,float32existingmodelpaths,threads1,seed20260912,
cudnn.benchmarkFalse,normalCUDAsettings. No deterministic-kernel/precisionchanges.
Recordsource/CLIPstatehashbefore/after; checksourcefrozen/eval/gradNone/state equality
aftereachadaptation andsupportunchanged. Persistallstepgradients/phi/losses/supports.

## Exact evaluation and advancement

Exactlyno_adapt,current_ours,nativePT_ours on700imageconditions;1400adaptiveepisodes.
Save21condition/method predictionfiles beforeofficialCOCOeval. ComputeAP/AP50/AP75
foraggregate and4fixedblocks (105evaluations), using originaltrainannotations onlyhere.
MacroAP=simplemean of6corruptedconditionAPs. Reportpoints on0–100scale; cleanseparate.
Candidate-minus-currentmacro>=+.10AP ANDpositive, >=3/4positiveblockmacro deltas,
>=4/6positivecorruptionAPdeltas,candidatemacro>noadapt,cleanAPdelta>=-.10AP.
All6flags mustpass unchanged. No intervalorpopulationclaimfromdevelopmentcohort.
Diagnostics mean/medianphi3normclean/corrupt,cleanupdatefraction,stepnativecomponent
losses,supportcounts,pairedpredictioncounts,synchronizedadaptlatencywith/withoutteacher
setup,peakCUDA. Nativeandcurrenttimingincludeidenticalterminaldiagnosticconvention.

Persistplan/cohort/code/testlogs/runcommand/environment/predictions/samples/AP/gate/
report/hashmanifest; appendCODEXmailbox,commit/push,mirrorA6000. Stopforresearchreview.
No threshold/topk/weights/K/LR/prompt/ISP/dose/calibrationsearch,spatialISP,componentonly
variants,FCOS/SSD/COCOval,meta-training,predictorredesign,masksearch ordeploymentreplacement.

Cohortprepared beforemodels:100selected from13762eligible (13798readablevalid),excluded36prior-source andall5000valIDs. SHA256 a01dfb1d40a6daceddccc1b7aa7f3f2e74871fd4a511d8d6c9acf8e50c2c111f. Fourfixedblocks of25; fullmanifestcontainsallJPEGhashes.
