# T017-A pre-outcome task-component attribution plan

R028/f8fd9eb accepts T016-A and closes diagonal calibration. Commit this plan and
T017A_input_manifest.json before new component outcomes. Scope: exact32 sourceepisodes,
original masks/supports and source model; no new objective or adaptation experiment.

## Frozen provenance and reuse

Pin A1 records/cohort/supports/environment and T015 records with the inputmanifest.
A1recordsSHAa0a8c59dbb6b2fd83343f774755c6061a28542bce3c1d2f00b1458e8bb655a17;
cohortSHAa4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e.
T015recordsSHAae01cea586d0585b510293460ce174e2cf65350cc3d2ba3bd364c24b7de55b12.
SourceweightsSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384;
stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf.
Same16images,32episodes,clean/corruptorder,4blocks,corruptionreceipts andseed20260913.
Reuse oracle.detector_task_loss unchanged (it already returns differentiable components),
common_jacobian.common_reference (one8columnISPJVP set shared by all cotangents),
load_episodes/support_mask, reference_checks, state_hash/frozen_unchanged andsavedarray
decomposition/cosine/reporthelpers. No donor/model/deployment/ISP/loss changes.
Baseline common_jacobian+differential_subspace tests onunchangedT016release:
12passed1.88s. Add one analysis collector/attribution module andfocusedtests.

## GPU collection and independent total comparator

Use A6000 CUDA float32 detector/ISP, float64JVP reductions, normaldeterministicFalse,
cudnn.benchmarkFalse,CUBLASunset,threads1,exactinheritedenvironment. Detectorfrozen/eval
except theexistingoracle's native lossbranch flags; restored eval afteritsseededforward.
Atidentityfunctionalphi0, one seeded detectorlossforward yields exactlyfourkeys:
loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg. Preserve originalvalues.
Compute each component imagecotangent with autograd.grad retain_graph=True, then the
independent summed-loss imagecotangent anddirectglobalgradient fromsameforward.
No parameter .backward or modelupdate. Reuse one8columnJVP for allfivecotangents.
Currenttotal is the independent summed-loss cotangent reference, NOT defined as the
sum ofcomponentreferences; this keeps componentclosure a meaningful independentcheck.

For each global/object/background coordinate compare summedcomponentreferences to
independentcurrenttotal with float64 bound1e-12+1e-10*sum(abs(componentgradient)).
Allcomponent/total regionpartitionchecks useinheritedcommon_referencebound.
Summedcomponents vs savedA1totalreference must havecos>=.999999,relativeL2<=1e-5
for eachregion andconcatenatedobj/bg; independentcurrenttotal uses sameparitycheck.
Directglobalvs currenttotal usesinheritedsameparitycheck. Checkallfinite,originalmask
hash/rectangles,sourcefreeze/hash/gradNone andISPidentity/unchanged/gradNone.
Save numericalrecord beforeassertion; stoponkeys/closure/parity/isolationfailure.
Float64projection doesnotguarantee componentcotangentscomputedthroughseparatefloat32
detectorbackwards sum exactly to the totalcotangent; any resulting closure failure
is reported as a numerical blocker, with no tolerance/precision rerun inthis task.

## Attribution after all32 numerical records pass

Reuse savedT015pseudo d_p,s_p,fullnorm; no pseudo model/loss call.
For eachcomponent saveglobal,obj,bg,s,d, loss, differentialnorm,totalnormratio,
A_c=dot(dtotal,dc)/(norm(dtotal)^2+EPS), andall6pairwisecosines (nullatzeronorm).
dtotal=sum_c dc; groupsloc=box_reg+rpn_box_reg,conf=classifier+objectness.
A/group sums compared to1 accounting explicitly forEPSdeficit EPS/(norm(dtotal)^2+EPS)
plus1e-12+1e-10*sumabsattributions. Atzerototal allA=0, reportdegenerate, nofavorablelabel.
Save unadjusted sum-to-oneerror andEPSdeficit; no denominatorchange.
Ccomponent=2*dot(dc,dp)/(savedfullpseudonorm+EPS), groupanalogous; sum closure to
currentCdiff checkedat1e-12+1e-10*sumabscontributions. AgainstsavedT015Cdiff, use
predeclared1e-7+1e-5*sumabscontributions forinheritedfloat32CUDAreplaydifferences;
save exacterror/bound,currentandsavedvalues. No pseudo recomputation or componentrescaling.
ComputeallrawattributionsandCchecks before cohortinterpretation; save blocker iffailed.

Scopes overall/clean/corrupted/4fixedblocks/existingcases only. Mean/median/min/max,
positivecounts, validcosineandzero counts; allper-episodevectors/components retained.
Loc/conf dominant iff overallandcorruptedmedianA>.50 AND>=3blockmedians>.50.
Neither =>mixedheterogeneous. Dominantgroupuseful iff Cpositive>=20overall,>=10corrupt,
overallmedianC>0 and>=3positiveblockmedians. ExactlyR028triage; nofavorablemeanselection.

## Tests, report and stop

Synthetictests foroneforwardfourcomponents/commonJVP, independentcomponentclosure,
attribution/group sums, negative/cancelling/zerodifferentialcases, pseudoCclosure and
fixedtriageboundaries. Focused thenfullregression thenonefixed32episodeGPUaudit.
Maximum32detectorforwards,160imagecotangentbackwardrequests,256ISPJVPcolumns;
zeroCLIP/pseudoforward/optimizer/AP. Capture peakCUDA/timing andperrecordcounters.
Persistfullreport/raws/hashes,appendCODEXmailbox,commit/push,mirrorA6000.
StopafterT017-A NEEDS_REVIEW ornumericalBLOCKED. No newcohort/regionalobjective/
localizationloss/CLIP/spatialfinite-step/AP/FCOS/SSD/masksearch/calibration/training/
predictorredesign ordeploymentcodechanges.
