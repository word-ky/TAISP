# T006 pre-run contract

R008 /06c1541, pointer864b0ee accepts/closes T005. T006 IN_PROGRESS.
Always read coordination/LATEST.md and its named continuation after mainmailbox.
Target FCOS ResNet50FPN COCO_V1 explicitly assigned; never choose based on results.
No meta-learning/T007, gating, stable/JS, spatialISP, smoothclamp or model updates.

## Reuse and experimental definitions

Reuse frozen FasterRCNN source, T005 det_pseudo fixed ROI loss/support selector,
unchanged8D ISP/functionalSGD, CLIP-global control, COCO IDs/corruptions, oracleloss,
COCOeval, environment/checksum, norm_match, report distribution/pairedbootstrap.
New analysis-only FCOS loader/native oracleloss + T006 study/report; no target
object or label arguments added to deployment code. No target calls in adaptation.

Same200IDs,sixcorruptions+clean,seed20260912,GPU0,threads1,phi0=0,lr0.1,K3.
Two adaptations in order global_generic,det_pseudo plus unadaptedbaseline.
Original source score>=0.5 descendingstable top20; CE original confidence weights,
fixedboxes/classes/no target; empty support exactzero update. No flip/stable/JS.
Source/target evaluate identical enhanced images at1/3, no target-specific ISP.
Raw and analysis-only one-step matched-to-contemporaneousCLIP norm for BOTH
objectives. No matching to annotated gradients. Clean gets same mechanisms as
corrupted cases but is reported separately. Source/target each fresh annotated
phi0loss/gradient once perimagecase, shared across variants; no cachedgradients.

FCOS uses installed torchvision0.19.0+cu121, FCOS_ResNet50_FPN_Weights.COCO_V1,
unchanged official pretrained transform and inference defaults. Exact enum/hash,
score/NMS/topk/center-radius/resize/normalization recorded inT006_fcos_pin.json
before full run. Initial Python download SSLcertificate error: retry with existing
system CA bundle SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; do not disable
TLS validation or change target. Availability still being checked.

Target oracleloss: only top model.training flag enables native FCOS annotated
loss branch; transform/backbone/head modules remain eval/frozen. Use native
classification/bbox_regression/bbox_ctrness sum, then restore eval. No target
loss/prediction/gradient influences support, adaptation, earlystop or step length.
Test targetexisting vsabsent source adaptation on realCPU (same exactrepeat
condition, avoiding known CUDAbackward nondeterminism); target forward hook
must fail if any targetcall occurs during adapt. GPUtarget gradients and frozen
weights/buffers validated separately with real model.

Report phi0cos(gself,gsrc),cos(gself,gtgt),cos(gsrc,gtgt), signed coordinate
products, Taylorpredictions, source/target lossesafter1/3 and normmatched1,
both/sourceonly/targetonly/neither beneficial fractions. Undefinedcos=nullraw,
explicit zero-coded overallpairedcos and rates retain fallback; validonlymeans
separate. No-adaptation pair has exactzero losschange by definition, and AP
computed onrawinput. No fabricated perimage AP intervals.

Report overallcorrupted, s1pooled3families, s2pooled3families, sixindividualcases,
cleanseparate. For imagebootstrap keepall chosen caseobservations perimage;
2000draws seed20260912,95percentile exploratory,no multiplicity adjustment.
AP/AP50/AP75 for bothdetectors before/after1/3, deltas vsbefore/CLIP, percase
and severity descriptive arithmetic means of3condition APs explicitly labeled
(not pooledCOCOAP). Raw percaseofficial AP remainsprimary.

2800adaptationrows (200x7x2),1400shared imagecases,70APevaluations (7x2detectors
x(before+2variantsx2steps)). Bothgradient vectors andlosses, percoordinatesigned
products saved. Mean phi/physical trajectories, saturation, basesupport/fallback,
meanconfidence, adaptation/setup time andallocatedmemory; sameprocessresident
source,target,CLIP, notminimal isolated deployment footprint.

Increment gates: baseline51realtests; targetloader/oracle andisolationtests;
2imagesmoke full endtoend andreport; fixed200run, no result-drivenchanges;
collectfullrawdata, paired/severity/cleanreport, NEEDS_REVIEW andcommit/push.

## Verified target availability and baseline

Baseline51realtests passed84.76s. FCOS download succeeds withsystemCA; one
intermediateSSHconnectiontimeout resolved onretry. SHA25699b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7.
Pinned officialdefaults:score0.2,NMS0.6,topk1000,max100detections,centerradius1.5,
resize shorter800/max1333,size-divisible32,mean[.485,.456,.406],std[.229,.224,.225].
InstalledFCOSforward source inspected:topmodel.training selectsnativecompute_loss;
no childtrainingflag needed. These settings fixed before smoke/results.
