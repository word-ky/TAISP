# T007 predeclared disjoint confirmation protocol

R009 accepts/closes T006. T007 directly authorized. No model execution on new
subset until its complete manifest (all JPEG hashes, annotation hash, zero
overlap) is committed and pushed. No T008/meta-training inferred.

Selection: official val2017 all5000 IDs, exclude the historical200 IDs in
shared/coco200/subset.json, sort remaining4800, random.Random(20260913).sample
200, then sort selected IDs. No outcome-dependent exclusions or replacements.
Copy identical historical official annotation file. Download selected JPEGs
and SHA256 exactbytes. New root shared/coco200_t007. Full manifest persisted
as research_log/T007_subset.json before any T007 model test/smoke/full execution.

Reuse T006 global8D ISP, hardclamp, zero phi0, lr0.1,K1/3, unchanged6corruptions
and clean. Source FasterRCNN COCO_V1 and target FCOS COCO_V1 pins unchanged.
CLIP prompts/preprocessing unchanged. Target/annotation analysis-only.

Three adapting variants: global_generic, det_pseudo, det_pseudo_clip_radius.
No_adapt shared baseline. Original source score>=.5 top20 support fixed.
Hybrid computes detector andCLIP gradients at each current phi, uses
g_det * ||g_clip||/(||g_det||+1e-12). eps=1e-12 numerical fixed. Zero detector
gradient/empty support => exactzero update, noCLIPfallback. No extra objective,
mask,clip,regularizer,predictor orlearnedgate. Hybridnorm relation includes eps,
so not claim strict equality for gradients comparable to eps.

Record step0/1/2 gradient vectors/norms/ratio/scale,phi,saturation,support,time;
step3 final phi/saturation/loss/time (no update gradient is applied at terminal).
Fresh shared annotated source/target gradients atphi0 and sameenhancedimages
evaluated bybothdetectors. Expect4200adaptationrows,1400imagecases,98APevals.
No_adapt AP shared perimagecase/detector, no fake repeatedbaseline adaptation.

Paired image-cluster bootstrap2000 draws seed20260912,95percentile exploratory
intervals, allchosen cases travel withimage; clean separately. Contrasts:
hybrid-rawpseudo, hybrid-CLIP, rawpseudo-CLIP. Cosine collinearity check,
not cosineimprovementclaim. Harmful raw tobeneficial hybrid FCOSloss1 flips
andreverse; strict harmful>0 andbeneficial<0, zeros reported separately.
Pre-update raw ratio strata fixed: [0,1),[1,2),[2,4),[4,infinity); zeroCLIPnorm
withnonzerodetector separate undefined/infinite bucket. Not deployment gating.

Report percase source/target AP/AP50/AP75 before/after1/3 andallcontrasts,
oracleloss1/3,benefit/jointfractions,cosine,phi trajectories,saturation,
fixedsupport/fallback,ratio/scale distribution,latency/memory; cleanprimary.
Severitygroups descriptive only; optionalmacro averages3condition officialAP,
not pooledCOCOAP. NoAP CIs. Allnegativeconditions/fallbacks retained.

Baseline T006 full54realtests102.35s already green onunchanged source.
Incremental tests: hybrid algebra/zero/episode/reset/support/model-freeze,
no target/annotations; datasetselection smoke usingfixture; two-image real
endtoend withreport, then fixednew200. No tuning fromsmoke/full outcomes.
