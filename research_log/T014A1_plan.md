# T014-A1 pre-outcome common-Jacobian replay plan

R025/c584887 accepts the old A run as BLOCKED, not a spatial scientific result.
Commit this plan before new real-model calls. The original run, assertion, report and
blocker remain unchanged; no old receipt is relabeled as passing.

## Frozen inputs and execution

Same T014A_source_manifest.json SHA a4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e.
Same H supportsSHA bf8f688741fcb4956c7ed40f52cc5f0a63d3030cad3ada968f1457a38c2cf747,
environmentSHA d0635da6c52a1f41a86181ace053c6c6fe1f38f342530b6fd9d806e8ee98cc42,
cohortSHA99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357.
Source weightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384,
stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf.
Old failed recordSHAffb8e0561043d6e5d1e9bbdd466fe53bc9b8192912660076a837ccdade52fcae;
old blocker.json SHA81c8a815f12c6d7026d2896db358888820896074a758fc0500d2a94d835c79c6.
Seed20260913, same source oracle samplingseed, corruption order andcachedpseudo supports.
Normalfloat32 CUDA detector/ISP, deterministicFalse,cudnn.benchmarkFalse,CUBLASunset,
threads1. No model/ISP/loss/deployment edits; only analysis/test/report changes.

## Common mapping and reductions

Extend the existing analysis-only identity_gradients helper with optional cotangent
return, preserving its default behavior and all old tolerances. For each objective,
execute its original source loss once and retain its image cotangent c, direct reverse
global gradient and old reverse object/background vectors/checks. Old reverse regional
discrepancies remain diagnostic, including failures under the old unchanged bound.

Use torch.func.jvp on the single existing functional mapping phi -> ISP(x.detach(),phi)
at float32phi0. Eight unit basis directions give eight exact autograd JVP columns Jk;
process columns sequentially and share each column between task andpseudo cotangents.
No finite differences and no detector call perregion. The JVP primal image must retain
the inherited identity tolerance atol2e-7,rtol1e-6. ISP inputs/mask/cotangents detached;
no ISPowned parameter gradients or changes.

Cast c,m,Jk to float64 before multiplication and reduction. Independently accumulate
sum(c*Jk), sum(c*m*Jk), sum(c*(1-m)*Jk); never derive one region by subtraction. Store all
three 8Dvectors, old three reverse vectors, closure/perity errors, scalar JVP column
norms, cotangent provenance andall frozen-state receipts. Float64 reductions run on
the same device; scalarvector geometry is NumPy float64CPU as before.

## Fixed numerical debug and conditional full cohort

Debug indices exactly[0,5,6] in thatorder. For bothtask/pseudo require:

- every coordinateclosure error <=1e-12+1e-10*(abs(ref_obj)+abs(ref_bg));
- direct reverse global vs refglobal cosine>=.999999 AND relativeL2<=1e-5,
  using norm(refglobal) as denominator (max(norm,1e-12) only forzero protection);
- inherited identity/sharedoutput andfrozen/sourcehash/gradNone/ISP checks pass.

Exact bothzero parity convention: cosine1 andrelativeerror0; ifonlyonevectorzero,
cosine0. No realnonzero vector is discarded. Reportoldreverse gradient check butdo
notgate this distinctnewestimator onit. Save eachdebugrecord beforechecking, andsave
Stage-C receipt. If anynewrequirement fails, STOPBLOCKED withno secondtolerance,
finite-difference/CPUdetector fallback, mask/image change or precision sweep.

Only ifall3debugrecords/sixobjectives pass: collectall32episodes fromindex0, asfresh
loss/cotangent evaluations. Do not splice oldorStage-C gradients into thefullcohort.
Everyfullepisode repeatsnewclosure,parity,image,isolation checks; stoponfailure.
Maxauthorized primaryscope3debug+32full=35episodes,70source-losscalls,280ISPJVPcolumns.
Debugrecords are numerical only; scientific geometry usesonlyfreshfull32 records.

## Science unchanged

Call unchanged spatial_action.geometry andsummarize onrefobject/refbackground vectors.
R024eps1e-12, masks/rasterization/strata/cases andallformulae unchanged. Gate:
overallmedianR>=1.20,corruptmedianR>=1.15; DeltaD>0 counts>=20/32 and>=10/16corrupt;
overallmedianDeltaD>0; >=3/4positiveblockmedianDeltaD. No threshold/mask/regioncountsearch.
Saveallrawvectors/numericalchecks andfullscope metrics; no finite-step/APperformanceclaim.

## Tests and stop

Baseline onacceptedanalysis release. Focused deterministic synthetic tests forJVP vs
reversegradient, independentfloat64closure, empty/full masks, detachedinputs andunchanged
ISP/state; newparity/closure boundary tests. Fullregression beforeStageC realreplay.
Persistplan/code/testlogs/exactcommands/debug/fullresults/report andSHAmanifest under
research_log; appendCODEX_TO_CHATGPT, commit/push andmirrorA6000projectroot.
StopafterA1/conditionalA rerunforresearchreview. NoT014-B/spatialdeployment/CLIPscaling/
training/redesign/FCOS/SSD/AP/newcohort/maskthreshold/regioncountsearch automatically.
