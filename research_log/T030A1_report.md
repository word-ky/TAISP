# T030-A1 — BLOCKED; replacement integrity gate failed

R046/27fed9c completed the bounded8episode x5repetition numerical audit. Do not correct the T030-A stop assertion, rerun120candidates, loadannotations or claim scientific utility. Conditional continuation is NOT authorized because direct-phi parity and multi-output comparison fail. OriginalT030-A failure remains unchanged.

## Frozen scope and execution

Plan/manifest9ce9b5e before outcomes; diagnostic codea2844cd; unchanged firstfourT030images304815,131976,507312,517967 with the original clean/assignedcorruption pairs, no newselection. OriginalcohortSHA3ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752; diagnosticmanifestSHA5034a1cd910e6c5f5d8d532f663337fe4c5d6fe274b918f757795d45cfe79d27. Native samplingseed20260930; source/environmentsetup20260913; score>=.50/stabletop20, one detachedsupport generation perepisode shared by all5reps.

Run20260914-133102-taisp-t030a1-attribution8x5, release20260914-132745-taisp-t030a1-attribution, actualTAISP_SOURCE_REVISION=a2844cd, A6000CUDA0/torch2.4.0+cu121; all40reps completed in64.28838972502854s, exit0 with BLOCKED scientific-process status. Initiallauncher20260914-132943 leftmeta/run.sh only afterSSHfailure; status showedno tmux/process/train.log before retry. Bothlauncherreceipts retained; no repeated outcome run.

Eachrep uses a freshnativeforward graph: scalar-sumVJP; one engine call onfouroutputs/unitgradoutputs; fourcanonical reversecalls; fourreversed reversecalls. Separate imagecotangents are summed in float64; allfourcontract against the sameaccepted8columnfloat32ISPJVP/float64reductions. Then a freshfixedseednativeforward differentiates directly through ISP tophi, plus one freshunchangedhardgradient control. No cotangent shared acrossreps and no averagedcandidate gradient.

Allfourimagecotangents saved perrep in40torchfiles (sum/multi float32, separate/reverse float64); raw8Dvectors, canonical/reversecomponentvectors, losses, target/support hashes, before/afterRNG hashes and modelstate hashes retained. Oldpercoordadditivity check remains diagnostic only; passed0/40 here and original failedrun was not relabeled.

## Exact replacement gate

|Criterion|Observed|Decision|
|---|---|---|
|Native repeat cosine>=.99999 on8/8|min=0.9999990149641096|PASS|
|Native maxpairrel<=.002 on8/8|max=0.0014573985962872816|PASS|
|Freshdirectphi rel<=1e-5 andcos>=.999999 on8/8|relativepasses 0/40 reps; cosinepasses 40/40|FAIL|
|Allsupport/RNG/state/import/finite checks|all40pass|PASS|
|Medianmulti no worse than bothseparate orders on>=7/8|3/8|FAIL|
|Eachmedianmulti<=.002|max=0.0007143314010970546|PASS|

First failed replacement item is direct-phi/common-JVP parity (item3); everyepisode's worst fresh-forward relativeL2 exceeds1e-5. Item5 also fails. No new tolerance, seed, weighting or criterion was introduced after outcomes.

## 8-D comparison, all episodes

Each discrepancy is relative to g_sum; episode columns are medianover5except repeatability extrema and maximumdirecterror. Pairwise dispersion is maximum over allordered distinctpairs, denominator max(norm(first),1e-12), as precommitted.

|Episode / image / condition|d_native|min cosine native|max directphi rel|median multi|median separate|median reverse|multi no worse|
|---|---:|---:|---:|---:|---:|---:|---|
|0 / 304815 / clean_s0|0.000738633914|0.999999779494|0.000652192132|0.000309030087|0.000557861592|0.000526505002|True|
|1 / 304815 / gamma_s2|0.0010076157|0.999999497375|0.000895881752|0.000714331401|0.00053756714|0.000858345072|False|
|2 / 131976 / clean_s0|0.000578060513|0.999999834439|0.000676005814|0.000439192402|0.000346675339|0.000506779984|False|
|3 / 131976 / contrast_s2|0.000704979|0.999999876879|0.000684623733|0.000498088041|0.000499749903|0.000464215311|False|
|4 / 507312 / clean_s0|0.000656143843|0.999999796709|0.000434325457|0.000573203861|0.000455829628|0.000449345957|False|
|5 / 507312 / color_cast_s2|0.000290827287|0.999999968711|0.000297986548|0.000228525729|0.000380758925|0.000350784972|True|
|6 / 517967 / clean_s0|0.00029378275|0.999999962605|0.000497714779|0.000242966798|0.000205459348|0.000216786677|False|
|7 / 517967 / gamma_s2|0.0014573986|0.999999014964|0.000948638452|0.000543170173|0.000619427304|0.000596766284|True|

## Image-cotangent localization and hard control

Image discrepancies already occur before ISP contraction. These are median relativeL2 to c_sum; hardcontrol statistics use the samefivefreshreps.

|Episode|image multi|image separate|image reverse|d_hard|min cosine hard|
|---|---:|---:|---:|---:|---:|
|0|0.000484562658|0.000528742651|0.000519815953|0.000193783594|0.999999988638|
|1|0.000456590981|0.000534497481|0.000535910831|4.75291164e-07|1|
|2|0.000396735419|0.000463797145|0.000465256136|0.000317836774|0.999999949613|
|3|0.000434878112|0.000510532534|0.00050919229|0.000312170595|0.999999962168|
|4|0.000468520857|0.000537812|0.000535834886|1.22238089e-07|1|
|5|0.00039895705|0.000496465536|0.000497874544|2.16752294e-07|1|
|6|0.000417301395|0.000472134581|0.00047066263|1.93758014e-07|1|
|7|0.000436834596|0.000517805596|0.000534691009|2.29334442e-07|1|

Reversing separate-backward order changes the projectedvector bymorethan1e-5relative tosum in40/40reps (predeclared diagnostic threshold, not a gate). Fullimage-space order differences and all40perrep comparisons inper_repetition.tsv; fullprecisionepisode metrics insummary.json. Hardcontrol variability is nonuniform and is not used to rescue nativefailure.

## Deterministic probe and interpretation limits

Exactlyone episode0 attempt afteritsfive ordinaryreps with deterministic_algorithms=True failed withRuntimeError: CuBLAS requires a process-start CUBLAS_WORKSPACE_CONFIG for deterministic execution. Fullerror retained indeterministic_probe.json. OriginalFalse setting and warn_only=False restored immediately; RNG andsourcehashunchanged. Noenvironmentvariable, driver, kernel, seed or deployment setting changed. Probe is non-gating.

The receipts support upstream image-gradient variability across reverse sweeps and independently generated nativeforward graphs. The scalar-sum candidate met the separately frozen0.2%repeatability threshold, but did not meet the muchtighter fresh-forward directphi equality bound, and multi-output was not consistently better in8D. Therefore this audit does not establish that removing only the oldcomponentadditivity stop condition is justified. It does not isolate a single responsibleCUDAoperator or establish source-task utility.

## Integrity, tests and handoff

All40state_before/state_after hashes equal73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf; detectorfrozen/eval/gradNone, ISPstateidentity, target/RNGrestore andtransitiveimportchecks pass. All44protected/prior files unchanged locally and46remote hashes match. No annotation JSON orreference taskgradient; APcalls0. Numericalgate failure does not negate these separate isolation checks.

Focused16pass7.94s;full254pass11skip4warnings19.80s. ExistingNVML/protobuf warnings retained. AllmodelcomputeA6000CUDA; CPU onlyserialization/manifest/report math. SHA-verified rawarchive and everyfile verified, including40cotangent tensors retained inside the local project archive and expanded server run directory. GitHub contains all JSON/TSV/log/code receipts and the tensor storage/hash index; the 40large .pt tensors remain in the local archive and server project directory rather than Git objects. report/TSV/manifest are reproduciblefromthese frozen outputs.

Stop BLOCKED. No full120candidate rerun or correctedassertioncommit because R046conditions failed. Preserve T03060reservedcohort and originalfailedrecord. Awaitexplicitresearchdecision; no objective/support/component/seed/bounds/averaging/deterministic-mode/GT/AP change. Futurefreshcohort exclusions remainT030A_train_cohort.json(3011reserved/prior IDs plus5000val).
