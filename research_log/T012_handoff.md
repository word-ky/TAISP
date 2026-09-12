# T012 active implementation handoff

R0144afc07a/pointercbbc8d8 closes T011 and authorizes fixed .5 half-dose on T009.
Read T012_plan.md, T012_references.json, coordination/CHATGPT_TO_CODEX_R014_T012.md.
Pre-model plan d753456; kernel/runner aeef36b; report+prompt comparison fix dddc238.
**FORMAL RUN ACTIVE: 20260912-212537-taisp-t012-coco1000-half-dose**.
Release20260912-212421-taisp-t012-full, source3c82267 (code/report dddc238),
started21:25:48+08. Full93real-model/regression tests running, then1000image driver
and automaticreport. Exactcommand T012_full_run_meta.json. DO NOT DUPLICATE.
Repairedsmoke20260912-212148 completed21:22:15+08 exit0;9focusedtests2.30s;
14episodes189AProws7.205856s; report/plot/14support/126endpointchecks passed.

Full baseline90real/regression tests passed222.54s in run20260912-211448-taisp-t012-study-smoke,
then aborted before adaptation: prompt tuple vs JSON list comparison; content identical.
Minimal list conversion fixed, test covers real types. Failed run/archive retained.
Focused10real half/full/reference tests passed63.51s; local5half+full passed5.44s;
report/ratio/optional reference contrasts4passed10.51s. No scientific parameter change.

Core new adapt_half_dose shares private loop with unchanged adapt_clip_radius signature.
Half diagnostics add dose_coefficient, pre_attenuation_hybrid_norm, applied_half_dose_norm.
Full baseline path unchanged. run_t009 --config configs/t012.yaml --reference-study <T009>
only writes half predictions and reuses exact no-adapt/full metrics, verifies pins/IDs/config.
Fixed seed20260912,K3,lr.1,alpha.5,eps1e-12,source original threshold.5/top20, allpins.

Remote /home/liujianhua/wjq/TAISP/.venv/bin/python; existingGPUworks despite NVMLwarning.
Workflow D:/work/claude-autodl/autodl-workflow-clean with project .autodl/config.json.
Data shared/coco1000_t009. T009full reference runs/20260912-144439-taisp-t009-coco1000/artifacts/study;
T009smoke reference runs/20260912-143436-taisp-t009-study-smoke/artifacts/study.
T011full controls runs/20260912-195353-taisp-t011-coco1000-offline/artifacts/study.
T011smoke local research_log/remote_runs/20260912-194144-taisp-t011-offline-smoke/artifacts/study;
its report was generated LOCALLY, so do not assume remote analysis.json exists.

New scripts/report_t012.py --study <half> --reference <T009matching> --controls <T011matching>.
Checks refs/sample hashes, exact originalsupport, all halfalgebra/phi/reset, reusedendpoints;
produces AP_tables.csv,macro_tables.csv,paired_dose.csv,analysis.json,results.md,report_receipt.
Smoke report should run locally after fetching (has botholdsmokerefs); fullreport canrunremote
with abovefullrefs. New standalone scripts/plot_t012.py --study --controls, run locally,
no remote matplotlib installed. Do not add new environments.

Next monitor activeformalrun train.log and completion/report_receipt. Aftercompletion
fetch rawrun archive with SHA verification; locally plot and independently verify
AP arithmetic/criterion and paired diagnostics. Write T012_report.md plus
CODEX_TO_CHATGPT NEEDS_REVIEW, commit/push and mirror results to remoteproject.
Expected7000halfrows21predictionJSONs126newAP+252reused=378AProws across6groups.
Retain rawprediction/sample receipts and all negatives. Half/full phi and update normratios
are paired only atpositivefulldenominator; bothzero/nonzero-over-zero counted explicitly.
Allsteps0..3 diagnostics; ratiosofappliedsteps0..2. Halftrajectory neednotstay .5.
R014 fivecriteria allrequired; smoke unassessed. No alpha sweep/gate/newcohort/T013/meta.

Smoke archive SHA79ec8eb7f9a2a0f4c0f532a6bfcf0edab5e9eb4e0a5139e432f2383f8180e2ae.
Full report source dddc238 already committed; no further scientific changes.

Operationalhistory: initialdeploySSHmkdir timedout255; retry succeeded. Firstsmoke
90tests passed thenpincomparetuple/listfailure (fixed beforeadaptation). Fullversion
push timedout443 thenretry succeeded; fullrelease extracted butcurrentlink timedout,
verified fourfilehashes thencompletedlink/state only. All failures retained inprogress.
Latestfulltests outcome stillpending; do not claim93passeduntilrunlogconfirms.
