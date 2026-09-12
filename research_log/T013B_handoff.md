# T013-B completed handoff

2026-09-13T01:51:35.3833075+08:00 — NEEDS_REVIEW. Read research_log/T013B_report.md.
Planb73bcd2, PartA42072e7, preoptimizertrainmicrosetfc88531, PartBcode1685fb6.
Run20260913-014612-taisp-t013b-source-three-steps,release20260913-014535-taisp-t013b-source-smoke.
Started01:46:16+08,finished01:46:40+08,exit0.93tests passed10skipped5.94s.
Part A12episode gatePASS,median.9980674725590613,12positive,EXACT/FD~1,allfinite.
PartB exactly3outerSGDsteps1e-3,4train2017IDs65088,426525,541157,129068,8episodes.
Loss .474083306 -> .472119273 -> .475948031 -> .473895423 (nonmonotonic).
Cleanloss .355870174 -> .351741889;corrupt .592296437 -> .596048958 (worse).
Head gradientnorms.168584,.174204,.187236,.191116. Trunk0initial thennonzero;
finalparameterchange.000527524506; models/buffers unchanged, no NaN/Inf/full saturation.
Cleanphi3mean .071202720 -> .090419135;corrupt .019964728 -> .019958154.
PeakCUDA3712453632bytes; smoke15.003842537s. No detection/generalizationclaim.
All raw logs/32episodes/supports/4records/checkpoint under research_log/remote_runs/<run>.
Source microsetSHA164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493;
PartAJSONSHA8176e57c52fcfa67eebeb3e53008b903bc45c4d7f2657066ae65c534fe65a2a4 matchedremote.
Initialdata PermissionError repaired by selecting readable availablefiles; datasetunchanged.
No deployment/ISP/models/losses/initialization changes; T013A strictCUDAfailureuntouched.
No activeprocess/transfer. Next readLATEST/researchreview; don'trerun unchangedwork.
Do not automatically start T013-C,longertraining,schedule/architectureselection,val/FCOS/SSD,
newtargetcohort,spatialISP,gates/dose. Absolute remotevenv /home/liujianhua/wjq/TAISP/.venv/bin/python.
