# T020-A NEEDS_REVIEW - fixed global linear gradient transport closed

2026-09-13T21:24:23.775106+08:00. R032/2654048 completed; plan766ae89, code0546b05, smoke31e0267.
Run: 20260913-203203-taisp-t020a-source200-crossfit.
Release: 20260913-202732-taisp-t020a-crossfit. Smoke: 20260913-202806-taisp-t020a-runtime-smoke.
Formal started20:32:11 and finished20:54:03+08, exit0. No active job.
200 new train2017 images/four50-image held-out folds; 836prior-source/all5000val excluded.
1400 paired source gradients, four Q fits on150images/1050episodes each, 2800 runtime
adaptive episodes,21predictionfiles,105official COCO evaluations. Model/gradient/runtime
work used A6000 CUDA; fit used CUDA float64; official AP aggregation used CPU.

Corruption macro AP: raw45.7285369463,current45.8688848296,transport45.8767699326.
Delta current +0.007885103017 AP, below frozen +0.15; delta raw +0.148232986343 AP.
Only1/4positive blocks and3/6positive conditions; both replication requirements fail.
Clean-current -0.027302685585 AP; clean-raw +0.230903596205 AP; clean criterion passes.
AP50 delta +0.017578982612; AP75 delta -0.014407043472 AP versus current.
Held-out mean gradient cosine decreases0.154443289188 to0.134915686288;
591/1388defined episodes improve (42.57925%);12zero-pseudo episodes retained,0zero-task.
No eligible development candidate. Close fixed global linear transport under R032.
Current Ours remains authoritative; no tuning/extra cohort or follow-on experiment.

Full168passed10skipped7.49s; focused16passed2skipped1.70s. CUDA smoke28K3episodes,
56normchecks passed without AP. Formal5600normchecks,2800episode isolation,1400paired
supports/fold exclusions and16actual remote release code hashes pass. Maximum relative
norm error1.57913324819e-7. Float32 source/runtime gp are not bit-identical (maxrelative
L2.002257067475,median9.94599722e-7,mincos.999999125542); disclosed, no rerun/tolerance change.
24empty adaptive episodes preserve identity. Frozen source/CLIP hashes unchanged.
76rawfiles57,905,525bytes fetched and SHA verified; archiveSHA
460f0c3692de64383b950f0780f098c9eddbade4717de571ae7ddec1aac9e7cd.
Report:T020A_report.md; full tables/matrices/alignment/receipt audit/hash manifest inT020A/.
Collection196.470625s, fit.040319s, runtime+AP1092.429241s. No scientific blocker.

Stop NEEDS_REVIEW. Keep15minute heartbeat; execute only a new explicit queued task.
T018/T019 negative branches and T017 blocker remain closed/preserved. No FCOS/SSD/val,
meta/predictor/spatial/gate/dose/coordinate/ridge/capacity experiments are authorized here.
