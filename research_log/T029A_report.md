# T029-A — NEEDS_REVIEW; exact power-2 soft pseudo FAIL

R044/023f58b executed. Close the exact alpha=2 soft-sharpened target family under the frozen conjunction. No runtime replacement, CLIP norm transfer, K-step/AP, FCOS/SSD, target/validation evaluation or tuning.

## Cohort and outcome ordering

Plan80319b1 precommitted60fresh train2017 images, seed20260929, excluding2891cumulative source/memory/audit IDs throughT028 and5000val IDs. Existing GT-valid noncrowd/readable-image eligibility is used only during outcome-free cohort preparation. Selection sorts sha256('20260929:'+ID);60clean plus20each gamma_s2/contrast_s2/color_cast_s2. Four15imageblocks each contain5percorruption, with pairs kept together. No outcome-based replacement. Cohort SHA256 719259e31ae53a23cf03431a926c65a1c5643027f1bdbd2570251cb5a61f1676.

Candidate codea2bc921; all120candidate receipts committed/pushed4ec7df4 BEFORE reference code8ef7b3e and annotation load. Candidate receives only image manifest/JPEGs; no oracle/reference/source-meta/CLIP/memory imports transitively. Separate reference verifies123candidate files and cohortSHA before opening annotation JSON. Actual source revisions a2bc921 and8ef7b3e are present in the respective raw run metadata; no placeholder or provenance repair.

## Literal objective and gradients

One original-view teacher, unchanged score>=.50/stable descendingtop20 selection. Hard baseline is unmodified DetectorNativeLoss(det_pseudo). The new analysis-only soft objective directly shares hard.boxes and hard.weights (originalscores/scores.sum), so support ordering, boxes and confidence weights are identical. Raw support indices/boxes/labels/scores/hash retained.

Full91-way original-image fixed-ROI logits z0; q=softmax(2*log_softmax(z0)), detached, includingbackground. Loss is sum_i w_i[-sum_c q_ic logsoftmax(z(phi))_ic], natural logs. Empty support returns enhanced.sum()*0 for both. No foreground truncation, extra confidence weighting, new box loss, flip, memory, CLIP, spatial state or auxiliary loss.

Both identity image-cotangents contract with the same accepted8-column global ISP Jacobian (float32 ISP/model, float64 reductions); direct reverse/JVP parity retained. Post-lock source reference is unchanged original-view four-native-loss sum at seed20260913. S_hard/soft=dot(t_s,g)/(norm(g)+1e-12); Delta=S_soft-S_hard. Exactzero gradients are retained. Gradient cosine and soft/hard norm ratio are null only when the denominator is zero, with undefined counts explicit.

## Frozen gate

|Metric|Overall120|Corrupt60|Required|
|---|---:|---:|---|
|S_soft>0|63|31|>=80 / >=45|
|Delta>0|69|36|>=68 / >=35|
|Median Delta|9.992983323885152e-05|0.0002848683855208589|>0 both|
|Mean Delta (diagnostic)|-0.0017901247163606075|-0.0029124519934180795|not a gate|

Positive block medians 2/4 (required>=3); exact values {'0': 0.0020973842010735717, '1': -2.0474175291483945e-05, '2': -1.0215105305220323e-05, '3': 0.0006811471709311089}. Clean medianDelta=8.202294632908037e-06 (required>=0). Integrity passes; full conjunction FAIL. Positive delta counts/medians do not rescue insufficient positive task utility and block consistency. Hard positive utility counts=54/120overall,25/60corrupted.

All overall/clean/corrupt/family/block counts, fractions, means, medians and linear quantiles[0,.25,.5,.75,1] are retained for S_hard/S_soft/Delta/norms/ratio/cosine in summary.json; per_episode.tsv retains all120episodes. Overall/corrupted median hard-softcosine=0.9853268211046227/0.9890397189377251; median normratio=0.632134905772345/0.6190861047121623. Family heterogeneity is diagnostic only; no bootstrap or new significance criterion.

## Integrity, tests and execution

All120candidate support/target/isolation and reference checks pass. Zero hard/soft=2/2; empty supports=2. Max hard/soft/reference reverse-JVP relativeL2=3.726058014345905e-06/4.587448692443035e-06/2.371677316329512e-06; partition maxerror=0.0. 39protected/prior files unchanged locally, all44protected/prior/new hashes match remote release.

Detector remains frozen/eval/gradNone, state73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf before/after both jobs. ISP unchanged/identity/gradNone outside differentiated temporary state. Model work on A6000 CUDA0, torch2.4.0+cu121; CPU only manifests and report aggregation.

Tests: baseline9pass2.92s; candidatefocused10pass5.62s; finalfocused12pass5.84s; full243pass11skip4warnings15.43s. Initial test expectation used literal1.4 instead of the existing float32 scores.sum(), producing an assertion mismatch; corrected only that test expectation, then passed. No objective or tolerance change. ExistingNVML/protobuf warnings retained. One SCP download timedout and succeeded via existing workflow legacySCP retry; no experiment restart.

Candidate run20260914-113829-taisp-t029a-candidates120, release20260914-113707-taisp-t029a-candidate-tested, 25.996136983972974s. Reference20260914-114328-taisp-t029a-reference120, release20260914-114121-taisp-t029a-reference, 26.81723849097034s. Both exit0. Candidate archiveSHA be209bdaf44a5c429c5b8b871a68e86dad5c3344bc0bcd5784dab776e5288800; reference b60f287c79fc10fe53144b955062b495ad97a1fab171cdc9a476de2da484ea31, verified before extraction plus per-fileSHA checks.

Candidate recordsSHA c33f5cebd61fdbd6b9d6fcc369eccf4a709e18f73c4866c230e8d4a528da20ba; reference recordsSHA d8b47a0bcdce599c93a184aefb0718f9f4ead769704d9ff0a127eccd79a93afb. Original logits/fullq, supports/weights, hard/soft losses/gradients, source references and all environment/integrity receipts preserved.

Stop NEEDS_REVIEW. No alpha/temperature/classmass/foreground/weights/support/K/LR/hard+soft rescue or T029-B without explicit new research task. Future fresh cohorts use T029A_train_cohort.json as additional_source:2891prior+60new=2951source IDs, plus5000val exclusion.
