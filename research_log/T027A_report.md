# T027-A — NEEDS_REVIEW; E1 FAIL, E2 FAIL

R042 executed. Close the exact horizontal-flip gradient consensus/reliability family under the frozen rules. No AP, K-step, runtime integration, threshold/view/weight sweep or next-stage nomination.

## Protocol and ordering

Plan/cohort 887ad8a; candidate implementation and successful preflight c713818. All48 candidate records AND overall median agreement were committed/pushed in 0b1435b BEFORE original-task reference code ed444b6 and reference execution. Candidate process is GT-free; annotations enter only separate post-lock task reference. Cohort eligibility uses inherited GT-valid/readable-image criteria before outcomes, without outcome-based selection.

24 fresh train2017 images, seed20260927, four6imageblocks;24clean and8each gamma_s2/contrast_s2/color_cast_s2. Excluded2687 previous source/memory/audit images plus5000val IDs. Full image/JPEG/exclusion hashes are retained. No source memory, CLIP, feature/logit consistency or box geometry objective is used.

Preflight tested unchanged ISP flip commutation at identity and three fixed nonzero states on two synthetic shapes, including exact0/1 pixels, before teacher outcomes. All8 cases passed atol2e-7/rtol1e-6; max absolute error5.960464477539063e-08. No ISP repair.

Each raw original/flip view independently generates unchanged FasterRCNN pseudo supports: score>=.50, stable descending top20. Supports are frozen per view with no matching/filter/transfer. Unchanged det_pseudo gradients use the same global8D ISP coordinates, without remapping. Normalize each nonzero view gradient; symmetrically normalize their sum. eps1e-12; either zero view gives agreement=-1 and zero consensus, degenerate sum also abstains. No CLIP magnitude or confidence reweighting.

Accepted common8-column float32 ISP JVP with float64 reductions; unchanged original-view native four-loss task reference at seed20260913. S_orig and S_cons are dot(task_gradient,direction)/(norm(direction)+eps); Delta_cons=S_cons-S_orig. S_flip is diagnostic only. No flipped-task oracle.

## Frozen E1 and E2

|Metric|Overall48|Corrupt24|Required|
|---|---:|---:|---|
|S_cons positive|31|18|>=30/48; >=15/24|
|Delta_cons positive|24|10|>=30/48; >=15/24|
|Median Delta_cons|0.00026414250351889084|-0.0017481201179853005|>0 both|
|Spearman(agreement,S_orig)|0.1054059921841077|-0.12347826086956522|>=0.30 both|
|High minus low positive fraction|0.08333333333333326|-0.08571428571428574|>=0.20 both|
|High minus low median S_orig|0.006771710208289535|-0.07667775011610971|>0 both|

Positive Delta block medians: 3/4 (required>=3); exact medians {'0': -0.0005330728690374959, '1': 7.096707581558462e-05, '2': 0.0028905416039201187, '3': 0.01182354370520613}.
E1 conjunction FAIL. Overall/corrupted mean Delta=0.005701991108267321/0.009632317211148425; positive means do not override the failed count and corrupted-median conditions.

Candidate-only overall median agreement pinned at 0.6972085021016277; high>=this value, low<this same value for both analyses. No recomputed corrupted threshold. Overall high/low n=24/24, positive S_orig=16/14; corrupt high/low n=14/10, positives=10/8. Both-view nonzero coverage48/48 and24/24; zero consensus/abstentions0. E2 conjunction FAIL. Spearman uses average tied ranks; undefined correlation or empty groups fail without alternate cuts.

Full overall/clean/corrupt/family/block counts, fractions, means, medians and linear quantiles[0,.25,.5,.75,1] for all utilities/agreement, E1/E2 booleans, high/low distributions and coverage are in T027A/summary.json. per_episode.tsv contains every episode. No episodes dropped.

## Integrity and execution

All48 candidate isolation and reference integrity checks passed. Max candidate reverse/JVP relative L2=3.224507594536501e-06; reference=1.4527302423107363e-06; partition error=0.0. Empty original/flip supports=0/0; minimum view gradient norm=7.240080850723346e-05.32 protected/prior files unchanged locally;36 protected/prior/new code pins match remote. Raw receipt hashes verified after archive SHA verification.

Source frozen/eval, parameter grads None; source state before/after73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. All model-bearing runs on NVIDIA RTX A6000 cuda:0, torch2.4.0+cu121. Candidate 20260914-073543-taisp-t027a-candidates48 completed14.41063988499809s; reference 20260914-074105-taisp-t027a-reference48 completed18.390366662992164s; both exit0. Tests: baseline13pass3.33s, candidate17pass3.85s, finalfocused20pass3.98s; full227pass11skip4warnings10.17s. Existing NVML/protobuf warnings retained. CPU used only for manifests and report aggregation.

Raw archive SHA256: candidate04b35c87896ef0591b2c93cd99b002bdebeab59c8d82257f0a74ea9efddff450; reference899d8e47111fdf1eaa587acd8b98ecb02ad1000fb5a440701158da896934caca.

Stop NEEDS_REVIEW. Future fresh cohorts use T027A_train_cohort.json as additional_source:2687prior+24new=2711source IDs plus5000val exclusion. R042 accepted the earlier T025 numeric correction; historical data and research-owned coordination files remain unchanged.
