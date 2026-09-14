# T031-A / R049 — scientific FAIL; NEEDS_REVIEW

The one global source-trained O(8) map fails the frozen holdout conjunction. This exact global orthogonal-transport family is closed without rescue tuning. This is only a first-order source gradient capacity audit. AP, K-step adaptation, cross-detector evaluation and deployment changes: zero.

## Frozen stages and ordering

Research c76b2e2; cohort/split/plan b019a89; candidate code 620076fe2d5530898ae4e69a9c2ed390c764994b; reference/fit code 14b5c68 committed before any task gradients. All480 candidate results committed d997b6152d1aef4875498f55179664748c17b183; candidate-lock descriptor 7569ed4.

Train360 references and R committed/pushed **6ee34b06813918a70c2a2022ed40985bbb65452b** before holdout launch; map-lock descriptor committed/pushed **bc8ee6df75bb1781ecba0be27a891ce76438b4e0**. The holdout process verifies this descriptor and every train/candidate file before importing oracle/common_jacobian or loading its annotation partition. Saved preflight: {"map_commit": "6ee34b06813918a70c2a2022ed40985bbb65452b", "map_sha256": "333e7dcdb0286cfd5957e68f0533c613ab71d1b68cadc614da58a641207f01ec", "map_lock_sha256": "c85720d731d5fe5f1c8c6ffd5fd98e1b4944da3c0566dd49c32656c26707cbf3"}.

240 fresh COCO train2017 images, seed20261001, excluding3011 prior/reserved IDs plus5000val and1400evaluation IDs. First180 images are train360episodes; remaining60 are holdout120episodes. Each contributes clean plus one frozen severity2 corruption; train60/family and holdout20/family, holdout four15-image blocks with5/family. All image pairs and hashes frozen before outcomes. Cohort SHA256 dc410b6afc5c28af84ed13809b300146d784aaf1cca4959e65115afcbe130123.

Preparation parsed the source annotation JSON only for outcome-free eligibility and to write separate train/holdout annotation files. Candidate generation used a manifest with no annotation paths and no oracle imports. Train reference loaded only the180-image train annotation subset. Holdout task gradients and metrics were first computed after the R commit; no holdout-based fitting, selection or retuning.

Candidate run20260914-165905-taisp-t031a-candidates480: 480 episodes, 296.4483901570202s. Train run20260914-170909-taisp-t031a-train360-map:360 episodes, 221.03978362301132s, source7569ed4. Holdout run20260914-171544-taisp-t031a-holdout120:120 episodes, 75.92225723603042s, sourcebc8ee6d. Train and holdout use identical reference release20260914-170250-taisp-t031a-reference. Exact commands, start/finish timestamps and exit status are preserved in each raw run.

All model-bearing calculations ran on A6000 CUDA0, frozen Faster R-CNN and unchanged identity ISP/common JVP, source-reference seed20260913. Only the single8x8 float64 SVD and summary algebra use CPU. No detector parameter updates; source state SHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf.

## Map and input locks

Candidate records SHA256 4a4736c99da7f7baf2c332d397133f449418e6fad27aa1c2b4d428a3fb7863d3; candidate manifest 8fb32640c416ca237762be805c26d0d66105aba6f0d11c0dfc2fc96cea4d891c; candidate descriptor 4123b543f05c7c6e4779ac11cc3dcc0c2ae2c398cc4b24c76a9b98dadfb4496e.

Train records SHA256 942a3bc20b154583317beeff02a2f8aab86dea366400d48f7b5b9f1bd184ee0a; train manifest 672f9cb035d3dae38e55f8f226a1e7acec7d3b688d053ad31c176fae91f96bb0; R SHA256 **333e7dcdb0286cfd5957e68f0533c613ab71d1b68cadc614da58a641207f01ec**. Holdout records SHA256 d2334c22651afc5f13679a481f86c623853120dd1c72a2f953c9a464b6949cc8; holdout manifest 69492c20e2044fe26d18d553e0025fbf6e3baf5caace1092ab145e446bbd12eb.

Exactly one full SVD of C=sum unit(t)unit(h)^T, R=U@Vt; full O(8), no determinant correction, bias, weights, ridge or normalization after transport. Train360episodes/180images; zero unit hard/task=0/0. Determinant=-0.9999999999999991; orthogonality maxabs=5.551115123125783e-16; singular values=[26.166257463138383, 13.552931309879014, 6.847862593522824, 3.71833339764002, 1.2896048060151475, 0.20985552725178408, 0.1090397614272763, 4.062327948267854e-13]. Smallest singular value is near zero; reported as a diagnostic, with no refit or determinant selection. Full C and R in T031A/R.json. Holdout maximum norm-preservation absolute error=3.552713678800501e-15.

## Frozen gate results

Thresholds: S_cal positive >=80/120 overall and45/60corrupt; Delta positive >=72/120overall and36/60corrupt; mean and median Delta strictly positive overall/corrupt; >=3/4 positive block medians; clean median>=0; all integrity checks.

|Gate|Pass|
|---|---|
|S_cal_overall|False|
|S_cal_corrupt|False|
|Delta_overall|False|
|Delta_corrupt|False|
|median_overall|False|
|median_corrupt|False|
|mean_overall|False|
|mean_corrupt|False|
|positive_blocks|False|
|median_clean|True|
|integrity|True|

|Group|N|S_hard>0|S_cal>0|Delta>0|Median Delta|Mean Delta|Median cos(h,t)|Median cos(cal,t)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|overall|120|70|72|53|-0.00400360891616|-0.0141990873033|0.1856214602493372|0.14241166557222706|
|clean|60|31|34|32|0.00076676711481|-0.00218693607768|0.03152470142317007|0.05797673763273764|
|corrupt|60|39|38|21|-0.0138594269454|-0.0262112385289|0.2904919186182825|0.15508903762546863|
|clean_s0|60|31|34|32|0.00076676711481|-0.00218693607768|0.03152470142317007|0.05797673763273764|
|color_cast_s2|20|10|11|8|-0.00516281516524|-0.0141707890654|-0.03227414055383116|0.025691324367436912|
|contrast_s2|20|17|13|3|-0.0227461014108|-0.0364474121995|0.6403881086416792|0.1681596506052993|
|gamma_s2|20|12|14|10|-0.00302908842578|-0.0280155143219|0.17792547151218574|0.21155370091939096|
|block0|30|20|20|13|-0.00991062817008|-0.040327616124|0.3462929495907471|0.2546946412031539|
|block1|30|16|17|12|-0.00444706061858|-0.0126337501996|0.20153191880999252|0.08891809415999935|
|block2|30|16|18|17|0.00307811505495|0.0104518669288|0.0427846496766328|0.24061882820858083|
|block3|30|18|17|11|-0.00790252865957|-0.0142868498185|0.16550534181812476|0.02753673960474748|

All exact-zero episodes remain counted. Holdout zero hard/cal=0/0. Full means, standard deviations, mean absolute values, RMS and energy shares for each of8coordinates, each condition and each block are in summary.json. Coordinate index is zero-based; energy share=sum g_j^2/sum_all_coordinates g^2.

|Group|Gradient|Max-energy coordinate|Max share|
|---|---|---:|---:|
|overall|g_hard|2|0.471963717346|
|overall|g_cal|2|0.40400487421|
|clean|g_hard|2|0.433521758721|
|clean|g_cal|2|0.487903587567|
|corrupt|g_hard|2|0.542789327447|
|corrupt|g_cal|1|0.347710240386|
|clean_s0|g_hard|2|0.433521758721|
|clean_s0|g_cal|2|0.487903587567|
|color_cast_s2|g_hard|2|0.422067176649|
|color_cast_s2|g_cal|1|0.491927583386|
|contrast_s2|g_hard|2|0.453566459907|
|contrast_s2|g_cal|1|0.359608158375|
|gamma_s2|g_hard|2|0.722399792269|
|gamma_s2|g_cal|2|0.367437487522|
|block0|g_hard|2|0.450802573762|
|block0|g_cal|1|0.406499227445|
|block1|g_hard|2|0.326407628186|
|block1|g_cal|1|0.243516029299|
|block2|g_hard|2|0.481378320951|
|block2|g_cal|2|0.423386741187|
|block3|g_hard|1|0.41354141466|
|block3|g_cal|1|0.479283127873|

## Integrity, tests and artifacts

All480 candidate and all480 reference integrity/RNG/state/parity receipts passed; all candidate vectors consumed unchanged. Raw file SHA checks: {'candidate': 483, 'train': 365, 'holdout': 125}. Train/holdout IDs disjoint; R committed before holdout; one fit only. All56 protected/prior files unchanged;58 candidate and61 reference remote hash checks passed. Existing detector/ISP/tta/deployment modules unchanged. No outcome-driven implementation edits.

Baseline9passed; candidate-focused6passed; reference/Procrustes-focused5passed; full271passed,11skipped in29.78s on A6000 environment. Known NVML/protobuf warnings unchanged. Focused tests cover exact column convention, O(8) reflection, one SVD, eps/zero handling, original hard candidate reuse, RNG/zero receipts, untouched import boundary and frozen mean/median/count gates. No new tests after outcomes because frozen code was unchanged.

Added analysis modules orthogonal_candidates.py, orthogonal_reference.py, orthogonal_transport.py and focused tests; report renderer scripts/report_t031a.py; frozen cohort/plan/code pins, input/map locks and raw runs under research_log. Complete per-episode scores: T031A/per_episode.tsv; distributions/coordinate diagnostics: T031A/summary.json; matrix: T031A/R.json; integrity: T031A/integrity.json. No labels in candidate files. Separate annotation subsets remain under remote project research_log/T031A with hashes in the cohort. No raw receipts deleted.

Transfer incident: first candidate SCP download stalled; terminated only that matching transfer process, existing workflow fallback completed. Archive and all483 raw file hashes verified; no experiment rerun.

Final disposition: close_exact_global_orthogonal_family. Stop NEEDS_REVIEW; no AP/K-step, runtime edits or alternative map family authorized. Future fresh-cohort exclusion should include T031A_train_cohort.json (3251 cumulative prior/reserved IDs).
