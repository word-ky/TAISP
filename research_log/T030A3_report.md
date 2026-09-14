# T030-A3 / R048 — scientific FAIL; NEEDS_REVIEW

The exact locked unweighted four-loss pseudo-native objective fails the original R045 conjunction and is closed without rescue tuning. Failed gate flags: ['S_native_overall', 'S_native_corrupt', 'Delta_overall', 'Delta_corrupt']. This is a local source first-order gradient analysis; no AP, K-step or cross-detector result is claimed. R045/R046 retain their original BLOCKED numerical verdicts, while R047 retains numerical PASS.

## Locked inputs, implementation and execution

R048 research860f4d5 reviewed f9ca28e5ab56375c06e63c97700b97574acba775. Input-lock descriptor and original gates committed ea67513 before annotations; reference code/tests committed d480de9 before the model-bearing outcome. Candidate runner59efa11b99648c74d952918b3de3bebfce6df64e, stored run20260914-151622-taisp-t030a2-candidates120. Candidate recordsSHA256 031731120f517c03bfb8a6a5ae595676ae26b6bb6ba6076cbcfde00be6a7bdb6; candidate rawmanifestSHA256 e5b019de2034f1697c5e630538869bf487674ca9cd06b10a3ed489bbdf6deac7. Original cohortSHA2563ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752.

Preflight verified all123candidate file hashes, 120individual/aggregate record equality and order, original cohort/corruption/JPEG manifests, reviewed/candidate commits, saved provenance, frozenstate, support/target equality andsupport hashes, finite8D gradients, saved RNG restoration and all7same-cotangent parity receipts. The preflight receipt is written before oracle/common_jacobian imports and before annotation load; oracle modules absent at that point. Candidate_all120_integrity true. The saved component-additivity diagnostic passes0/120 and is deliberately not a gate under R047/R048. No candidate/support/pseudo-target regeneration occurred; report independently compares allstored candidate vectors byte-equivalent as JSON values.

Promoted the preserved T030 reference draft into taisp.analysis.pseudo_native_reference; kept score/describe algebra and R045 thresholds, deferred oracle imports until lock verification, removed only the stale component-additivity prerequisite, and recorded per-reference RNG/state and exact consumed candidate vectors. All53protected/prior files unchanged, including candidate/deployment code and original drafts; all55remote hashes matched. No outcome-driven code change.

Reference run20260914-160040-taisp-t030a3-reference120, release20260914-155837-taisp-t030a3-reference, TAISP_SOURCE_REVISION=d480de9. A6000CUDA0, torch2.4.0+cu121, unchanged source-reference seed20260913 and native four-loss task definition. Exact command in rawrun.sh: python -m taisp.analysis.pseudo_native_reference --cohort research_log/T030A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-151622-taisp-t030a2-candidates120/artifacts/candidate --candidate-lock research_log/T030A3_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference".

Completed120episodes in87.01122627704171s. Frozen Faster R-CNN stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf; weightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384.

Annotation SHA256: 610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d. Labels were used only to construct the unchanged reference task gradient after candidate lock verification. S=<task,g>/(norm(g)+1e-12); Delta=S_native-S_hard. Exact-zero candidates retained. No deterministic/CuBLAS/kernel/seed/dtype/model change.

## Original R045 scientific gate

|Criterion|Observed|Verdict|
|---|---|---|
|1: S_native>0 >=80/120|74/120|False|
|2: S_native>0 >=45/60 corrupted|42/60|False|
|3: Delta>0 >=68/120|66/120|False|
|4: Delta>0 >=35/60 corrupted|32/60|False|
|5: median Delta>0 overall and corrupted|0.003484209199114012; 0.0042116080176258855|True|
|6: >=3/4 positive block medians|3/4; [-0.0005735732333550583, 0.01640573342034415, 0.0029184774353523753, 0.0023496265121768153]|True|
|7: clean median Delta>=0|0.003374409078447967|True|
|8: all integrity|120/120; state/RNG/JVP/lock pass|True|

## Overall, condition and block summaries

Counts are strictly positive, with zeros included in denominators. Cosines/norm ratios with zero denominators remain undefined rather than filtered from score distributions. Full means, medians, quantiles, zero and undefined counts are in summary.json.

|Group|N|S_hard positive|S_native positive|Delta positive|median Delta|mean Delta|median hard/native cosine|median native/hard norm ratio|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|overall|120|76|74|66|0.00348420919911|-0.0213316820955|0.764480593911836|0.614219398537154|
|clean|60|36|32|34|0.00337440907845|0.00429877503147|0.7284522465627947|0.7133707212584918|
|corrupt|60|40|42|32|0.00421160801763|-0.0469621392225|0.779229029374737|0.5674242291198535|
|clean_s0|60|36|32|34|0.00337440907845|0.00429877503147|0.7284522465627947|0.7133707212584918|
|color_cast_s2|20|11|14|9|-0.000694851350658|0.0259173605199|0.8164424653532776|0.6552914248877661|
|contrast_s2|20|19|16|10|-0.00390894370722|-0.164967703586|0.8339715997328896|0.5152955475094769|
|gamma_s2|20|10|12|13|0.0262938938323|-0.00183607460109|0.7414498647673923|0.5483225301560842|
|block0|30|19|18|14|-0.000573573233355|-0.0629243501707|0.7277536339698948|0.4003666836423958|
|block1|30|15|20|19|0.0164057334203|0.0678216218669|0.6875148123520767|0.9172304272943255|
|block2|30|22|21|17|0.00291847743535|0.00191771972989|0.7800643533670336|0.7236227594979338|
|block3|30|20|15|16|0.00234962651218|-0.0921417198082|0.7836733664030482|0.5890946493633158|

## Component diagnostics only

These component gradients are the stored diagnostic vectors from the immutable candidate lock. No component was recomputed, selected, weighted or promoted to a candidate.

|Group|Component|positive task score|median task score|median cosine to task|median norm|
|---|---|---:|---:|---:|---:|
|overall|loss_classifier|74|0.037265183003780164|0.2947956169965534|0.08255090885683981|
|overall|loss_box_reg|68|0.007177824929568947|0.12539137716357462|0.06657782950548669|
|overall|loss_objectness|64|0.009080507434136818|0.12979764854773898|0.014054992583742841|
|overall|loss_rpn_box_reg|72|0.015083372345891688|0.24704994241745654|0.007886606738990203|
|clean|loss_classifier|35|0.01937529731043592|0.2612532162472758|0.0719865538830398|
|clean|loss_box_reg|30|0.0006906242797112146|0.001178426881687334|0.06440635984447443|
|clean|loss_objectness|32|0.009080507434136818|0.15024948498967322|0.014054992583742841|
|clean|loss_rpn_box_reg|32|0.0036429824840567768|0.0961744067915467|0.009415232417157965|
|corrupt|loss_classifier|39|0.058741665255259|0.36893352898416265|0.0854517195790665|
|corrupt|loss_box_reg|38|0.024085507049440513|0.24617268748108484|0.07558241399223348|
|corrupt|loss_objectness|32|0.016969226639157765|0.11350853051384818|0.01393433562373032|
|corrupt|loss_rpn_box_reg|40|0.04382414970059477|0.42523783401751414|0.006481415134621624|
|clean_s0|loss_classifier|35|0.01937529731043592|0.2612532162472758|0.0719865538830398|
|clean_s0|loss_box_reg|30|0.0006906242797112146|0.001178426881687334|0.06440635984447443|
|clean_s0|loss_objectness|32|0.009080507434136818|0.15024948498967322|0.014054992583742841|
|clean_s0|loss_rpn_box_reg|32|0.0036429824840567768|0.0961744067915467|0.009415232417157965|
|color_cast_s2|loss_classifier|10|0.001599661178171743|0.21879771442328452|0.07458565953092902|
|color_cast_s2|loss_box_reg|12|0.007847223728957664|0.3394189462687385|0.07039271942946637|
|color_cast_s2|loss_objectness|8|-0.02221435240681056|-0.16303327576220025|0.009418132434894796|
|color_cast_s2|loss_rpn_box_reg|13|0.02230934379088788|0.4016113332351433|0.0063651288208396075|
|contrast_s2|loss_classifier|17|0.19411881378156803|0.6008039617776935|0.12996251296530992|
|contrast_s2|loss_box_reg|13|0.25246512071615623|0.5080018039988113|0.09488282612438668|
|contrast_s2|loss_objectness|16|0.12289388224342145|0.2575681434230613|0.01711611721512628|
|contrast_s2|loss_rpn_box_reg|14|0.27838067465376515|0.6463128199786001|0.014598427964602218|
|gamma_s2|loss_classifier|12|0.015891919959822743|0.16393499527770664|0.07470935393637577|
|gamma_s2|loss_box_reg|13|0.021073970937606795|0.12450630765029683|0.06325519898582568|
|gamma_s2|loss_objectness|8|-0.005763019677083224|-0.07220659284440835|0.01269768650593052|
|gamma_s2|loss_rpn_box_reg|13|0.01956138088921022|0.28757055721030544|0.005753117491774052|
|block0|loss_classifier|18|0.045760966484794524|0.39452049104425924|0.12961878924811393|
|block0|loss_box_reg|13|-0.009289208017148227|-0.12035242815757612|0.059992498021984406|
|block0|loss_objectness|15|0.003397365438718783|0.03558536868150206|0.011657317847875912|
|block0|loss_rpn_box_reg|15|0.0019394959902391518|0.022974135416518805|0.010712757359771176|
|block1|loss_classifier|19|0.04032981898514658|0.41384585848347794|0.0719865538830398|
|block1|loss_box_reg|21|0.027629604264043407|0.41904871661518617|0.06159331732548544|
|block1|loss_objectness|18|0.013926944322657755|0.24770105945850926|0.013756438686167011|
|block1|loss_rpn_box_reg|20|0.05630791234518492|0.34624280513836314|0.009173722412912505|
|block2|loss_classifier|19|0.06859048372891206|0.37496106961458064|0.06819121307193081|
|block2|loss_box_reg|19|0.02460943888293765|0.17862173794953662|0.07042627037035894|
|block2|loss_objectness|19|0.03401238592037145|0.1989881284240383|0.023489008640942163|
|block2|loss_rpn_box_reg|18|0.01601908825340062|0.2253616763936315|0.006875576591540871|
|block3|loss_classifier|18|0.01765981774306573|0.10699200030641012|0.09363820134864201|
|block3|loss_box_reg|15|-0.0022590368390069675|-0.07688658544244564|0.06242496920359501|
|block3|loss_objectness|12|-0.013610157334828184|-0.18925153501469533|0.013204491844500413|
|block3|loss_rpn_box_reg|19|0.013163425706064446|0.20488163301519963|0.00681620648055257|

## Tests, receipts and decision

Baseline11passed3warnings7.87s; focused12passed3warnings7.77s; full264passed11skipped4warnings25.03s, before annotated outcomes. Tests cover original gate conjunction/clean criterion/zero preservation, exact score epsilon, failed diagnostic versus mandatory RNG/nonfinite checks, and invalid lock stopping before oracle import. Existing NVML/protobuf warnings retained. Deployment155750SSH timeout preceded upload/modelrun; workflowstatus showed no job, deployment155837 succeeded. Only one annotated reference run was launched.

All120 references have frozen/eval/gradNone and restored RNG/state, accepted same-cotangent JVP/direct ISP parity. Candidate hashes still match the pre-GT lock after execution. Reference recordsSHA256 b492a6b2b3435dcef3dda0b7dd81a59811701d10d0a40a03b1ae037f1dda81a0; rawmanifestSHA256 149aaf824e04b1d40c35bfe4460f8ba0ccfc8ecd6d31e0b40ae9b09dc923dbdd. Raw120individual receipts plus records.json, summary, preflight, environment, logs, per_episode.tsv and manifest retained local/server/GitHub. Model-bearing compute onA6000CUDA; offline report/hash work onCPU.

Disposition: close_exact_four_loss_native_pseudo_family. Stop NEEDS_REVIEW, no activejob. No AP/K-step/FCOS/SSD, no component/support/seed/K-LR/CLIP rescue. Futurefreshcohort exclusions remainT030A_train_cohort.json(3011reserved/prior IDs plus5000val).
