# T006 final recovery handoff

Updated 2026-09-12T10:06:42+08:00. T006 NEEDS_REVIEW, no active experiment. T001-T005 accepted/closed.
Read coordination/LATEST.md and named continuation after main research mailbox.
Current instruction R008/T006; no T007/meta/gates authorized.

Fullrun 20260912-092401-taisp-t006-coco200,release20260912-092347-taisp-t006-full,source/reportd7d0510,
driver8c0f480,target6429337. 54realtests102.35s,study1527.182541s,200images,
2800rows,1400paired imagecases,14groupsx200,70APevals; exit0at09:51:22+08.
Local root artifacts: research_log/remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study.
Remote: /home/liujianhua/wjq/TAISP/runs/20260912-092401-taisp-t006-coco200/artifacts/study.
T006_report.md and CODEX_TO_CHATGPT.md contain full conclusions/reproducibility.
No need rerun tests/study or regenerate report unless an actual change warrants it.

Target cosine -0.01162CLIP to0.08159pseudo,paired+0.093CI[.036,.152].
Target rawbenefit48.92to54.33%,paired+5.42ppCI[.58,10.25];matched56.17%,
paired+7.25ppCI[2.58,12]. Rawtargetmeanloss+.001474 vsCLIP+.000718,
pairedCIcrosszero;matched-.000434,paired-.001152CI[-.001803,-.000516],
absolute matchedmeanCIcrosszero. TargetAP3vsCLIP5/6better,vsbefore3/6better;
contrast_s2/color_cast_s1/color_cast_s2 negativevsbefore. Source s1/s2split
repeats buttarget+0.205AP3vsCLIPratio inboth,notseverity-only.
Source-targetcos.366CI[.325,.405],75.17%positive,bothrawbenefit31%vs23.67%CLIP.
Cleanpseudo phi3.07801vsCLIP.02877;sourceAP3+.310,target+.804,noidentitybehavior.
Interpretation limitedtransfer,notpure sourceonly orrobustrestoration;awaitreview.

Auditpassed allrows/groups/pairing, sharedsourceANDtargetoraclegrads/losses,
zero phi, same200IDs asT005, CLIP-only normreference, scoreweights/products,
allpredictions,pinnedFCOSmetadata. Fourfallbacks exactzero source/targetloss:
contrast_s2(2),color_cast_s1(1),color_cast_s2(1);no cleanfallback. Figureinspected.
Raw23544440bytes SHAeec9ff8b8a8c14442d0b4a5d25cd4fb896caca63b3e92892bd9f41a960f9ee70.
Rawuncompressed keptlocalremoteGitHub. Fullreceiptarchive31081757bytes
SHA1c121d5904a0df5dd5b1a72d3e658752dbbfa9343a8d69be83c3c95d75071553.

Operational details: baseline51realtests84.76s;target4tests20.60s;smoke53tests
100.40s and28rows16.280468s. Download CAfailure fixedsystemCA;oneSSHtimeout;
smokeSFTPstall recoveredlegacySCP;fullreceipt firstSSHclosed, retrysucceeded.
No TLSbypass/modelchange. Manual reportassembly firstcompared optionalcachepath
withrunmetadata; comparedcorresponding pinnedfields (allmatch). Windows GBKread
ofUTF8report failed; explicitUTF8resolved,no scientificdatachange.
Source pairednumbersdiffer slightlyfromT005 inknownCUDAnonbitwise setting;
report contemporaneous source/target,not historical substitution.

Remote root research_log mirrors summary/generatedartifacts. Avoid symlink tar
cross-device errors: transform research_log/remote_runs/ to actual runs/.
Finalcommit hash can be recovered from gitlog; report source revision is fixed.
