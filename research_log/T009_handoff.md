# T009 completion and archival handoff

Formal scientific study is complete and NEEDS_REVIEW. Read T009_report.md and
coordination/CHATGPT_TO_CODEX_R011_T009.md (R011); no T010/meta authorized.
Run20260912-144439-taisp-t009-coco1000 completed16:41:56+08,exit0;source69bfb66,
release20260912-144331-taisp-t009-full.1000images21000rows84predictions504APevals.
All raw receipt and derived arithmetic checks passed. Figure visually inspected.
Predeclaredexternalcriterionpassesbuttiny:FCOS+.070272,SSD+.032161vsnoadapt,
each4/5positiveblocks;vsraw+.089203/+.033084,each3/5positiveblocks.
Cleanphi3reduced54.263%but993/1000stillupdate. See fullreport fornegativeoutcomes.

## Remaining archival delivery, no scientific rerun

Active local exec session9887 downloads remote shared/t009_raw_receipts.tar.gz to
project .autodl/t009_raw_receipts.tar.gz, then verifiesSHA andextractsto
research_log/remote_runs/20260912-144439-taisp-t009-coco1000/artifacts/study.
Do not duplicate the transfer while active. Lastobserved~58MB/371572557bytes.
SHA346679ead129558154ddf40bfec25c5b6e370927076875cd718b44b84376b76d.
Archive contains immutable environment/subset/samples/predictions only;
finalderivedfiles alreadyseparatelydownloaded andhashverified.

At next heartbeat poll9887 orinspectfile/process ifsessionunavailable. Afterextract,
check samplesSHAcdbe9cf51feb17dc5ec2e60f0ea396227f4b4af301b2f6dc135f7382f0e8910c
against retained receipt. Run local scripts/audit_t009_raw.py ifneeded toverifydownloadeddata.
Samples.jsonl125507835bytes exceedsGitHub100MB; preserveoriginal locally/remotely,
create deterministiclossless samples.jsonl.gz (mtime0), andadd exactrawpathignore
likeT003. Checkindividualpredictionfilesizes beforegitadd. Commitpushcompressed
samples and84predictionfiles, updatearchive-deliveryreceipt. All scientificoutputs
alreadycomplete; do not rerunadaptation/COCOeval orchangeprotocol.

Remote root /home/liujianhua/wjq/TAISP; raw runs/<run>/artifacts/study.
Workflow D:/work/claude-autodl/autodl-workflow-clean, projectconfig.autodl/config.json.
Neverprintconfigsecrets. Projectreads/edits/gitfromprojectcwd; workflowfromworkflowcwd.
Mirrorfinalreport/derivedaudit/figure underremoteprojectresearch_log. No remotematplotlib.
Existing15minheartbeat canfinisharchivaldelivery; stayquietonunchangedprogress.
