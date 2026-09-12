Active SSDcompatibility run20260912-142824-taisp-t009-ssd-compat,release142737,codeb492c9d; no duplicate.

# T009 in-progress handoff

2026-09-12T14:24:35+08:00. R011 acceptsT008;T009 frozen external-validation task active.
Plan92aa4df pushed; cohort preparation uses shared/prepare_t009_subset.py,
1000 images seed20260914,excludes historical200+T007200,five random-order blocks.
Download complete exit0; manifest committed e98a7fd before any model execution;
remote root shared/coco1000_t009. Full manifest research_log/T009_subset.json;
SHA256155bb6f047d374342623f48488bcb2b33601a4ecbd372976a433c1b289546e49.
Verified1000unique IDs,0/0overlap,5x200blocks,original selection order.
No T009model execution yet. SSD adapter/tests and K3-only driver implemented;
staticcompile passes. Real SSDcompatibility pending cohort manifest commit.
Local baseline3pass1skip10.74s; subset2pass.57s; localreplicationtest lacked
pycocotools(1failed/1passed),same4cohort/evaluator tests passed remotely1.52s
on release20260912-141922-taisp-t009-plumbing. No environmentchange.
Latestdriver/report notyetdeployed. Existing current remote release is plumbing.
After datasetdone: fetchmanifest,validate zero0/0overlap/1000/5x200/JPEGhashes,
commit/push. Then deploylatestcode and run tests/test_ssd_target.py with
TAISP_REAL_MODELS=1 TAISP_SSD_PIN_OUT=$AUTODL_ARTIFACTS_DIR/ssd_pin.json,
using the existing workflow. Test uses historicaldata;CPUhybrid3repeat/GPU
SSDrepeat/mapping/freeze/serialization/COCOeval. Commitmetadata pin as
research_log/T009_ssd_pin.json before 2image smoke/fullresults.
Freshfullregression70expected before full1000; smoke K3driver +report.
No deployloss/tta/model source changes,only analysis/data/targetevaluation.
Finalfullrunexpected21000samples,84predictionfiles,504APevals aggregate+5blocks.
Remotecommands/environment use T007handoff and workflow; no newtargetgradients.
