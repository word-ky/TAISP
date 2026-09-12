# T009 SSD compatibility receipt (in progress)

2026-09-12T14:32:00+08:00. Cohort manifest e98a7fd committed before model execution.
Initialrun20260912-142824-taisp-t009-ssd-compat,release20260912-142737,
codeb492c9d:1failed,3warnings,37.81s,exit1. Official SSDweight download via
Python urllib failed CERTIFICATE_VERIFY_FAILED(local issuer). No SSD prediction
or full-cohort scientific results were inspected; no substitute model selected.
Failed log retained under research_log/remote_runs/20260912-142824-taisp-t009-ssd-compat/.
System curl verified TLS and returned HTTP200 for the same official URL:
https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth
Downloaded142594222bytes in37s with normal certificate validation, no insecure flag.
SHA256 b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d
matches official b556d3b4prefix. Artifact in A6000projectshared/ and copied to
existing torch cache. No package/model/config/code changes for retry.
Retryrun20260912-143109-taisp-t009-ssd-compat-retry uses same release/code/test.
Compatibility completion/pin pending; no fullcohort evaluation yet.

2026-09-12T14:33:39+08:00: Retry PASSED:1passed,4knownwarnings,66.32s,exit0at14:32:24+08. Classmapping allCOCOcategories,score/boxserialization andCOCOeval,allfrozenparams/buffers,exactGPU SSDrepeat and exactCPUhybrid K3/source-support/enhanced-imageabsence/presence checks passed. Metadata savedT009_ssd_pin.json; committed before fullcohortresult inspection.
