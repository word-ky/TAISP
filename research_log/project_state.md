# T024-A NEEDS_REVIEW — both ROI equivariance candidates FAIL
2026-09-14 +08; R039/29bd004. Plan3a7e764, candidatebf78bc5,
ALL candidate receipts committed/pushed9be51ee before reference638b763.
Fresh16train/32episodes, excludes1436prior+5000val. Both runs completed on
NVIDIA RTX A6000 CUDA12.1; CPU only lightweight preparation/reporting.
Feature: S_c19/32overall,10/16corrupt; Delta17/32,8/16; corruptedmedian
-.0004964978183184029;2/4positiveblocks. Logit:S_c19/32,11/16;
Delta16/32,7/16;corruptedmedian-.024653052599886323;1/4blocks.
Both fail original conjunction; close fixed ROI flip-equivariance family.
No zero/near-zero candidate gradients;32integrity pass,23protected unchanged.
21focusedtests/full204pass10skip. No AP/runtime/training/optimizersteps.
Report: research_log/T024A_report.md. No active job. Stop NEEDS_REVIEW.
Wait explicit next research decision; do not tune or run T024-B automatically.
GPU preference persists for all model-bearing future work.
