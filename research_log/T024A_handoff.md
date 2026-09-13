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

Candidate run:20260914-041922-taisp-t024a-candidates32 (9.267749s).
Reference run:20260914-042406-taisp-t024a-reference32 (17.862508s).
Raw receipt roots research_log/remote_runs/<run>/artifacts/candidates|reference.
Candidate raw archive SHA256:f7df34146aea45a2a0eb9856c8362376fad2a80aadf8e3e53ab6ea9b57adc2ce
Reference raw archive SHA256:71d4e3c02ab084b8fa8a73210c11f748cd74ce314cadee7f0b62138af0dd0fd0
