# T013-D pre-outcome repeatability/common-mode plan

2026-09-13 03:27 +08. R018/b9e59da accepted T013-C; no longer training authorized.
Primary diagnostic: exactly12 no-update repeats, all8 fixed train2017 episodes,
all raw repetitions retained without selection or retries for unfavorable values.

Pins (SHA256):
- T013B manifest:164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493
- T013B supports:6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35
- T013C receipt:7c35fb9e5f7b1c923c06f625ddf806d443e763b56aa52d8027594f0b95cce943
- T013C original predictor checkpoint:a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120
- Faster R-CNN:258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384
- CLIP:a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f;
  modelopenai/clip-vit-base-patch32,revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.

Reconstruct initial predictor with seed20260913 and T013C constructor order:
source,CLIP,ISP,predictor; verify exact state equality with saved originalcheckpoint.
Every repeat deep-copies that same original state, torch.manual_seed20260913,
accepted K3/lr.1/eps1e-12, unchanged float32 model execution and fixed oracle seed.
Directly reuse saved pseudo supports (field equality); episode order65088clean/gamma2,
426525clean/contrast2,541157clean/colorcast2,129068clean/gamma1. No optimizer calls.
Use existing source_outer_episode/evaluate/geometry; model weights/buffers/.grad
and unchanged original/copy/ISP states checked. All raw losses,phi3,phi0/headgradients,
saturation/geometry retained perrepeat. Do not modify existing CUDA tolerances.

Summary conventions: population std (denominator12), raw min/max/range/mean,
maxabsolute deviation fromrepeat0. Per-episode phi3 coordinate stats and radius
stats; per-episode gradient cosines againstrepeat0. Aggregate clean/corrupt phi/head
cosines for eachrepeat and median/min/max/sign consistency, clean/corrupt normstats.

T013C effect comparison: use saved deltas only; never rerun optimizerprobes.
For all3probes and all3groups and all8episodes, compare abs(lossdelta) with new
no-update std/range and count abs(repeat-r0)>=abs(effect) among12 (includesr0,
report denominator explicitly). Compare saved probe phi3-basalphi3 coordinatewise
and radius deltas similarly; also group meanradius. Not p-values or confidence
intervals. Zero-scale ratios are null with raw effect/scale retained.

Algebra: reconstruct W=-1e-3*mean(T013Cheadweightgrad),b=-1e-3*mean(headbiasgrad)
infloat32asoriginalSGD, then evaluate retained features inCPUfloat64. Verify against
saved jointWh/b/output and record maxerrors. Split Wh into mean and centeredpart.
Record everytermnorm, centered/totalWh energy, pairwise distances, four same-image
distances, centeredSVD and numericalrank using default matrix_rank tolerance.
Exactly2diagnosticcounterfactuals: Wh only and (Wh+b)-mean(Wh+b). Record energies,
centered fractions/distances. No modelcalls/optimizer inthisstage, not deployable.

After primary12results, summaries and algebra are saved: attempt one optional
deterministic-kernel branch on the firstclean/corrupt pair using only
torch.use_deterministic_algorithms(True). Keep all other envsettings unchanged;
do not set CUBLAS_WORKSPACE_CONFIG, add fallback kernels, or relax errors. If it
raises, preserve traceback/operator and stopbranch. If successful, run that pair
exactly3times total, report exactness/differences. Restore flag after branch.

Baseline/affected tests precede outcomes; fullremote non-real regression thenone
bounded diagnostic. Report limitations and R018 interpretation. StopafterT013D;
no T013E/training/regularizer/biasremoval/centering/redesign/LRsweep/valtargetAP/newdata.
