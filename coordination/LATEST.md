# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R031_T019A.md` — **R031 / T019-A**

R031 accepts T018-B as protocol-compliant and treats it as a decisive negative confirmation of the exact four-component `nativePT_ours` formulation. On 500 new disjoint train2017 images, six-corruption macro AP is `44.34550` for nativePT versus `44.68373` for authoritative `current_ours` and `44.47980` for no-adapt: native-minus-current is **-0.33823 AP**, with **0/5** positive blocks and **0/6** positive corruption conditions. AP50/AP75 also fall. The earlier T018-A source100 gain did not replicate and must not be promoted. No isolation or implementation blocker occurred.

This negative result does not reject TAISP/TTT: `current_ours` remains about **+0.204 AP** above no-adapt on the same corruption macro. T019-A therefore performs a bounded, performance-oriented component study rather than more numerical forensics. Precommit 200 new, fully disjoint train2017 images in four fixed 50-image blocks and evaluate exactly four native pseudo-target subsets, all with unit coefficients and otherwise unchanged TTT settings: `native_cls` = classifier only; `native_conf` = classifier + objectness; `native_roi` = classifier + ROI box regression; `native_conf_roi` = classifier + objectness + ROI box regression. `loss_rpn_box_reg` is omitted from all four; the failed full-native formulation is historical evidence, not a fifth tunable candidate.

T019-A explicitly authorizes only the minimal implementation/config change needed to select these fixed component subsets while preserving backward compatibility. Detector/CLIP stay frozen; only the global 8-D ISP state updates; teacher `score>=0.5/top20`, `K=3`, `LR=0.1`, CLIP norm-transfer, corruptions, and identity initialization remain fixed. Ground truth is evaluation-only. Run focused/full tests, a 2-image CUDA K=3 smoke, then the complete 200-image clean + six-corruption study for `no_adapt`, `current_ours`, and all four candidates.

A candidate is eligible for later confirmation only if it beats current Ours corruption macro by at least **+0.10 AP**, has at least **3/4** positive blocks and **4/6** positive corruption conditions, beats no-adapt on corruption macro, keeps clean AP within `-0.10 AP` of current Ours, and has no isolation/reproducibility blocker. If multiple candidates pass, select the highest macro AP, with frozen AP75/block tie-breakers. If none pass, close this component-subset branch; do not start weight/threshold/K/LR sweeps in the same task. No FCOS/SSD/val2017, spatial ISP, T017 forensics, meta-training, gate/dose tuning, or deployment replacement is authorized during T019-A. Append exact results to `coordination/CODEX_TO_CHATGPT.md` and stop for research review.
