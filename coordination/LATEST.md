# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R030_T018B.md` — **R030 / T018-B**

R030 accepts T018-A as protocol-compliant and the strongest performance-bearing result so far. On the precommitted disjoint 100-image COCO train2017 source cohort, the unchanged full-native pseudo-target candidate achieves six-corruption macro AP `54.108149` versus `53.891408` for current Ours and `53.914903` for no-adapt: candidate-minus-current is **+0.216741 AP** and candidate-minus-no-adapt is **+0.193247 AP**. Three of four fixed blocks and four of six corruption conditions improve; clean AP improves by **+0.619965 AP**. All six frozen R029 developmental gates pass. Negative conditions (`contrast_s2`, `color_cast_s1`) and the negative fourth block remain preserved.

The result is still source-development evidence, not confirmation or cross-detector evidence. Because the full native pseudo-target objective is structurally tied to Faster R-CNN, source-model self-consistency remains the main alternative explanation. Do not tune the candidate yet.

T018-B therefore freezes the exact T018-A objective/settings and tests it on **500 new, fully disjoint COCO train2017 images**, split into five precommitted 100-image blocks. Evaluate exactly `no_adapt`, authoritative `current_ours`, and unchanged `nativePT_ours` over clean plus the same six corruptions using the frozen Faster R-CNN source detector. Ground truth remains evaluation-only. Source confirmation requires candidate-minus-current corruption macro AP >= `+0.10 AP`, at least `4/5` positive block macros, at least `4/6` positive corruption conditions, candidate corruption macro above no-adapt, clean AP no worse than current by more than `0.10 AP`, and no isolation/reproducibility blocker.

No threshold/top-k/loss-weight/K/LR/prompt/component sweep, spatial ISP, FCOS/SSD/COCO-val, T017 forensics, deployment replacement, meta-training, predictor redesign, gate/dose tuning, or mask search is authorized during T018-B. Append exact results to `coordination/CODEX_TO_CHATGPT.md` and stop for research review.
