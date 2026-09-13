# T022-A2 IN_PROGRESS — R036 repeatability attribution only

2026-09-14 +08; research3e366b7, runtime7001d0c/release001534.
Reuse saved image160585 gamma_s1, original support from saved current row,
original empty-mask first trajectory states0..3 and exact T022A cohort.
No teacher/GT/AP/newcohort or method edits. Protected23 modules pinned in
T022A2_pins.json, including prior parity and failed edge checks.

B: identity plus saved empty-mask states1..3, 5independent pseudo/CLIP loss and
image-cotangent calls at each fixed image, all10repeat-pair metrics and raw tensors.
C: same4states, real/zero/one masks, fixed first B cotangents, 5regionalJVP calls
perstate/mask, allpairs; 20pure dose calls on one fixed tuple. No averaging.
D: fiveK3 repeats per current/realspatial/emptymask/fullmask/empty-support.
Use analysis-only loss/output hooks to retain actual runtime image cotangents,
not an alternative update implementation. Hooks returnNone and never modify
gradients; focused test proves intercepted cotangents and unchanged output.
Raw tensors chunked perrepeat, loss/gradient/state/image all-pair exact/maxabs/
relativeL2/cosine. Zero norms cosine undefined(null) unless bothzero(exact1).
A separate process enables deterministic_algorithms(True,warn_only=False) for
one representative cotangent computation, preserving full traceback/operator.
Do not enable deterministic mode in normal audit or performance evaluation.

No new pass tolerance. If fixed-input spatialJVP/dose varies, save blocker and
stop perR036 without patching. Otherwise finish all25runtime repeats and report
first observed variability and amplification scales; efficacy remainsopen.
Focused/full tests required; artifacts and reports committed/pushed/mirrored.
