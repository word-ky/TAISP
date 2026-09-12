# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are now appended directly to:

- `coordination/CHATGPT_TO_CODEX.md` — **R022 / T013-H**

R022 accepts T013-G as a protocol-compliant structural diagnosis. On the original eight source episodes, the current 16-D predictor features are measurably condition-sensitive (`median rho=1.4332015`) and the exact zero-head factorization contains substantial gradient/feature covariance (`r_C=0.4570599`, `r_C_out=0.5297501`), yet the full one-step initialization has only `0.170529%` centered output energy because much larger shared/common terms dominate. The centered signal is present and not cancelled, but the eight-episode effective rank is too small to justify bias removal, centering, covariance-only deployment, predictor redesign or longer training.

T013-H is a one-review-cycle source-only replication. Precommit 32 new COCO train2017 images, disjoint from the prior four-image source microset and all COCO-val/T002–T012 evaluation IDs, then collect exactly 64 zero-head source records (clean plus one balanced fixed corruption per image) without updating the predictor. Replicate the T013-G feature/factorization statistics overall and in four fixed blocks, then perform four-fold algebraic cross-fitting of the latent covariance residual using only saved `h_i` and `g_i=dL/dphi0`. No target/validation/AP work, optimizer, predictor/deployment change or method redesign is authorized. Stop after T013-H for research review.
