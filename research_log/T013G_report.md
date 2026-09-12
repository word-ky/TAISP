# T013-G final report — NEEDS_REVIEW

2026-09-13, R021/3a1f363. The offline saved-array audit is complete. The predeclared
triage is **mixed/common-term domination**: the frozen 16-D features respond measurably
to these input conditions, and gradient-feature covariance is substantial. The almost
shared output is dominated by large common terms, not by absence of all feature variation
or a negligible covariance contribution. This is an eight-episode structural audit,
not evidence of detector performance or a decision to redesign the predictor.

## Provenance, execution and tests

Plan [T013G_plan.md](T013G_plan.md), commit `21eaa31`, preceded implementation
`a7a5494` and outcome computation. The T013-C receipt, original/joint checkpoint files,
and six saved array groups were individually hash-verified. H comes from the saved
joint conditioning audit, G from the original explicit phi0 gradients, and the saved
per-episode head gradients and both forms of phi0 are retained. Original and joint
feature-trunk tensors are exactly identical. All file/array SHA256 values and JSON
field mappings are in the plan and [raw audit provenance](remote_runs/20260913-070237-taisp-t013g-offline-factorization/artifacts/audit/audit.json).

No model was reconstructed or executed for the audit; no detector/CLIP/ISP forward,
gradient recomputation, optimizer, new image, or performance evaluation occurred.
Small checkpoint tensors were only deserialized on CPU. All new algebra used NumPy
float64 from the saved arrays, with the original episode order unchanged.

Run `20260913-070237-taisp-t013g-offline-factorization`, release
`20260913-070203-taisp-t013g-offline-factorization`, source `a7a5494`.
Started 07:02:42+08, finished 07:02:52+08, exit 0. CPU audit computation recorded
0.008695731s (excluding interpreter/module startup). Python 3.12.12, NumPy 1.26.4.

- Local existing algebra baseline: **5 passed, 13.51s**.
- Local new focused run aborted at NumPy matmul. A minimal NumPy+torch example
  reproduced `OMP: Error #15`, duplicate `libiomp5md.dll`, with local NumPy1.26.4 /
  torch2.13.0+cpu. Both logs remain in T013G; no unsafe duplicate-runtime flag used.
- Existing server CPU environment: **8 focused tests passed, 2.29s**;
  **107 regression tests passed / 11 skipped, 5.18s**, then the offline audit.
  CUDA was hidden for these checks. This is CPU validation, not a real-model study.
- All seven real-artifact reconstruction comparisons passed the predeclared bound.
  No mathematical tolerance, formula, task threshold or saved artifact was changed.

## 1. The frozen features have non-negligible condition sensitivity here

`||mu|| = 0.3977067973`; total feature energy 1.3846080924;
centered energy 0.1192425194; centered fraction **8.612005%**.
Centered-feature participation effective rank is **1.6854135944**.
Singular values are approximately
`[0.2956373, 0.1673070, 0.06097193, 0.01120326, 0.002178159,
0.001171974, 0.000519514, 6.89e-18]`.

| Image / corruption | Condition displacement d | Median distance to other clean images | rho | Nearest clean is own image |
| --- | --- | --- | --- | --- |
| 65088 / gamma_s2 | 0.1372296054 | 0.0823915766 | 1.6655780975 | yes |
| 426525 / contrast_s2 | 0.0507376897 | 0.0823915766 | 0.6158116135 | yes |
| 541157 / color_cast_s2 | 0.1963596443 | 0.1124464786 | 1.7462498306 | yes |
| 129068 / gamma_s1 | 0.0906811734 | 0.0755157333 | 1.2008249064 | yes |

Median rho is **1.4332015019**, well above the fixed 0.10 weak-sensitivity threshold.
All four corrupted features have their own image as nearest among the four clean
features; this is descriptive neighbor geometry, not classification accuracy.
The low effective rank and heterogeneous corruption set limit generalization. We do
not average feature-difference directions or call them a universal corruption direction,
and measured displacement does not establish useful task-specific information.

## 2. The zero-head gradient factorization is consistent with saved artifacts

With `A = g_bar mu^T` and `C = mean((g_i-g_bar)(h_i-mu)^T)`:

| Quantity | Value |
| --- | --- |
| Frobenius norm A | 0.05781624365 |
| Frobenius norm C | 0.04867108510 |
| cosine(A,C) | 0.2572822233 |
| raw gradient cross term 2<A,C> | 0.001447973709 |
| r_C | **0.4570598744** |
| r_C_out | **0.5297500659** |
| float64 mean(g_i h_i^T) minus (A+C), max abs error | 6.9388939e-18 |

Both covariance ratios exceed 0.10, so the predeclared “small covariance despite
input variation” branch does not apply. Full H, Z, G, A, C, G_W matrices, Euclidean
and cosine matrices, gradient arrays and component outputs are retained in
[the raw audit](remote_runs/20260913-070237-taisp-t013g-offline-factorization/artifacts/audit/audit.json)
and [complete tables](T013G/tables.md).

The fixed elementwise bound was
`8 * eps_float32 * (max_abs_saved_tensor + abs_saved_element)`.
It accounts for retained float32 product/reduction/update rounding; passing it is
not a claim of bitwise identity. All errors and bounds were retained:

| Reconstruction | Maximum absolute error | Maximum error / element bound |
| --- | --- | --- |
| per-episode dL/dW = g_i h_i^T | 7.956641568e-9 | 0.01579694 |
| per-episode dL/db = g_i | 0 | 0 |
| saved mean head-weight gradient = A+C | 1.189885048e-9 | 0.02796657 |
| saved joint checkpoint W | 2.558472357e-12 | 0.07265921 |
| saved joint checkpoint b | 1.225271262e-11 | 0.07655889 |
| summed output vs saved float64 conditioning output | 1.321919664e-11 | 0.06543383 |
| summed output vs executed float32 phi0 | 1.628323801e-11 | 0.08170106 |

## 3. Shared energy dominates; centered covariance is present and not cancelled

Output components are `-eta*g_bar`, `-eta*A*h_i`, `-eta*C*h_i`, with eta=0.001.

| Component | Raw energy | Centered energy |
| --- | --- | --- |
| bias / common gradient | 1.690688908e-7 | 5.74e-42 (float64 roundoff; mathematically zero) |
| mean-feature term | 4.358394862e-9 | 1.286345467e-10 |
| gradient-feature covariance term | 4.184892837e-10 | 1.632458587e-10 |
| vector sum | 2.408641323e-7 | 4.107435022e-10 |

The raw pairwise cross terms are bias–A **5.348348847e-8**, bias–C
**1.158378967e-8**, and A–C **1.951079179e-9**. They are positive and must be included;
raw component energies cannot be added as if orthogonal. The centered A–C cross term
is also positive, **1.188630969e-10**; centered bias cross terms are zero up to roundoff.

Thus the centered A/C signal is **not destructively cancelled** in this saved update.
The shared bias term supplies the largest raw-energy term (its energy/full-sum energy
ratio is 70.192639%, a nonorthogonal descriptive ratio), and its positive cross terms
with A and C enlarge the total. The resulting centered/full fraction is only
**0.170529127%**. The covariance contribution is substantial relative to the centered
signal but small relative to the dominant common output energy. This identifies how
the relative signal is overwhelmed without treating the decomposition as causal
identifiability or justifying bias deletion alone; T013-D already found Wh itself
predominantly common-mode.

## 4. Exactly two algebraic counterfactuals

| Diagnostic | Centered energy | Four same-image output distances, in fixed image order |
| --- | --- | --- |
| common-only: -eta*(g_bar+A*h_i) | 1.286345467e-10 | 6.521667687e-6; 2.517830270e-6; 9.545074676e-7; 4.331391906e-6 |
| covariance-only: -eta*C*(h_i-mu) | 1.632458587e-10 | 2.576800955e-6; 8.889469362e-7; 9.059480047e-6; 1.591419737e-6 |

Complete pairwise distances for each counterfactual and each original component are
in the tables. These outputs show nonzero sample-specific covariance variation in
the retained arrays. Neither output is a proposed deployment parameterization, and
neither was evaluated for loss/AP. No inference is made from T013-F's pooled improvement.

## Delivery and stop

Only `taisp/analysis/predictor_factorization.py`, synthetic tests,
`scripts/report_t013g.py` and project records were added. No predictor/model/ISP/tta/
deployment code changed. [Exact run command](remote_runs/20260913-070237-taisp-t013g-offline-factorization/meta.json),
[test/run log](remote_runs/20260913-070237-taisp-t013g-offline-factorization/train.log),
[completion](remote_runs/20260913-070237-taisp-t013g-offline-factorization/artifacts/audit/completion.json),
raw audit and [SHA256 artifact manifest](T013G/artifact_manifest.json) are retained.
The run has five files, 140,370 bytes. The renderer uses only Python's standard library:

```powershell
D:\anaconda3\python.exe scripts/report_t013g.py --audit research_log/remote_runs/20260913-070237-taisp-t013g-offline-factorization/artifacts/audit/audit.json --output research_log/T013G/tables.md
```

**NEEDS_REVIEW.** Fixed triage: medianrho>=0.10 and both covariance ratios>=0.10,
with common-mode dominated full output => mixed/common-term domination. No redesign
is selected from eight episodes. No T013-H, longer training, regularizer, bias removal,
centering, new feature backbone, new data, target/AP, spatial ISP, gating, dose or
deterministic-kernel work started.
