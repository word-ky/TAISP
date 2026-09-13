# T013-I pre-outcome linear capacity plan

R023/c016e51 accepts and closes T013-H. This plan is committed before any new
fit metric. Use only the accepted H records, cohort and audit; no new data/model
execution, optimizer, deployment edit or predictor update. H's failed39/64 remains failed.

## Inputs and fixed folds

H run20260913-081117-taisp-t013h-source-replication; report65707ec.
Verify file SHA256 against both these pins and the retained H artifact manifest:

- artifacts/collection/records.json:
  ac3d462521ad84e71af298eb57ae91b732f97e74fcf5336de91ab0967c41c2ab
- artifacts/collection/cohort.json:
  99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357
- artifacts/algebra/audit.json:
  ef31b0394eca3e1d4ee773380b65f2a1a11f37620b67d05c3bb77b2da6ba0a72

Exactly64 saved H16D/g8D records, existing order clean/corrupt pairs, four consecutive
16episodeblocks. Hold outoneblock, train other48episodes/24images; retainallfolds.
Reuse source_replication.fold_indices, cosine and predictor_factorization.factorize.

## Fit and references

NumPy float64 CPU. Compute train-only mu/gbar, X=Htrain-mu,Y=Gtrain-gbar.
Economy SVD X=U S Vt; retain s>eps64*max(X.shape)*smax. Minimum-norm
coef=V_kept*(1/s_kept)*U_kept.T*Y; B=coef.T. No ridge, scaling, feature/rank choice,
regularization or hyperparameter search. Retain all singular values, tolerance, rank,
B norm, raw largest/smallest singular ratio and retained-subspace condition number.
Zero singular denominator gives null condition plus an explicit rank-deficient flag;
zero prediction cosine is0. Non-finite fit/prediction is a blocker, no regularization.
Report norm and conditioning literally; describe any large observed coefficient/condition
without making a new outcome-dependent acceptance gate. No numerical kernel changes.

Heldout true residual r=Gtest-gbar. Linear rhat=(Htest-mu)@B.T; ghat=gbar+rhat.
Common-only residual estimate0, fullgradient ghat=gbar.
Inherited covariance residual estimate=(Htest-mu)@C.T. For its gradient-direction
cosine and delta, use this residual alone (no added gbar), exactly matching R022's
delta_cov=-.001*C(h-mu). Retain the distinction explicitly in records and report.
Check reconstructed covariance deltas against pinned H audit with its fixed8eps32
roundoff rule. No detector/ISP re-evaluation.

Everyepisode: r,rhat,ghat/direction, residual dot/cosine, fullgradient cosine,
SSE, residual energy, gdotdelta, delta, and r/rhat/ghat norms.
Summaries: pooled; eachfold; clean/corrupted; exactcases clean_s0,gamma_s1,gamma_s2,
contrast_s2,color_cast_s2; also family-only gamma/contrast/color_cast/clean (gamma severities
pooled) to remove ambiguity. Use pooledsum SSE and residualenergy for R2, never meanofR2.
Eachscope reports allthree references. Denominatorzero R2 is undefined/null andcannot
pass a positiveR2 gate; the real retained residuals are not discarded.

## Matched null

Exactly128 deterministic permutations from NumPy default_rng(20260913), iteration order
permutation0..127 thenfold0..3. For eachfold generate permutation(24) and map its sorted
train gradient imagepairs onto the unchanged feature imagepairs, preserving even/odd
clean/corrupt positions. No condition-family stratification, replacement, reseeding or
rejectionofidentity permutations. Feature/gradient marginals unchanged, heldoutdata fixed.
Use identical SVD/tolerance/centering for all512 fits. Retain everypairmapping, fitted
B, numericaldiagnostics andheldoutpredictedresidual/fullgradient arrays plus metrics.

Null distributions: pooledresidualR2, medianresidualcosine, positiveresidualdotcount.
Percentile rank=100*count(null<observed)/128 (strict empirical CDF; reportties separately).
Corrected upper-tail=(1+count(null>=observed))/129. Null95thpercentile uses
np.quantile(null,.95,method='linear'); gate requires observed strictlygreater than it.
No alternative quantile/permutation/seed tried after outcomes. This matched null is
descriptive on fixed source arrays, not detector performance or population certification.

## Gate and stop

Conjunction: pooledR2>=.10; positiveR2 in>=3of4folds; medianresidualcosine>=.20;
positive residualdots>=44of64, including>=20of32clean and>=20of32corrupt;
observedR2 ANDmedianresidualcosine strictly exceed respective null95thpercentiles.
Keep exactboundary semantics andreportallflags. Positive=>latertraining-dynamics task
may be considered byresearchlead. Negative=>current16Dfeature+linear-head branch has
not demonstrated sufficient heldoutresidualpredictability; reportclosure/researchpivot,
without choosing orimplementing a pivot. T013-H remains failed either way.

## Implementation and verification

Only new analysis/test/report files and research records. Baseline onacceptedremoteCPU
release, then synthetic SVD reconstruction/minimum-norm/rank cutoff, pair/fold isolation,
permutation determinism/marginals, pooledmetrics andstrict gate/tailboundary tests.
Focused and fullregression before one new saved-array audit. Use existing remoteCPU
environment (known localNumPy+torch OpenMP conflict). Retainexactcommand, revisions,
provenance/nulls/predictions/tests/report under research_log andmirror A6000 projectroot.
Append CODEX_TO_CHATGPT, commit/push andstop NEEDS_REVIEW. No newmodelrun, T013-J,
meta-training, centering/biasremoval/covdeployment, predictorredesign, regularizer,
targetAP, spatialISP, gating/dose orkernelwork without subsequentexplicitreview/task.
