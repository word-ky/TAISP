# T034-A / R052 — FAIL; NEEDS_REVIEW

Oracle ceiling audit only. All240 H/C/N candidate records were committed before source annotations and task gradients. No candidate regeneration, learned mixer, finite ISP step, AP or deployment change.

## Frozen gate

| Check | Result |
|---|---|
| overall_HCN_ge_070 | True |
| overall_HCN_above_null95 | True |
| overall_E_ge_010 | True |
| overall_E_above_null95 | False |
| corrupt_HCN_ge_070 | True |
| corrupt_HCN_above_null95 | True |
| corrupt_E_ge_010 | True |
| corrupt_E_above_null95 | False |
| three_blocks | False |
| rank_ge2_90percent | True |
| integrity | True |

Rank histogram: {'0': 0, '1': 1, '2': 0, '3': 239}; rank>=2: 239/240; BASIS_COLLAPSE=False; zero tasks=0; blocks passed=0/4.

## Observed medians and matched null

| Group | N | C_H | C_HC | C_HN | C_HCN | HCN null95 | E | E null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 240 | 0.506733253 | 0.762286223 | 0.797013255 | 0.913276949 | 0.882886150 | 0.324511388 | 0.367286739 |
| clean | 120 | 0.494385155 | 0.722282430 | 0.795201611 | 0.903246640 | 0.885481044 | 0.340571778 | 0.371334510 |
| corrupt | 120 | 0.534683049 | 0.777676324 | 0.798490981 | 0.918466797 | 0.895866270 | 0.316095503 | 0.387397867 |
| family_gamma_s2 | 80 | 0.422513509 | 0.755989227 | 0.789208666 | 0.903812468 | 0.879316166 | 0.386164335 | 0.390708057 |
| family_contrast_s2 | 80 | 0.558738341 | 0.784299171 | 0.822447248 | 0.934357629 | 0.912424246 | 0.270148582 | 0.386571783 |
| family_color_cast_s2 | 80 | 0.475826267 | 0.735786944 | 0.791313507 | 0.907249676 | 0.885569122 | 0.348553208 | 0.401261765 |
| block0 | 60 | 0.513415003 | 0.797638814 | 0.801327585 | 0.930946503 | 0.903384686 | 0.346480407 | 0.429770390 |
| block1 | 60 | 0.458174584 | 0.728665414 | 0.799327811 | 0.892808843 | 0.890960580 | 0.320533420 | 0.376034808 |
| block2 | 60 | 0.530193477 | 0.786279593 | 0.810682793 | 0.929564285 | 0.904286942 | 0.290995348 | 0.381647382 |
| block3 | 60 | 0.483496143 | 0.693723519 | 0.789208666 | 0.909004309 | 0.896825617 | 0.352498666 | 0.403052984 |

Means/min/max for all nested ceilings and E, all256 null-median distributions for every group, all256x240 null H/HCN/E values, per-episode SVD bases/spectra/rank/tolerance and task vectors are retained in summary and raw records. Exact nonzero candidate normalization, epsilon task normalization, float64 SVD and linear null quantiles follow the precommit.

## Pairwise candidate cosine medians and zeros

| Group | H-C | H-N | C-N | Zero H/C/N |
|---|---:|---:|---:|---|
| overall | 0.042312931532249756 | 0.7684622073917542 | -0.09746804666076496 | {'hard': 1, 'clip': 0, 'native': 1} |
| clean | 0.0797394608594594 | 0.7803610136541025 | -0.005224343905197787 | {'hard': 0, 'clip': 0, 'native': 0} |
| corrupt | -0.05537819785028412 | 0.7415439257908286 | -0.12143039772347793 | {'hard': 1, 'clip': 0, 'native': 1} |
| family_gamma_s2 | -0.1090676347478936 | 0.7776939591321297 | -0.11370312851998066 | {'hard': 0, 'clip': 0, 'native': 0} |
| family_contrast_s2 | 0.08439021393235388 | 0.8524860184424461 | -0.03040661131122092 | {'hard': 1, 'clip': 0, 'native': 1} |
| family_color_cast_s2 | 0.04663359008477726 | 0.5368711297573436 | -0.1488071780404286 | {'hard': 0, 'clip': 0, 'native': 0} |
| block0 | 0.06692461857296872 | 0.8073802310147189 | -0.2041316564562594 | {'hard': 0, 'clip': 0, 'native': 0} |
| block1 | 0.14143467988179165 | 0.7007987788300345 | -0.17081596855503098 | {'hard': 0, 'clip': 0, 'native': 0} |
| block2 | 0.029083666822189012 | 0.726926113779113 | -0.07654828143247261 | {'hard': 0, 'clip': 0, 'native': 0} |
| block3 | 0.01818551818364731 | 0.7754133617599172 | 0.14144001931091554 | {'hard': 1, 'clip': 0, 'native': 1} |

## Integrity and execution

Cohort plan commit d14a1f31f63ee700eb0ab9432ed9a71d401351c0; candidate code 0f1ddd264d392e0536953d29a632a6382a3c5cf2; reference code committed5fc938974506cb03d7d3f503d579ebd3ff43b850 before reference reveal.
Candidate commit 05bf4b15f229a28c858df35d139e6d4f11c303d3; lock SHA cb2de0ff0b8d4edfd38a9618262ecedcd6f25013ae49f9fc504a3cd99cc2e769; reference process source revision 9ebe0fe87f16657eec256adc78f23d535857eafb.
Candidate run 20260914-203002-taisp-t034a-candidates240: 764.22210055904s; reference run 20260914-204507-taisp-t034a-reference240-null256: 181.98617894598283s. Model device cuda:0 / NVIDIA RTX A6000; CUDA 12.1; torch 2.4.0+cu121. CPU used for fixed NumPy SVD/permutation statistics only.
Every240 candidate and240 reference state/RNG/JVP check passed. Raw verified files {'candidate': 243, 'reference': 246}; 79 protected/new code pins unchanged. Detector state 73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf; CLIP state 6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c; CLIP weight a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f.
Native literal fullscalar is authoritative. Component-additivity is retained as a non-gating R047 diagnostic. Support boxes/labels/weights, pseudo targets, source/CLIP state and RNG receipts, zero flags and direct-reverse comparisons retained.
Candidate records SHA 7169646d4cf19232b69e364f6a0111ed531b335ae2a79c55610aed34c2e5e901; reference records SHA 25ce3eff29d725b59fe35f8a42fec29d9e53c9ef2201642c4eaf9bc8599e616b; null raw SHA f8fc7a4caad601291e6e106baacec91d73f3d6d724c5f807728069b8af2f26cf.

## Validation

Donor baseline:11passed1skipped5.75s. Candidate+native focused:9passed8.53s. Span/null math:3passed2.03s. Reference order+math:4passed4.75s. Full suite:292passed11skipped4warnings49.36s; receipt:research_log/T034A_full_tests.txt. Opt-in real-model tests skipped by default; actual240 candidate and240 reference episodes are the real CUDA integration.

## Decision boundary

Full frozen conjunction fails. Close the exact old hard/CLIP/native gradient-basis mixing direction. No basis expansion, nonlinear mixer, threshold changes or outcome-driven rescue. Stop NEEDS_REVIEW for a new research instruction.

Operational receipts: intermittent SSH/GitHub connection failures recovered; status confirmed the same candidate process continued without restart. Reference download made very slow intermittent progress; stopped only its matching SCP process44496 and the existing workflow legacy-SCP fallback completed. Both archive SHAs verified; no experiment rerun. Preparer import precedence repaired before cohort generation. No numerical protocol change or deleted recovery artifact.
