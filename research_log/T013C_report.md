# T013-C source-gradient conflict and image-conditioning audit

**NEEDS_REVIEW. The bounded diagnostic completed; no further training is active.**
The measured clean/corrupted aggregate gradients oppose each other, and the first
joint predictor output is dominated by a shared offset. However, the finite-step
probes do not show the predeclared two-way cross-harm pattern. Their loss changes
are poorly predicted by the FO linear approximation, and a no-update CUDA repeat
has material loss variation. These observations do not isolate one causal bottleneck.

Pre-real plan **cdd5c58**, analysis code **68a6b07**. Run
**20260913-030301-taisp-t013c-conflict-conditioning**, release
**20260913-030230-taisp-t013c-conflict**. Started03:03:05+08, finished03:03:31+08
on2026-09-13, **exit0**. Full regression **95 passed,10 skipped,6.28s** precedes
the real diagnostic; focused tests14passed/1skipped16.99s, baseline7passed15.42s.

## Frozen setup and execution

Exactly the T013-B four train2017 images (65088,426525,541157,129068), each clean
then its fixed gamma_s2/contrast_s2/color_cast_s2/gamma_s1 episode. Manifest SHA
164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493 and T013-B saved
support SHA6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35 are
asserted before execution. Supports are loaded directly and reconstructed fields
match the saved float32/integer values exactly. No support inference, image
selection, data download or validation/target cohort is used.

Seed20260913 and original T013-B construction order (source,CLIP,ISP,predictor)
reconstruct the fresh zero-head initializer; the trained T013-B checkpoint is not
loaded. Its original state is saved and remains unchanged. Three independent
deep copies each receive exactly one SGD step at1e-3 using the mean joint,
clean-only or corrupted-only head gradient. Trunk gradients are exactly zero
at this initialization, as verified for all8 episodes. The copies' trunks remain
identical to the original. There are **three optimizer steps total**, never three
sequential steps on one predictor.

The accepted K=3/lr=.1/eps1e-12 full hybrid, ISP, predictor architecture, prompts,
losses, deployment signatures and T013-A failed strict CUDA assertion are
untouched. Ground truth is accepted only by the reused analysis outer objective;
predictor input is episode RGB. All detector/CLIP parameters/buffers stay unchanged,
all model `.grad` fields remain None, and model submodules return to eval after
the existing oracle loss branches. All recorded gradients/results are finite.
ISP-owned phi stays zero. Baseline plus3probes plus1no-update repeat retain40
episode measurements; only the initial8 require predictor-gradient computation.

## Stage B: strong aggregate opposition, heterogeneous individual pairs

| Gradient space | Clean/corrupt cosine | Dot product | Clean norm | Corrupt norm | Norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit phi0 (8D) | **-.747507938** | -.023812659 | .090474579 | .352099579 | .256957364 |
| Flattened head W,b (136D) | **-.726191498** | -.028621645 | .098302461 | .400939679 | .245180175 |

The corrupted aggregate is about four times larger in norm. Same-image
clean/corrupted phi-gradient cosines are **+.308034562,−.342674728,−.516680954,
+.588911657** in fixed image order; not every pair conflicts. Aggregate phi
coordinate signs agree on only1/8 coordinates; head-coordinate signs agree on
42/136. Full8x8 cosine matrix, all8 gradient vectors, weight/bias gradients and
their unnormalized scales are in [complete tables](T013C/tables.md) and raw JSON.

| Phi coordinate | Clean gradient mean | Corrupted gradient mean |
| --- | ---: | ---: |
| gamma | .000672155 | -.081828785 |
| red gain | -.029477930 | .155875026 |
| green gain | -.052913807 | .091714376 |
| blue gain | .020283684 | .107892118 |
| contrast | .022612300 | -.017384031 |
| brightness | -.059547417 | .268784753 |
| tone | -.001087925 | .015263765 |
| sharpening | .006823203 | -.012184962 |

Head bias gradients equal the explicit phi0 gradients through the linear head.
Per-episode W-gradient norms range .003472776–.777623296 and bias norms
.009132471–1.560622692. No per-episode normalization is applied before averaging.

## Stage C: finite probes do not establish cross-harm

Predictions use **head parameter space**, matching the actual SGD update:
`predicted_delta_i = -1e-3 * dot(g_head_i, mean_group_g_head)`.
Common baseline means are clean **.356072162**, corrupted **.600125877**, joint
**.478099019**. All after-values, components, states and every positive/negative
per-episode delta are retained in [complete tables](T013C/tables.md).

| Independent probe | Predicted clean delta | Actual clean delta | Predicted corrupt delta | Actual corrupt delta | Actual joint delta | Sign agreement n8 | Pearson r n8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint | +.000009479 | -.003568106 | -.000066065 | -.006689828 | -.005128967 | 3/8 | .332738169 |
| clean-only | -.000009663 | -.002963094 | +.000028622 | -.000711638 | -.001837366 | 6/8 | .032156512 |
| corrupted-only | +.000028622 | -.006151966 | -.000160753 | -.000044076 | -.003098021 | 3/8 | -.698274083 |

All three measured group means improve relative to this baseline; consequently
the R017 combination **opposing gradients AND cross-harm in both group-specific
probes is not met**. The FO predicted joint direction would slightly harm clean
and improve corrupted loss, which is opposite the T013-B three-step asymmetry;
the current joint probe instead improves both. It is not appropriate to claim
that this local linear model explains T013-B's clean-improves/corrupt-worsens result.

After-loss means (clean/corrupt/joint) are joint
.352504056/.593436049/.472970053; clean-only
.353109068/.599414239/.476261653; corrupted-only
.349920196/.600081801/.475000999. These are source microset diagnostics, not AP
or evidence of generalization. Correlations are descriptive, n=8.

| Probe/group | phi0 mean / max | phi3 mean / max | Saturation mean / max |
| --- | --- | --- | --- |
| joint clean | .000174265 / .000179100 | .099900238 / .190356091 | .018635253 / .047591146 |
| joint corrupt | .000172558 / .000183348 | .019839183 / .026473491 | .013286315 / .051800001 |
| clean-only clean | .000106208 / .000108170 | .072627245 / .149497166 | .013081977 / .027006222 |
| clean-only corrupt | .000104774 / .000107954 | .018808604 / .025534155 | .013242029 / .051622856 |
| corrupt-only clean | .000420677 / .000432099 | .090070655 / .185685650 | .014991640 / .043700863 |
| corrupt-only corrupt | .000416586 / .000441868 | .019976741 / .026864838 | .014548696 / .056849524 |

All support counts are nonzero. Clean states remain larger than corrupted
states. No claimed identity protection follows from these probes.

## No-update repeat: reset is exact, CUDA trajectories are not

The original predictor remains bitwise unchanged and its phi0 outputs repeat
exactly (maximum difference0), after all three independent probes. With no
optimizer update, its repeated loss means are clean **.356556193**, corrupted
**.599193961**, joint **.477875077**: deltas **+.000484032,−.000931916,−.000223942**.
Maximum per-episode absolute loss variation is **.006211102**; maximum phi3
coordinate difference is **.002021194**. All8 repeat deltas are in the tables.

This repeat variation is materially larger than many of the predicted deltas.
One repeat is not a noise distribution or confidence interval. The actual
single-step probes cannot separate real-objective FO approximation error,
finite-step/inner-loop effects and numerical variation. No RNG, kernel,
tolerance, lr or model setting was changed after observing this result. No
additional probes were run to obtain a preferred sign. The earlier T013-A
bitwise failure remains intact.

## Stage D: small nonzero conditioning, dominant shared output

For joint one-step output, norm(b)=**.000145374048**. The8 weight-term norms are
.000026014,.000017337,.000027819,.000031102,.000033922,.000039426,.000028236,
.000022517; total output norms range **.000162663–.000183348**. AllWh/b/phi0
vectors and h features are saved.

| Energy term | Value | Fraction of actual total output energy |
| --- | ---: | ---: |
| Sum norm(Wh)^2 | 6.727963738e-9 | 2.7932606% |
| 8 norm(b)^2 | 1.690689112e-7 | 70.1926395% |
| Cross term 2 sum dot(Wh,b) | 6.506728447e-8 | 27.0140998% |
| Total | 2.408641594e-7 | 100% |

When normalizing the two component energies by their sum (excluding the cross
term), shares are **3.8271236% Wh /96.1728764% bias**. This is a descriptive,
nonorthogonal decomposition, not a uniquely identifiable allocation of total
energy. Wh itself includes a shared feature component.

More directly, centered across-episode output energy is only **.1705291%** of
total output energy (the shared mean accounts for99.8294709%). Coordinate
population standard deviations span3.4000e-7–5.0765e-6. Leading centered singular
values are1.9064306e-5,6.8771071e-6,3.2670572e-8; all8 values and8x8 covariance
are retained. Same-image clean/corrupt distances are8.771288e-6,3.310317e-6,
9.413953e-6,5.752268e-6. The24 different-image episode-pair distances range
1.214251e-6–2.380824e-5 (mean9.599458e-6).

Thus image conditioning is nonzero but small relative to the common offset at
this first update. This supports R017's **insufficient early image-conditioning
risk**, without claiming what longer training would do or redesigning the head.

## Commands, environment and delivery

Remote Python3.12.12,torch2.4.0+cu121,RTX A6000,one CPU thread. Peak allocated CUDA
memory **4,840,772,096 bytes**; diagnostic elapsed **16.318372116s**. Four existing
NVML/protobuf warnings in regression and existing model/tokenizer warnings are
retained. No implementation or real-run failure occurred. The first queue read
attempted a nonexistent continuation filename; LATEST correctly directed R017
to the main queue, which was read before planning. No research instruction changed.

```bash
export TAISP_SOURCE_REVISION=68a6b07 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.gradient_conflict --manifest research_log/T013B_train_microset.json --supports /home/liujianhua/wjq/TAISP/runs/20260913-014612-taisp-t013b-source-three-steps/artifacts/smoke/supports.json --output "$AUTODL_ARTIFACTS_DIR/audit"
```

Local focused command: `D:/anaconda3/python.exe -m pytest tests/test_gradient_conflict.py
tests/test_source_meta_smoke.py tests/test_initialization.py tests/test_meta_gradient.py
tests/test_trust_radius.py -q`, with PYTHONUTF8=1 and OPENBLAS/OMP threads1.
The complete remote regression skips10 opt-in real-model tests; the distinct
authorized diagnostic executes actual source/CLIP models. `scripts/report_t013c.py`
renders retained measurements to `T013C/tables.md` without model execution.

All gradients, matrices, vectors, three probes, four predictor checkpoints,
repeat measurements, logs and run metadata are stored at
`research_log/remote_runs/20260913-030301-taisp-t013c-conflict-conditioning`
locally and the corresponding A6000 `runs/` directory. No rerun for archival.

**Review recommendation:** retain the measured local gradient opposition and
shared-offset dominance. Finite-step cross-harm is not established; the R017
aligned-gradient curvature branch also does not apply because aggregates are
opposing. A subsequent research decision should account for numerical repeat
variation before interpreting tiny finite-step effects. Do not infer that longer
training, an LR sweep or an identity regularizer is now authorized. Stop at
T013-C: no T013-D, longer training, objective/architecture change, val/target/AP
evaluation, new cohort, spatial ISP or gating/dose work.
