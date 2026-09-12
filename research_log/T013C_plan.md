# T013-C pre-outcome diagnostic plan

2026-09-13 02:57 +08. R017/d7ed09c is appended to the main research queue.
T013-B CLOSED; this is a fixed diagnostic, not longer training.

Inputs: exactly T013B_train_microset.json, SHA
164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493;
saved T013-B supports.json, SHA
6523a277cf2183a66412f3ad0a0099687a3851a0c6ff6ce4c81939b3c6112e35.
Read these support fields directly; do not infer new support. Image IDs/order:
65088,426525,541157,129068, each clean then gamma_s2/contrast_s2/color_cast_s2/gamma_s1.
No image selection, data modification, annotation change or support recomputation.

Reuse source_meta_smoke.load_episodes/source_outer_episode, frozen model loaders,
existing ParameterPredictor, full hybrid K3/lr.1/eps1e-12 and oracle seed20260913.
Reconstruct the original T013-B predictor by identical torch.manual_seed20260913
and constructor order: source, CLIP, ISP, predictor, in the same remote environment.
Save its initial state; deep-copy it for each independent probe. Do not load the
T013-B three-step checkpoint as initialization. Never change deployment code.

Stage B: eight independent original-predictor episodes, no optimizer. Record full
dL/dphi0, head.weight and head.bias gradients, trunk-gradient norm (expected0),
loss/components/states. Geometry: 8x8 phi-gradient cosines; four clean/corrupt
same-image pairs; group mean vectors, dot/cosine/clean-to-corrupt norm ratio in
phi and flattened head space; per-coordinate groupmeans/signs; weight/bias norms.

Stage C: exactly three SGD steps TOTAL, one for each independent original copy:
g_joint=mean8, g_clean=mean4clean, g_corrupt=mean4corrupt, coefficient1e-3.
Predicted per-episode delta is -1e-3 dot(g_head_episode, g_head_direction), so the
gradient space matches the actual predictor update. Trunk gradient is zero and
remains unchanged for this first step. No norm normalization or probe selection.
Evaluate eight outer losses after each step, with no second update; retain all
positive/negative changes. Report clean/corrupt/joint means, predicted/actual
per-episode deltas, sign agreement and Pearson correlation (n8 descriptive),
phi0/phi3 mean/max, saturation, empty counts and frozen model state checks.
One final NO-UPDATE repeat of the original baseline on all8 episodes records
CUDA same-path numerical variation and demonstrates episode resets; it is not
an additional optimizer probe. Keep existing T013-A zero-tolerance failure intact.

Stage D on joint only: report feature vectors h, Wh,b,phi0 and norms; coordinate
population std (denominator8), centered singular values, full pairwise phi0
distance matrix and same-image clean/corrupt distances. Energy: E_W=sum||Wh||^2,
E_b=8||b||^2, cross=2sum<Wh,b>, E_total=E_W+E_b+cross. Report normalized component
shares E_W/(E_W+E_b), E_b/(E_W+E_b), and their individual ratios to E_total plus
cross/E_total. These are descriptive; Wh includes a shared feature component,
so also report centered output energy/E_total rather than equating allWh with
unique image information. No invented cutoff for 'overwhelmingly' conditional.

Focused tests: hand-computed gradient geometry/predicted deltas, independent
one-step copies and SGD scaling, energy identity/centered variation, and existing
label-isolation/reset/deployment regression. Then full non-real remote regression
before a single bounded real diagnostic. No new optimizer/lr/regularizer/architecture.

Interpret R017 branches using actual group signs and observed geometry; retain
CUDA repeat variation as a limitation rather than forcing a causal conclusion.
Stop at report/review: no T013-D, longertraining, identityregularizer, architecture
change, target/val/AP evaluation, spatialISP or gates/dose search.
