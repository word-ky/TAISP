"""R003: fixed-run first-order prediction, paired correlations and cluster CIs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def correlation(x, y):
    """Pearson over last dimension; also supports bootstrap row matrices."""
    x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 0)


def describe(rows, lr, *, replicates=2000, seed=20260912):
    predicted = -lr * np.einsum("ij,ij->i", [r["g_det"] for r in rows], [r["g_sem"] for r in rows])
    observed = np.array([r["det_loss_delta_sem1"] for r in rows])
    cosine = np.array([r["gradient_cosine"] for r in rows], dtype=float)
    ids = np.array([r["image_id"] for r in rows])
    unique_ids = np.unique(ids)
    # All six conditions for a sampled image travel together in the overall CI.
    clusters = np.array([np.flatnonzero(ids == image_id) for image_id in unique_ids])
    choices = np.random.default_rng(seed).integers(0, len(unique_ids), (replicates, len(unique_ids)))
    selected = clusters[choices].reshape(replicates, -1)
    bp, bo, bc = predicted[selected], observed[selected], cosine[selected]
    agreement = (np.sign(predicted) == np.sign(observed)).astype(float)

    def estimate(value, bootstrap):
        return {"estimate": float(value), "ci95": np.nanquantile(bootstrap, [0.025, 0.975]).tolist(),
                "valid_bootstrap_replicates": int(np.isfinite(bootstrap).sum())}

    result = {
        "observations": len(rows), "image_clusters": len(unique_ids),
        "zero_predicted": int((predicted == 0).sum()), "zero_observed": int((observed == 0).sum()),
        "sign_agreement": estimate(agreement.mean(), agreement[selected].mean(axis=1)),
        "spearman_linear_observed": estimate(correlation(rankdata(predicted), rankdata(observed)),
            correlation(rankdata(bp, axis=1), rankdata(bo, axis=1))),
        "pearson_linear_observed": estimate(correlation(predicted, observed), correlation(bp, bo)),
        "spearman_cosine_observed": estimate(correlation(rankdata(cosine), rankdata(observed)),
            correlation(rankdata(bc, axis=1), rankdata(bo, axis=1))),
        "pearson_cosine_observed": estimate(correlation(cosine, observed), correlation(bc, bo)),
        "linear_mean": estimate(predicted.mean(), bp.mean(axis=1)),
        "observed_mean": estimate(observed.mean(), bo.mean(axis=1)),
    }
    for name, values in (("linear", predicted), ("observed", observed), ("cosine", cosine),
                          ("absolute_taylor_error", abs(observed-predicted))):
        result[name + "_quantiles"] = dict(zip(["p05", "p25", "p50", "p75", "p95"],
                                                np.quantile(values, [.05, .25, .5, .75, .95]).tolist()))
    return result, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study
    rows = [json.loads(line) for line in (study / "samples.jsonl").read_text().splitlines()]
    environment = json.loads((study / "environment.json").read_text())
    lr = environment["config"]["semantic_lr"]
    overall, predictions = describe(rows, lr)
    cases = list(json.loads((study / "summary.json").read_text()))
    groups = {"overall": overall}
    for case in cases:
        chosen = [r for r in rows if f"{r['family']}_s{r['severity']}" == case]
        groups[case], _ = describe(chosen, lr)
    payload = {"formula": "delta_det_linear = -semantic_lr * dot(g_det, g_sem)",
               "semantic_lr": lr, "bootstrap": {"replicates": 2000, "seed": 20260912,
                   "unit": "image_id clusters; all six conditions kept together overall",
                   "interval": "95% percentile, exploratory without multiplicity adjustment"},
               "sign_rule": "exact numpy.sign; zeros counted and not discarded", "groups": groups}
    (study / "linear_analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (study / "linear_scatter.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["image_id", "family", "severity", "gradient_cosine", "delta_det_linear", "det_loss_delta_sem1"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "delta_det_linear": float(p)} for r, p in zip(rows, predictions))
    lines = ["# R003 first-order mechanism diagnostic", "",
             "Prediction: `-0.1 * dot(g_det, g_sem)` from saved initial gradients; "
             "observed change is annotated detector loss after minus before one semantic step.", "",
             "95% percentile intervals use 2,000 image-cluster bootstrap draws, seed 20260912. "
             "Overall draws retain all six observations for each selected image; these are "
             "exploratory intervals without multiplicity adjustment. No protocol rerun/tuning is used.", "",
             "| Group | Sign agreement [95% CI] | Spearman linear vs observed [95% CI] | Pearson linear vs observed | Spearman cosine vs observed [95% CI] |",
             "|---|---:|---:|---:|---:|"]
    def interval(d, percent=False):
        scale = 100 if percent else 1
        suffix = "%" if percent else ""
        return f"{scale*d['estimate']:.3f}{suffix} [{scale*d['ci95'][0]:.3f}, {scale*d['ci95'][1]:.3f}]"
    for key, g in groups.items():
        lines.append(f"| {key} | {interval(g['sign_agreement'], True)} | {interval(g['spearman_linear_observed'])} | "
                     f"{g['pearson_linear_observed']['estimate']:.4f} | {interval(g['spearman_cosine_observed'])} |")
    lines += ["", "## Distribution summaries", "",
              "| Group | Linear Δ p05 / median / p95 | Observed Δ p05 / median / p95 | Cosine p05 / median / p95 |",
              "|---|---:|---:|---:|"]
    for key, g in groups.items():
        def quantiles(name):
            return " / ".join(f"{g[name+'_quantiles'][q]:.6g}" for q in ("p05", "p50", "p95"))
        lines.append(f"| {key} | {quantiles('linear')} | {quantiles('observed')} | {quantiles('cosine')} |")
    lines += ["", "Expected local relationship: positive linear/observed correlation, negative cosine/observed "
              "correlation. Native detector proposal matching and sampling can change across a finite step even "
              "with paired RNG seeds. Correlation/sign disagreement therefore concerns this measured native "
              "loss, and should not be interpreted as a universal failure of first-order calculus.", "",
              "Full Pearson/Spearman intervals, p25/p75, mean intervals and absolute Taylor-error quantiles "
              "are retained in linear_analysis.json. No samples are trimmed or winsorized.", "",
              "![All observations: Taylor prediction vs measured detector loss change](linear_prediction.png)", ""]
    (study / "linear_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), layout="constrained")
    for ax, case in zip(axes.flat, cases):
        indices = [i for i, r in enumerate(rows) if f"{r['family']}_s{r['severity']}" == case]
        ax.scatter(predictions[indices], [rows[i]["det_loss_delta_sem1"] for i in indices],
                   s=12, alpha=.6, linewidths=0, color="#176b91", rasterized=True)
        ax.axhline(0, lw=.6, color="#777777")
        ax.axvline(0, lw=.6, color="#777777")
        ax.axline((0, 0), slope=1, color="#b64b38", lw=.8, ls="--")
        ax.set(title=case.replace("_", " "), xlabel="Taylor prediction: −η g_det · g_sem",
               ylabel="Observed detector loss after − before")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"T002 · first-order prediction · all {len(rows):,} observations", fontsize=12)
    fig.savefig(study / "linear_prediction.png", dpi=180)
    fig.savefig(study / "linear_prediction.pdf")
    plt.close(fig)
    print(study / "linear_analysis.md")


if __name__ == "__main__":
    main()
