"""Render tables and an unfiltered scatter plot from completed T002 receipts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study
    metrics = json.loads((study / "metrics.json").read_text())
    summary = json.loads((study / "summary.json").read_text())
    environment = json.loads((study / "environment.json").read_text())
    completion = json.loads((study / "completion.json").read_text())
    rows = [json.loads(line) for line in (study / "samples.jsonl").read_text().splitlines()]
    semantic_gradients = np.array([r["g_sem"] for r in rows])
    detector_gradients = np.array([r["g_det"] for r in rows])
    observations = {
        "observations": len(rows), "unique_images": len({r["image_id"] for r in rows}),
        "all_gradients_finite": bool(np.isfinite(semantic_gradients).all() and np.isfinite(detector_gradients).all()),
        "semantic_zero_norm_count": int((np.linalg.norm(semantic_gradients, axis=1) == 0).sum()),
        "detector_zero_norm_count": int((np.linalg.norm(detector_gradients, axis=1) == 0).sum()),
        "overall_mean_cosine": float(np.mean([r["gradient_cosine"] for r in rows])),
        "positive_alignment_fraction": float(np.mean([r["gradient_cosine"] > 0 for r in rows])),
        "detector_loss_decrease_fraction": float(np.mean([r["det_loss_delta_sem1"] < 0 for r in rows])),
        "semantic_loss_decrease_3_fraction": float(np.mean([r["semantic_loss_3"] < r["semantic_loss_before"] for r in rows])),
        "mean_absolute_semantic_gradient": np.abs(semantic_gradients).mean(0).tolist(), "cases": {},
    }
    for case in summary:
        group = [r for r in rows if f"{r['family']}_s{r['severity']}" == case]
        observations["cases"][case] = {
            "mean_signed_physical_change_3": np.mean([r["physical_change_3"] for r in group], axis=0).tolist(),
            "saturation_3_p95": float(np.quantile([r["saturation_3"] for r in group], .95)),
            "fraction_saturation_3_above_10pct": float(np.mean([r["saturation_3"] > .1 for r in group])),
        }
    (study / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    fields = ["case", "n", "AP_corrupted", "AP_semantic1", "AP_semantic3", "AP_oracle1",
              "cosine_mean", "cosine_median", "positive_alignment_pct", "loss_decrease_pct",
              "loss_delta_sem1", "loss_delta_sem3", "loss_delta_oracle", "semantic_loss_3",
              "saturation_before_pct", "saturation_3_pct", "adapt_seconds_3", "peak_memory_mb"]
    table = []
    for case, s in summary.items():
        entry = {"case": case, "n": s["count"]}
        for method in ("corrupted", "semantic1", "semantic3", "oracle1"):
            entry["AP_" + method] = 100 * metrics[f"{case}_{method}"]["AP"]
        entry.update({
            "cosine_mean": s["cosine_mean"], "cosine_median": s["cosine_median"],
            "positive_alignment_pct": 100 * s["positive_alignment_fraction"],
            "loss_decrease_pct": 100 * s["semantic_step_reduces_detector_loss_fraction"],
            "loss_delta_sem1": s["det_loss_delta_sem1_mean"],
            "loss_delta_sem3": s["det_loss_delta_sem3_mean"],
            "loss_delta_oracle": s["det_loss_delta_oracle_mean"],
            "semantic_loss_3": s["semantic_loss_3_mean"],
            "saturation_before_pct": 100 * s["saturation_before_mean"],
            "saturation_3_pct": 100 * s["saturation_3_mean"],
            "adapt_seconds_3": s["adapt_seconds_3_mean"],
            "peak_memory_mb": s["peak_allocated_mb_mean"],
        })
        table.append(entry)
    with (study / "summary_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = ["# T002 fixed-subset results", "",
             f"Source `{environment['source_revision']}`; {completion['images']} COCO-val2017 images, "
             f"{completion['samples']} image/corruption observations. Seed {environment['config']['seed']}.", "",
             f"Clean subset bbox AP: **{metrics['clean']['AP'] * 100:.3f}**. "
             "All AP values below are points on a 0–100 scale, evaluated on the same fixed subset.", "",
             "| Condition | No adaptation | CLIP 1 step | CLIP 3 steps | Oracle 1 step |",
             "|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['case']} | {r['AP_corrupted']:.3f} | {r['AP_semantic1']:.3f} | "
                     f"{r['AP_semantic3']:.3f} | {r['AP_oracle1']:.3f} |")
    lines += ["", "| Condition | Mean cosine | Median cosine | Positive alignment | Semantic step lowers detector loss |",
              "|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['case']} | {r['cosine_mean']:.4f} | {r['cosine_median']:.4f} | "
                     f"{r['positive_alignment_pct']:.1f}% | {r['loss_decrease_pct']:.1f}% |")
    lines += ["", "| Condition | Detector loss Δ, 1 step | Detector loss Δ, 3 steps | Oracle loss Δ | CLIP loss after 3 steps |",
              "|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['case']} | {r['loss_delta_sem1']:.6f} | {r['loss_delta_sem3']:.6f} | "
                     f"{r['loss_delta_oracle']:.6f} | {r['semantic_loss_3']:.6f} |")
    lines += ["", "| Condition | Saturation before | Saturation after 3 steps | Adaptation, 3 steps (s/image) | Peak allocated (MiB) |",
              "|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['case']} | {r['saturation_before_pct']:.3f}% | {r['saturation_3_pct']:.3f}% | "
                     f"{r['adapt_seconds_3']:.4f} | {r['peak_memory_mb']:.1f} |")
    names = ["gamma", "R", "G", "B", "contrast", "brightness", "tone", "sharpening"]
    for key, label in (("physical_change_1_mean_abs", "Physical parameter absolute change, 1 step"),
                       ("physical_change_3_mean_abs", "Physical parameter absolute change, 3 steps"),
                       ("g_sem_mean_abs", "Mean absolute semantic gradient per raw coordinate"),
                       ("g_det_mean_abs", "Mean absolute oracle gradient per raw coordinate")):
        lines += ["", f"## {label}", "", "| Condition | " + " | ".join(names) + " |",
                  "|---|" + "---:|" * 8]
        for case, values in summary.items():
            lines.append(f"| {case} | " + " | ".join(f"{v:.6g}" for v in values[key]) + " |")
    lines += ["", "The oracle is analysis-only. Negative loss delta indicates improvement. "
              "No samples or negative families are omitted. Latency includes the three-step "
              "adaptation call and diagnostics, excludes data loading/oracle/detection evaluation. "
              "Peak allocated memory includes loaded models. Native proposal matching makes "
              "detector loss piecewise smooth; it is distinct from AP. See docs/T002.md.", "",
              "![All per-sample gradient cosines versus detector-loss changes](gradient_alignment.png)", ""]
    (study / "results.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), layout="constrained")
    for ax, (case, s) in zip(axes.flat, summary.items()):
        group = [r for r in rows if f"{r['family']}_s{r['severity']}" == case and r["gradient_cosine"] is not None]
        ax.scatter([r["gradient_cosine"] for r in group], [r["det_loss_delta_sem1"] for r in group],
                   s=12, alpha=0.65, linewidths=0, color="#176b91", rasterized=True)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.axvline(0, color="#999999", lw=0.6, ls="--")
        ax.set(xlim=(-1.05, 1.05), title=f"{case.replace('_', ' ')} (n={len(group)})",
               xlabel="cos(g_sem, g_det)", ylabel="Detector loss after − before")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"T002 · one semantic step · all {len(rows):,} observations", fontsize=12)
    fig.savefig(study / "gradient_alignment.png", dpi=180)
    fig.savefig(study / "gradient_alignment.pdf")
    plt.close(fig)
    print(study / "results.md")


if __name__ == "__main__":
    main()
