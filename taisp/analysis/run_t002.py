"""Fixed COCO-subset feasibility study. This is the only annotated driver."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time

import torch
import yaml

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import CLIP_MODEL, CLIP_REVISION, POSITIVE_PROMPTS, NEGATIVE_PROMPTS, load_clip_guidance
from taisp.models.detector import load_detector
from taisp.tta import AdaptConfig, adapt
from .coco import COCOSubset, prediction_records, subset_ap
from .corruptions import CASES, corrupt
from .oracle import detector_task_loss


def physical_vector(isp, phi):
    p = isp.decode(phi)
    return torch.cat([p["gamma"].reshape(-1), p["gains"].reshape(-1),
                      p["contrast"].reshape(-1), p["brightness"].reshape(-1),
                      p["tone"].reshape(-1), p["sharpening"].reshape(-1)])


def summarize(rows):
    summaries = {}
    for family, severity in CASES:
        group = [r for r in rows if r["family"] == family and r["severity"] == severity]
        cosines = [r["gradient_cosine"] for r in group if r["gradient_cosine"] is not None]
        entry = {
            "count": len(group), "valid_cosines": len(cosines),
            "cosine_mean": statistics.mean(cosines) if cosines else None,
            "cosine_median": statistics.median(cosines) if cosines else None,
            "positive_alignment_fraction": sum(c > 0 for c in cosines) / len(cosines) if cosines else None,
            "semantic_step_reduces_detector_loss_fraction": statistics.mean(r["det_loss_delta_sem1"] < 0 for r in group),
        }
        for name in ("det_loss_before", "det_loss_delta_sem1", "det_loss_delta_sem3",
                     "det_loss_delta_oracle", "semantic_loss_before", "semantic_loss_1",
                     "semantic_loss_3", "adapt_seconds_3", "peak_allocated_mb",
                     "saturation_before", "saturation_1", "saturation_3", "g_sem_norm", "g_det_norm"):
            entry[name + "_mean"] = statistics.mean(r[name] for r in group)
        for name in ("physical_change_1", "physical_change_3", "g_sem", "g_det"):
            entry[name + "_mean_abs"] = torch.tensor([r[name] for r in group]).abs().mean(0).tolist()
        summaries[f"{family}_s{severity}"] = entry
    return summaries


def environment_metadata(config):
    from huggingface_hub import hf_hub_download

    clip_path = Path(hf_hub_download(CLIP_MODEL, "pytorch_model.bin", revision=CLIP_REVISION, local_files_only=True))
    detector_path = Path(torch.hub.get_dir()) / "checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    def checksum(path):
        h = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    return {
        "config": config, "source_revision": os.environ.get("TAISP_SOURCE_REVISION", "working-tree"),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "pycocotools", "numpy", "Pillow")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "clip_model": CLIP_MODEL, "clip_revision": CLIP_REVISION,
        "clip_sha256": checksum(clip_path), "detector_sha256": checksum(detector_path),
        "positive_prompts": POSITIVE_PROMPTS, "negative_prompts": NEGATIVE_PROMPTS,
        "interpretation": "Fixed COCO-val subset feasibility; no full benchmark claim",
    }


def run(config, data_root, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.backends.cudnn.benchmark = False
    device = torch.device(config["device"])
    data = COCOSubset(data_root)
    ids = data.ids[:limit] if limit is not None else data.ids[:config["count"]]
    guidance = load_clip_guidance(device, local_files_only=True)
    detector = load_detector(device)
    isp = DifferentiableISP().to(device)
    settings = AdaptConfig(steps=config["semantic_steps"], lr=config["semantic_lr"],
                           consistency_weight=config["consistency_weight"],
                           regularization_weight=config["regularization_weight"])
    meta = environment_metadata(config)
    meta["evaluated_image_ids"] = ids
    meta["smoke_limit"] = limit
    meta["annotation_sha256"] = data.manifest["annotation_sha256"]
    (output / "environment.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.copyfile(data.root / "subset.json", output / "subset.json")
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    predictions = {"clean": []}
    for family, severity in CASES:
        for method in ("corrupted", "semantic1", "semantic3", "oracle1"):
            predictions[f"{family}_s{severity}_{method}"] = []
    rows = []
    initial_physical = physical_vector(isp, torch.zeros(8, device=device))
    started = time.time()
    with (output / "samples.jsonl").open("w", encoding="utf-8") as samples:
        for index, image_id in enumerate(ids):
            clean, targets = data.load(image_id, device)
            with torch.no_grad():
                predictions["clean"].extend(prediction_records(image_id, detector(clean)[0]))
            for family, severity in CASES:
                x = corrupt(clean, family, severity)
                phi0 = torch.zeros(8, device=device, requires_grad=True)
                loss0, components = detector_task_loss(detector, isp(x, phi0), targets, seed=config["seed"] + image_id)
                g_det = torch.autograd.grad(loss0, phi0)[0].detach()
                loss0_value = loss0.detach().item()
                del loss0, components
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                # The label-free deployment call receives only x and CLIP, no
                # detector, annotations, family, severity or oracle gradients.
                result = adapt(x, isp, guidance, config=settings)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                g_sem = x.new_tensor(result.diagnostics[0]["gradient_per_coordinate"])
                phi1 = x.new_tensor(result.diagnostics[1]["phi"])
                sem1 = isp(x, phi1).detach()
                oracle_phi = -config["oracle_lr"] * g_det
                oracle_image = isp(x, oracle_phi).detach()
                norm_product = (g_sem.norm() * g_det.norm()).item()
                cosine = float(torch.dot(g_sem, g_det) / norm_product) if norm_product > 0 else None
                with torch.no_grad():
                    loss1, _ = detector_task_loss(detector, sem1, targets, seed=config["seed"] + image_id)
                    loss3, _ = detector_task_loss(detector, result.enhanced, targets, seed=config["seed"] + image_id)
                    oracle_loss, _ = detector_task_loss(detector, oracle_image, targets, seed=config["seed"] + image_id)
                    for method, enhanced in (("corrupted", x), ("semantic1", sem1),
                                             ("semantic3", result.enhanced), ("oracle1", oracle_image)):
                        key = f"{family}_s{severity}_{method}"
                        predictions[key].extend(prediction_records(image_id, detector(enhanced)[0]))
                row = {
                    "image_id": image_id, "family": family, "severity": severity,
                    "gradient_cosine": cosine, "g_sem": g_sem.cpu().tolist(), "g_det": g_det.cpu().tolist(),
                    "g_sem_norm": g_sem.norm().item(), "g_det_norm": g_det.norm().item(),
                    "det_loss_before": loss0_value, "det_loss_delta_sem1": loss1.item() - loss0_value,
                    "det_loss_delta_sem3": loss3.item() - loss0_value,
                    "det_loss_delta_oracle": oracle_loss.item() - loss0_value,
                    "semantic_loss_before": result.diagnostics[0]["semantic"],
                    "semantic_loss_1": result.diagnostics[1]["semantic"],
                    "semantic_loss_3": result.diagnostics[-1]["semantic"],
                    "physical_change_1": (physical_vector(isp, phi1) - initial_physical).cpu().tolist(),
                    "physical_change_3": (physical_vector(isp, result.phi) - initial_physical).cpu().tolist(),
                    "saturation_before": result.diagnostics[0]["saturation_rate"],
                    "saturation_1": result.diagnostics[1]["saturation_rate"],
                    "saturation_3": result.diagnostics[-1]["saturation_rate"],
                    "adapt_seconds_3": duration, "peak_allocated_mb": peak_memory,
                    "diagnostics": result.diagnostics,
                }
                samples.write(json.dumps(row, allow_nan=False) + "\n")
                samples.flush()
                rows.append(row)
            print(f"completed {index + 1}/{len(ids)} image_id={image_id} elapsed={time.time()-started:.1f}s", flush=True)
    metrics = {}
    for name, records in predictions.items():
        (prediction_dir / f"{name}.json").write_text(json.dumps(records) + "\n")
        print(f"Evaluating {name}", flush=True)
        metrics[name] = subset_ap(data.coco, ids, records)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "scatter.csv").open("w", newline="", encoding="utf-8") as f:
        names = ["image_id", "family", "severity", "gradient_cosine", "det_loss_delta_sem1", "g_sem_norm", "g_det_norm"]
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "completion.json").write_text(json.dumps({"images": len(ids), "samples": len(rows),
                                                         "elapsed_seconds": time.time() - started}) + "\n")
    print(f"Finished {len(ids)} images / {len(rows)} corrupted samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/t002.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke only; clearly recorded in metadata")
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.data_root, args.output, args.limit)


if __name__ == "__main__":
    main()
