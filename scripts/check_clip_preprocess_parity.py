"""R003 forward-only geometry and content parity with Transformers 4.44.2."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import CLIPImageProcessor
from transformers.image_transforms import center_crop, get_resize_output_image_size

from taisp.losses.clip_semantic import CLIPTensorPreprocess

SHAPES = [(612, 612), (479, 641), (641, 479), (333, 517), (517, 333), (480, 640), (391, 641)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    candidate = CLIPTensorPreprocess().eval()
    reference = CLIPImageProcessor(
        do_resize=True, size={"shortest_edge": 224}, resample=Image.Resampling.BICUBIC,
        do_center_crop=True, crop_size={"height": 224, "width": 224},
        do_rescale=True, rescale_factor=1/255, do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    source = Image.open(args.image).convert("RGB")
    generator = np.random.default_rng(20260912)
    rows = []
    with torch.no_grad():
        for h, w in SHAPES:
            natural = np.asarray(source.resize((w, h), Image.Resampling.BICUBIC))
            noise = generator.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            for content, array in (("natural", natural), ("noise", noise)):
                tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float()/255
                actual = candidate(tensor)
                golden = reference(images=Image.fromarray(array), return_tensors="pt")["pixel_values"]
                resize_shape = get_resize_output_image_size(array, 224, default_to_square=False,
                                                            input_data_format="channels_last")
                resized_tensor = F.interpolate(tensor, size=resize_shape, mode="bicubic",
                                                align_corners=False, antialias=True)
                # Isolate geometry from PIL quantization/interpolation: apply
                # the pinned reference crop to exactly the same resized tensor.
                ref_crop = center_crop(resized_tensor[0].numpy(), (224, 224),
                                       input_data_format="channels_first", data_format="channels_first")
                geometry_reference = (torch.from_numpy(ref_crop).unsqueeze(0)-candidate.mean)/candidate.std
                difference = (actual - golden).abs()
                rows.append({"height": h, "width": w, "content": content,
                    "reference_resize_shape": list(resize_shape), "output_shape": list(actual.shape),
                    "geometry_max_abs_diff": (actual - geometry_reference).abs().max().item(),
                    "normalized_mae": difference.mean().item(),
                    "normalized_rmse": difference.square().mean().sqrt().item(),
                    "normalized_p99_abs": difference.flatten().quantile(0.99).item(),
                    "normalized_max_abs": difference.max().item()})
    result = {"reference": "transformers==4.44.2 CLIPImageProcessor; explicit pinned OpenAI ViT-B/32 settings",
              "source_image": args.image.name, "seed": 20260912,
              "predeclared_tolerance": {"geometry_max_abs_diff": 1e-7, "normalized_rmse": 0.02},
              "geometry_pass": all(r["geometry_max_abs_diff"] <= 1e-7 for r in rows),
              "content_pass": all(r["normalized_rmse"] <= 0.02 for r in rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("geometry_pass", "content_pass")}))
    print("max_geometry_error", max(r["geometry_max_abs_diff"] for r in rows),
          "max_normalized_rmse", max(r["normalized_rmse"] for r in rows))


if __name__ == "__main__":
    main()
