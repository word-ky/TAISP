"""Obtain official COCO val annotations and a seeded 200-image subset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from urllib.request import urlretrieve
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    annotations = args.root / "instances_val2017.json"
    if not annotations.exists():
        archive = args.root / "annotations_trainval2017.zip"
        print("Downloading official COCO annotations", flush=True)
        urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
        with zipfile.ZipFile(archive) as z:
            annotations.write_bytes(z.read("annotations/instances_val2017.json"))
    data = json.loads(annotations.read_text())
    images = {x["id"]: x for x in data["images"]}
    ids = sorted(random.Random(args.seed).sample(sorted(images), args.count))
    destination = args.root / "val2017"
    destination.mkdir(exist_ok=True)

    def download(image_id):
        info = images[image_id]
        path = destination / info["file_name"]
        if not path.exists():
            urlretrieve(info["coco_url"], path)
        return {"id": image_id, "file_name": info["file_name"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(download, ids))
    manifest = {"seed": args.seed, "count": args.count, "image_ids": ids,
                "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
                "source": "COCO val2017; uniform sample of all 5000 image IDs",
                "images": records}
    (args.root / "subset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(records)} images: {args.root / 'subset.json'}", flush=True)


if __name__ == "__main__":
    main()
