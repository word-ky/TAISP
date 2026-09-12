"""Obtain official COCO val annotations and a seeded 200-image subset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
from urllib.request import urlretrieve
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--replication-block-size", type=int)
    parser.add_argument("--annotations-from", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    annotations = args.root / "instances_val2017.json"
    if not annotations.exists():
        if args.annotations_from:
            shutil.copyfile(args.annotations_from, annotations)
        else:
            archive = args.root / "annotations_trainval2017.zip"
            print("Downloading official COCO annotations", flush=True)
            urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
            with zipfile.ZipFile(archive) as z:
                annotations.write_bytes(z.read("annotations/instances_val2017.json"))
    data = json.loads(annotations.read_text())
    images = {x["id"]: x for x in data["images"]}
    excluded_sets = [set(json.loads(path.read_text())["image_ids"]) for path in args.exclude_manifest]
    excluded = set().union(*excluded_sets)
    selection_order = random.Random(args.seed).sample(sorted(set(images)-excluded), args.count)
    ids = sorted(selection_order)
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
                "source": "COCO val2017; random.Random(seed).sample(sorted(all IDs minus exclusions), count), then sort",
                "images": records}
    if args.exclude_manifest:
        manifest.update(excluded_image_ids=sorted(excluded), overlap_count=len(set(ids)&excluded),
                        excluded_manifest_sha256=hashlib.sha256(args.exclude_manifest[0].read_bytes()).hexdigest(),
                        exclusions=[{'manifest': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'count': len(old), 'overlap_count': len(set(ids)&old)}
                                    for path, old in zip(args.exclude_manifest, excluded_sets)])
    if args.replication_block_size:
        manifest.update(selection_order=selection_order, replication_blocks={
            f'block_{i//args.replication_block_size+1}': selection_order[i:i+args.replication_block_size]
            for i in range(0, len(selection_order), args.replication_block_size)},
            execution_order='image_ids is sorted; replication_blocks preserve the original random selection order')
    (args.root / "subset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(records)} images: {args.root / 'subset.json'}", flush=True)


if __name__ == "__main__":
    main()
