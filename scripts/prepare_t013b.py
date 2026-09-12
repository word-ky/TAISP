"""Select a fixed already-available train2017 microset; no download/model calls."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(images, annotations, output):
    data = json.loads(annotations.read_text(encoding='utf-8'))
    valid = {}
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        if not ann.get('iscrowd', 0) and w > 0 and h > 0 and ann.get('area', w*h) > 0:
            valid.setdefault(ann['image_id'], []).append(ann)
    infos = {info['id']: info for info in data['images']}
    # This installation mixes readable images with inaccessible /root symlinks.
    eligible = sorted(i for i in valid if os.access(images/infos[i]['file_name'], os.R_OK)
                      and (images/infos[i]['file_name']).is_file())
    ids = random.Random(20260913).sample(eligible, 4)
    records = []
    for image_id in ids:
        info = infos[image_id]
        records.append({'image_id': image_id, 'file_name': info['file_name'],
                        'path': str(images/info['file_name']),
                        'sha256': sha256(images/info['file_name']),
                        'width': info['width'], 'height': info['height'],
                        'coco_url': info.get('coco_url'), 'annotations': valid[image_id]})
    result = {'seed': 20260913, 'split': 'train2017', 'images_root': str(images),
              'annotation_file': str(annotations), 'annotation_sha256': sha256(annotations),
              'annotation_image_count': len(infos), 'eligible_available_images': len(eligible),
              'selection': 'Random(20260913).sample(sorted(valid available IDs),4)',
              'image_ids': ids, 'images': records,
              'corrupted_cases': ['gamma_s2', 'contrast_s2', 'color_cast_s2', 'gamma_s1'],
              'episode_order': 'selected image order, clean then corrupted'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'image_ids': ids, 'eligible_available_images': len(eligible)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.images, args.annotations, args.output)
