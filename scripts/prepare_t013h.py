"""Pre-outcome deterministic source cohort; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_t013b import sha256


def prepare(images, annotations, val_annotations, ledger, output):
    assert sha256(annotations) == '610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d'
    assert sha256(val_annotations) == 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
    assert sha256(ledger) == 'ed548679f5030aa17e7a9358d6709e41ab4fa87abd8044575aa03dcfa55ef7e2'
    exclusion = json.loads(ledger.read_text())
    val_ids = {i['id'] for i in json.loads(val_annotations.read_text())['images']}
    assert len(val_ids) == 5000 and set(exclusion['evaluation_ids']) <= val_ids
    excluded = val_ids | set(exclusion['evaluation_ids']) | set(exclusion['prior_source_ids'])
    data = json.loads(annotations.read_text())
    valid = {}
    for ann in data['annotations']:
        _, _, w, h = ann['bbox']
        if not ann.get('iscrowd', 0) and w > 0 and h > 0 and ann.get('area', w*h) > 0:
            valid.setdefault(ann['image_id'], []).append(ann)
    infos = {i['id']: i for i in data['images']}
    available = [i for i in valid if os.access(images/infos[i]['file_name'], os.R_OK)
                 and (images/infos[i]['file_name']).is_file()]
    eligible = [i for i in available if i not in excluded]
    key = lambda i: (hashlib.sha256(f'20260913:{i}'.encode()).hexdigest(), i)
    ids = sorted(eligible, key=key)[:32]
    assert len(ids) == 32
    records = []
    for index, i in enumerate(ids):
        info = infos[i]
        records.append({'image_id': i, 'file_name': info['file_name'], 'path': str(images/info['file_name']),
                        'sha256': sha256(images/info['file_name']), 'width': info['width'], 'height': info['height'],
                        'coco_url': info['coco_url'], 'annotations': valid[i], 'block': index//8,
                        'selection_hash': key(i)[0]})
    result = {'seed': 20260913, 'split': 'train2017', 'images_root': str(images),
              'annotation_file': str(annotations), 'annotation_sha256': sha256(annotations),
              'val_annotation_file': str(val_annotations), 'val_annotation_sha256': sha256(val_annotations),
              'exclusion_ledger_sha256': sha256(ledger), 'val_excluded_count': len(val_ids),
              'evaluation_excluded_count': len(exclusion['evaluation_ids']),
              'prior_source_ids': exclusion['prior_source_ids'], 'annotation_image_count': len(infos),
              'available_valid_count': len(available), 'eligible_available_images': len(eligible),
              'selection': "first32 sorted by (sha256(UTF8('20260913:'+decimalID)), ID), pre-outcome eligibility",
              'image_ids': ids, 'images': records,
              'corrupted_cases': ['gamma_s1', 'gamma_s2', 'contrast_s2', 'color_cast_s2']*8,
              'blocks': [ids[i:i+8] for i in range(0, 32, 8)],
              'episode_order': 'selected image order, clean then corrupted',
              'model_config_pins': 'research_log/T013H_plan.md; unchanged T013B/C source path'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('image_ids', 'available_valid_count', 'eligible_available_images')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'annotations', 'val-annotations', 'ledger', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    prepare(**vars(p.parse_args()))
