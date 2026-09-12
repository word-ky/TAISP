"""Official COCOeval of precommitted T010 choices over frozen T009 predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from taisp.analysis.gating import HYBRID, compose_predictions
from taisp.analysis.replication import replication_ap


def evaluate_case(study, prepared, annotations, output, detector, case):
    from pycocotools.coco import COCO
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    manifest, configs = read(prepared/'manifest.json'), read(prepared/'grid.json')
    rows = [json.loads(s) for s in (prepared/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
    masks = np.load(prepared/'decisions.npz')['selected']
    ids = manifest['image_ids']
    indexes = [j for j, r in enumerate(rows) if r['case'] == case]
    paths = [study/'predictions'/f'{detector}_{case}_{v}.json' for v in ('no_adapt', HYBRID)]
    raw, hybrid = [read(p) for p in paths]
    input_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    anchors = read(study/'metrics.json')
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotations))
    panel = {'detector': detector, 'case': case, 'input_prediction_sha256': input_hashes, 'configs': {}}
    for config, mask in zip(configs, masks):
        assert hashlib.sha256(mask.tobytes()).hexdigest() == config['decision_sha256']
        selected = {rows[j]['image_id'] for j in indexes if mask[j]}
        predictions = compose_predictions(raw, hybrid, selected, ids)
        digest = hashlib.sha256((json.dumps(predictions)+'\n').encode()).hexdigest()
        if config['name'] in ('no_adapt', 'full_hybrid'):
            variant = 'no_adapt' if config['name'] == 'no_adapt' else HYBRID
            assert digest == input_hashes[f'{detector}_{case}_{variant}.json']
            values = {group: anchors[group][f'{detector}_{case}_{variant}'] for group in manifest['groups']}
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                values = replication_ap(coco, ids, predictions,
                                        {k: v for k, v in manifest['groups'].items() if k != 'aggregate'})
        panel['configs'][config['name']] = {'decision_sha256': config['decision_sha256'],
            'selected_image_ids': sorted(selected), 'composed_prediction_sha256': digest, 'evaluations': values}
        print(f"evaluated {detector}/{case}/{config['name']}", flush=True)
    target = output/'panels'/f'{detector}_{case}.json'
    target.write_text(json.dumps(panel, indent=2)+'\n', encoding='utf-8')
    return {'panel': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'evaluations': sum(len(c['evaluations']) for c in panel['configs'].values())}


def run(study, prepared, annotations, output, workers):
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    (output/'panels').mkdir(exist_ok=True)
    manifest = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    env = json.loads((study/'environment.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(annotations.read_bytes()).hexdigest() == env['annotation_sha256']
    for name, expected in manifest['outputs_sha256'].items():
        assert hashlib.sha256((prepared/name).read_bytes()).hexdigest() == expected, name
    jobs = [(d, c) for d in env['detectors'] for c in manifest['cases']]
    receipts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_case, study, prepared, annotations, output, d, c) for d, c in jobs]
        for future in as_completed(futures):
            receipts.append(future.result())
            print(f'completed {len(receipts)}/{len(jobs)} detector-condition panels', flush=True)
    receipt = {'source_revision': os.environ.get('TAISP_SOURCE_REVISION', 'unrecorded'),
        'input_revision': env['source_revision'], 'preparation_revision': manifest['source_revision'],
        'images': len(manifest['image_ids']), 'rows': manifest['rows'], 'configurations': manifest['configurations'],
        'groups': len(manifest['groups']), 'workers': workers, 'elapsed_seconds': time.time()-started,
        'official_evaluations': sum(r['evaluations'] for r in receipts),
        'reused_official_anchor_evaluations': 2*len(jobs)*len(manifest['groups']),
        'model_or_adaptation_inference': False,
        'annotation_sha256': env['annotation_sha256'],
        'T009_metrics_sha256': hashlib.sha256((study/'metrics.json').read_bytes()).hexdigest(),
        'preparation_manifest_sha256': hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest(),
        'panels': sorted(receipts, key=lambda r: r['panel'])}
    (output/'completion.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'panels'}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    run(a.study, a.prepared, a.annotations, a.output, a.workers)


if __name__ == '__main__':
    main()
