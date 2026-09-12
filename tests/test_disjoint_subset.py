import hashlib
import json
import sys
import random

from scripts import prepare_coco_subset


def test_download_manifest_excludes_historical_ids(tmp_path, monkeypatch):
    annotation = tmp_path/'annotations.json'
    annotation.write_text(json.dumps({'images': [{'id': i, 'file_name': f'{i}.jpg', 'coco_url': f'fixture:{i}'} for i in range(8)]}))
    historical = tmp_path/'old.json'
    historical.write_text(json.dumps({'image_ids': [0, 1, 2]}))
    root = tmp_path/'new'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--root', str(root), '--count', '4', '--seed', '20260913', '--exclude-manifest', str(historical), '--annotations-from', str(annotation)])
    monkeypatch.setattr(prepare_coco_subset, 'urlretrieve', lambda url, path: path.write_bytes(url.encode()))
    prepare_coco_subset.main()
    m = json.loads((root/'subset.json').read_text())
    assert len(m['image_ids']) == 4 and not set(m['image_ids'])&{0, 1, 2}
    assert m['overlap_count'] == 0
    assert m['annotation_sha256'] == hashlib.sha256(annotation.read_bytes()).hexdigest()
    for image in m['images']:
        assert image['sha256'] == hashlib.sha256((root/'val2017'/image['file_name']).read_bytes()).hexdigest()
    prepare_coco_subset.main()
    assert json.loads((root/'subset.json').read_text()) == m


def test_two_exclusions_and_random_order_replication_blocks(tmp_path, monkeypatch):
    annotation = tmp_path/'annotations.json'
    annotation.write_text(json.dumps({'images': [{'id': i, 'file_name': f'{i}.jpg', 'coco_url': f'fixture:{i}'} for i in range(20)]}))
    old_paths = [tmp_path/'historical.json', tmp_path/'t007.json']
    for path, ids in zip(old_paths, [[0, 1], [2, 3]]):
        path.write_text(json.dumps({'image_ids': ids}))
    root = tmp_path/'new'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--root', str(root), '--count', '10', '--seed', '20260914',
        '--exclude-manifest', str(old_paths[0]), '--exclude-manifest', str(old_paths[1]),
        '--replication-block-size', '2', '--annotations-from', str(annotation)])
    monkeypatch.setattr(prepare_coco_subset, 'urlretrieve', lambda url, path: path.write_bytes(url.encode()))
    prepare_coco_subset.main()
    m = json.loads((root/'subset.json').read_text())
    expected = random.Random(20260914).sample(list(range(4, 20)), 10)
    assert m['selection_order'] == expected and m['image_ids'] == sorted(expected)
    assert list(m['replication_blocks'].values()) == [expected[i:i+2] for i in range(0, 10, 2)]
    assert all(r['overlap_count'] == 0 for r in m['exclusions'])
    assert m['excluded_image_ids'] == [0, 1, 2, 3]
