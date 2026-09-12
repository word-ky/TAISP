import hashlib
import json
import sys

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
