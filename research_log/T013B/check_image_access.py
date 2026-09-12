"""Read-only diagnosis of the observed train2017 symlink permission failure."""
import json
import os
from pathlib import Path

root = Path('/home/wenchang/asdasdsad/wjq/dapd_query_opt44/assets/rexeval/coco/train2017')
files = list(root.glob('*.jpg'))
readable = [p.name for p in files if os.access(p, os.R_OK)]
print(json.dumps({'image_root': str(root), 'jpg_entries': len(files),
                  'symlink_entries': sum(p.is_symlink() for p in files),
                  'readable_images': len(readable), 'readable_filenames': readable}, indent=2))
