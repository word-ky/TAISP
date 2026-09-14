"""R052 outcome-free cohort and pair-permutation precommit; no model imports."""
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from prepare_t018a import prepare
from prepare_t013b import sha256

FAMILIES = ['gamma_s2', 'contrast_s2', 'color_cast_s2']


def permutations(records):
    rng = np.random.default_rng(20261005)
    result = []
    for _ in range(256):
        assignment = np.arange(len(records))
        for block in range(4):
            for family in FAMILIES:
                indices = [i for i, r in enumerate(records)
                           if r['block'] == block and r['family'] == family]
                assignment[indices] = rng.permutation(indices)
        result.append(assignment.tolist())
    return result


def main(project):
    output = project/'research_log/T034A_train_cohort.json'
    assert not output.exists()
    with contextlib.redirect_stdout(io.StringIO()):
        prepare(project, output, count=120, block_size=30, seed=20261005,
                additional_source=Path('research_log/T033A_train_cohort.json'))
    manifest = json.loads(output.read_text())
    assert len(manifest['prior_source_ids']) == 3791
    manifest['corrupted_cases'] = [FAMILIES[i % 3] for i in range(120)]
    for r, family in zip(manifest['images'], manifest['corrupted_cases']):
        r['family'] = family
    manifest['episode_order'] = 'selected image order; clean_s0 then assigned corruption'
    output.write_text(json.dumps(manifest, indent=2)+'\n')
    candidate = {k: manifest[k] for k in ('images', 'corrupted_cases', 'episode_order')}
    (project/'research_log/T034A_candidate_images.json').write_text(json.dumps(candidate, indent=2)+'\n')
    perm = {'seed': 20261005, 'numpy': np.__version__, 'generator': 'default_rng/PCG64',
            'order': 'permutation, block 0..3, gamma/contrast/color_cast; ascending pair indices',
            'fixed_points_allowed': True, 'cohort_sha256': sha256(output),
            'pair_source_indices': permutations(manifest['images'])}
    (project/'research_log/T034A_permutations.json').write_text(json.dumps(perm, indent=2)+'\n')
    print(json.dumps({'cohort_sha256': sha256(output), 'images': 120,
                      'excluded_prior': 3791, 'permutations': 256}))


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]))
