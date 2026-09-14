"""R050 deterministic spherical K=2; fitting accepts gradient vectors only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EPS = 1e-12


def units(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    n = np.linalg.norm(x, axis=1)
    return x / (n[:, None] + EPS), n == 0


def assign(vectors, centroids):
    u, zero = units(vectors)
    c = np.asarray(centroids, dtype=np.float64)
    cn = np.linalg.norm(c, axis=1)
    assert np.all(cn > 0)
    # Row normalization is immaterial for argmax, column normalization is not.
    a = np.argmax(u @ (c / cn[:, None]).T, axis=1)
    a[zero] = 0
    return a


def fit_router(vectors):
    u, zero = units(vectors)
    assert u.shape == (360, 8) and np.isfinite(u).all()
    nz = np.flatnonzero(~zero)
    if not len(nz):
        return dict(status='FAIL_ROUTER_COLLAPSE', reason='all_zero', counts=[360, 0], iterations=0)
    first = int(nz[0]); c0 = u[first]
    cos = (u[nz] @ c0) / (np.linalg.norm(u[nz], axis=1) * np.linalg.norm(c0))
    second = int(nz[int(np.argmin(cos))])
    c = np.stack([c0, u[second]])
    previous = None; trace = []; converged = False
    for iteration in range(1, 21):
        a = assign(vectors, c)
        counts = np.bincount(a, minlength=2).tolist()
        trace.append(counts)
        if previous is not None and np.array_equal(a, previous):
            converged = True
            break
        means = [u[a == k].mean(0) if counts[k] else np.zeros(8) for k in range(2)]
        norms = [np.linalg.norm(m) for m in means]
        if any(n == 0 for n in norms):
            return dict(status='FAIL_ROUTER_COLLAPSE', reason='empty_or_zero_mean_centroid',
                        counts=counts, assignments=a.tolist(), iterations=iteration, count_trace=trace)
        c = np.stack([m/n for m, n in zip(means, norms)])
        previous = a.copy()
    return dict(status='router_complete' if min(counts) >= 72 else 'FAIL_ROUTER_COLLAPSE',
                centroids=c.tolist(), assignments=a.tolist(), counts=counts, iterations=iteration,
                count_trace=trace, initial_indices=[first, second], zero_count=int(zero.sum()),
                converged=converged,
                dtype='float64', device='cpu', epsilon=EPS, K=2, restarts=0)


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(candidate_root, lock_path, output):
    lock = json.loads(lock_path.read_text()); assert lock['candidate_commit']
    for f, h in lock['files'].items(): assert sha(candidate_root/f) == h, f
    all_rows = json.loads((candidate_root/'records.json').read_text())
    assert len(all_rows) == 480 and all(r['integrity_passed'] for r in all_rows)
    rows = [r for r in all_rows if r['partition'] == 'train']
    # Only this numeric array enters the clustering routine. No metadata or labels.
    result = fit_router([r['g_hard'] for r in rows])
    result.update(candidate_lock_sha256=sha(lock_path), candidate_records_sha256=sha(candidate_root/'records.json'),
                  train_episode_indices=[r['episode_index'] for r in rows],
                  code_sha256=sha(Path(__file__)), annotations_loaded=False, task_gradients_computed=0)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'iterations': result['iterations']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', type=Path, required=True)
    p.add_argument('--candidate-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();run(a.candidate_root, a.candidate_lock, a.output)
