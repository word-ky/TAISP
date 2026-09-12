"""T011 condition/block matched identity-hash controls, analysis only."""
import hashlib

import numpy as np


CANDIDATE = 'support_confidence_low_50'
SEEDS = tuple(range(2026091600, 2026091800))


def stratum_indices(rows, groups):
    block_of = {image_id: name for name, ids in groups.items() if name != 'aggregate' for image_id in ids}
    strata = {}
    for j, row in enumerate(rows):
        key = (row['case'], block_of[row['image_id']])
        strata.setdefault(key, []).append(j)
    return strata


def matched_mask(rows, strata, candidate, seed):
    mask = np.zeros(len(rows), dtype=bool)
    for indexes in strata.values():
        count = int(np.asarray(candidate)[indexes].sum())
        def key(j):
            row = rows[j]
            payload = f"TAISP-T011|{seed}|{row['image_id']}|{row['case']}".encode('utf-8')
            return hashlib.sha256(payload).digest(), row['image_id']
        ranked = sorted(indexes, key=key)
        mask[ranked[:count]] = True
    return mask


def randomization(candidate, values):
    a = np.asarray(values, dtype=float)
    def describe(x):
        return {'count': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
                'p05': float(np.quantile(x, .05, method='linear')), 'p95': float(np.quantile(x, .95, method='linear'))}
    return {'candidate': float(candidate), 'random': describe(a), 'candidate_minus_random': describe(candidate-a),
            'strict_empirical_percentile': float(100*np.mean(a < candidate)),
            'random_equal_candidate': int(np.sum(a == candidate)),
            'random_greater_or_equal_candidate': int(np.sum(a >= candidate)),
            'one_sided_tail': float((1+np.sum(a >= candidate))/(len(a)+1))}


def selection_information(candidate_target_deltas, random_p95, beats_median_blocks, clean_deltas):
    return (all(v > 0 for v in candidate_target_deltas) and
            all(c > p for c, p in zip(candidate_target_deltas, random_p95)) and
            min(beats_median_blocks) >= 4 and min(clean_deltas) >= -.10)
