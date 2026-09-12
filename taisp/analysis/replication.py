"""Official AP on a fixed cohort and predeclared replication blocks."""
from .coco import subset_ap


def replication_ap(coco, ids, predictions, blocks):
    available = set(ids)
    groups = {'aggregate': ids, **{name: [i for i in members if i in available] for name, members in blocks.items()}}
    out = {}
    for name, members in groups.items():
        if members:
            chosen = set(members)
            out[name] = subset_ap(coco, members, [p for p in predictions if p['image_id'] in chosen])
    return out


def ap_contrasts(metrics, cases, detectors, variants, references=('no_adapt', 'global_generic', 'det_pseudo')):
    """AP values stay in [0,1]; report deltas/macro in AP points."""
    out = {}
    for detector in detectors:
        out[detector] = {}
        for variant in variants:
            by_reference = {}
            for reference in references:
                deltas = {case: {metric: 100*(metrics[f'{detector}_{case}_{variant}'][metric]-
                                    metrics[f'{detector}_{case}_{reference}'][metric])
                                 for metric in ('AP', 'AP50', 'AP75')}
                          for case in cases}
                corrupted = [c for c in cases if c != 'clean_s0']
                by_reference[reference] = {'cases': deltas,
                    'macro_corruption_AP_delta': sum(deltas[c]['AP'] for c in corrupted)/len(corrupted),
                    'positive_corruption_count': sum(deltas[c]['AP'] > 0 for c in corrupted),
                    'clean_AP_delta': deltas['clean_s0']['AP']}
            out[detector][variant] = by_reference
    return out
