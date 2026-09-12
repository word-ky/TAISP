"""Plot completed T008 statistics in a process without model-library imports."""
import json
from pathlib import Path
import numpy as np


def main():
    out = Path('research_log/T008')
    a = json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    H = 'det_pseudo_clip_radius'
    cases = ['gamma_s1', 'gamma_s2', 'contrast_s1', 'contrast_s2', 'color_cast_s1', 'color_cast_s2', 'clean_s0']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    for ax, metric, label, scale in zip(axes, ['recall75', 'best_iou_mean', 'class_score_mean', 'fp50'],
                                       ['Recall75 delta (pp)', 'Best IoU delta', 'Class score delta', 'FP50 delta per image'], [100, 1, 1, 1]):
        for det, offset, color in [('source', -.12, '#3574a5'), ('target', .12, '#c65a3a')]:
            s = [a['contrasts'][f'{det}/{H}-minus-no_adapt/K3'][c]['proxies'][metric]['mean_delta'] for c in cases]
            v = np.array([r['estimate'] for r in s])*scale
            bounds = np.array([r['ci95'] for r in s])*scale
            ax.errorbar(v, np.arange(7)+offset, xerr=[v-bounds[:, 0], bounds[:, 1]-v], fmt='o', markersize=3, color=color, label=det, capsize=2)
        ax.axvline(0, color='#999999', lw=.8)
        ax.set_yticks(range(7), cases)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
    axes[0].legend(loc='best', fontsize=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle('T008: hybrid minus no-adapt, K=3; paired image-cluster 95% intervals')
    fig.tight_layout()
    fig.savefig(out/'proxy_deltas.png', dpi=160)
    fig.savefig(out/'proxy_deltas.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
