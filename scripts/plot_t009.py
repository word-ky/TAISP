"""Aggregate and fixed-block macro AP deltas, with no invented AP intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.study/'analysis.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, detector, title in zip(axes, ['target', 'ssd'], ['FCOS', 'SSD300-VGG16']):
        r = data['replication'][detector]
        names = ['aggregate', *r['no_adapt']['blocks']]
        x = np.arange(len(names))
        for reference, offset, color, label in [('no_adapt', -.18, '#3175a5', 'Hybrid minus no-adapt'),
                                                ('det_pseudo', .18, '#c45d3b', 'Hybrid minus raw pseudo')]:
            values = [r[reference]['aggregate'], *r[reference]['blocks'].values()]
            ax.bar(x+offset, values, .34, color=color, label=label)
        ax.axhline(0, color='#555555', lw=.8)
        ax.axvline(.5, color='#aaaaaa', linestyle=':', lw=.8)
        ax.set_xticks(x, [n.replace('block_', 'Block ') for n in names], rotation=20)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Macro corruption AP delta (points)')
    axes[0].legend(fontsize=8)
    scope = '1,000 images and five fixed blocks' if data['full_cohort_complete'] else 'SMOKE: partial cohort/blocks only'
    fig.suptitle('T009 K=3: '+scope+' (no AP confidence intervals)')
    fig.tight_layout()
    fig.savefig(args.study/'replication_AP.png', dpi=180)
    fig.savefig(args.study/'replication_AP.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
