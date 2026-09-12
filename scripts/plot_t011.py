"""Matched-selector control distributions, not AP confidence intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    a = json.loads((args.study/'analysis.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, detector, label in zip(axes, ['target', 'ssd'], ['FCOS', 'SSD300-VGG16']):
        values = [r[detector+'_macro_AP_delta_no_adapt'] for r in a['configurations'] if r['role'] == 'random']
        stats = a['aggregate_distributions'][detector]['macro_AP_delta_no_adapt']
        ax.hist(values, bins=20, color='#7c99b4', edgecolor='white', label='Matched random controls')
        ax.axvline(stats['candidate'], color='#b33636', lw=2, label='Frozen confidence candidate')
        ax.axvline(stats['random']['median'], color='#394d5e', ls=':', label='Control median')
        ax.axvline(stats['random']['p95'], color='#555555', ls='--', label='Control 95th percentile')
        ax.set_title(f"{label}: percentile {stats['strict_empirical_percentile']:.1f}%, tail {stats['one_sided_tail']:.4f}", fontsize=10)
        ax.set_xlabel('Macro corruption AP delta vs no-adapt (points)')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Number of control draws')
    axes[0].legend(fontsize=7, loc='best')
    scope = '1,000-image development cohort' if a['full_cohort'] else 'SMOKE: partial cohort only'
    fig.suptitle('T011: '+scope+'; 200 matched controls')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.study/'matched_random.png', dpi=180)
    fig.savefig(args.study/'matched_random.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
