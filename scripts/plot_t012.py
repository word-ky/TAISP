"""Standalone half-dose versus frozen random thinning visualization."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    args = p.parse_args()
    a = json.loads((args.study/'analysis.json').read_text(encoding='utf-8'))
    controls = json.loads((args.controls/'analysis.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, d, label in zip(axes, ('target', 'ssd'), ('FCOS', 'SSD300-VGG16')):
        values = [r[d+'_macro_AP_delta_no_adapt'] for r in controls['configurations'] if r['role'] == 'random']
        result = a['random_controls'][d]['aggregate']
        full = a['contrasts']['aggregate'][d]['det_pseudo_clip_radius']['no_adapt']['macro_corruption_AP_delta']
        ax.hist(values, bins=20, color='#7c99b4', edgecolor='white', label='Frozen matched random controls')
        ax.axvline(result['candidate'], color='#b33636', lw=2, label='Fixed half-dose hybrid')
        ax.axvline(result['random']['median'], color='#394d5e', ls=':', label='Random median')
        ax.axvline(full, color='#754491', ls='--', label='Full hybrid')
        ax.set_title(f"{label}: half-dose {result['candidate']:+.4f} AP", fontsize=11)
        ax.set_xlabel('Macro corruption AP delta vs no-adapt (points)')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Number of frozen control draws')
    axes[0].legend(fontsize=7, loc='best')
    scope = '1,000-image development cohort' if a['full_cohort'] else 'SMOKE: partial cohort only'
    fig.suptitle('T012: fixed half dose; '+scope)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ('png', 'pdf'):
        fig.savefig(args.study/('half_dose_controls.'+ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
