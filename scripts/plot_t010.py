"""T010 fixed-grid AP versus clean coverage; no AP intervals or selected-only curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.study/'analysis.json').read_text(encoding='utf-8'))
    rows = data['configurations']
    scores = list(dict.fromkeys(r['score'] for r in rows if r['orientation'] is not None))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = {s: plt.get_cmap('tab10')(i) for i, s in enumerate(scores)}
    for ax, detector, label in zip(axes, ['target', 'ssd'], ['FCOS', 'SSD300-VGG16']):
        ax.axvspan(0, 50, color='#888888', alpha=.07)
        ax.axhline(0, color='#555555', lw=.8)
        full = next(r for r in rows if r['config'] == 'full_hybrid')
        ax.axhline(full[detector+'_macro_AP_delta_no_adapt'], color='#777777', lw=.8, ls=':')
        for r in rows:
            if r['score'] not in scores:
                ax.scatter(100*r['clean_coverage'], r[detector+'_macro_AP_delta_no_adapt'], marker='s', c='black', s=35)
                continue
            ax.scatter(100*r['clean_coverage'], r[detector+'_macro_AP_delta_no_adapt'],
                       color=colors[r['score']], marker='o' if r['orientation'] == 'high' else '^',
                       s=30+60*r['nominal_coverage'], edgecolor='black' if r['promising_rule_met'] else 'white', linewidth=.8)
        ax.set(title=label, xlabel='Clean selected coverage (%)', xlim=(-3, 103))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Macro corruption AP delta vs no-adapt (points)')
    handles = [Line2D([], [], marker='o', linestyle='', color=colors[s], label=s) for s in scores]
    handles += [Line2D([], [], marker=m, linestyle='', color='gray', label=l) for m, l in [('o', 'adapt-high'), ('^', 'adapt-low')]]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8)
    scope = '1,000-image development cohort' if data['full_cohort'] else 'SMOKE: partial cohort only'
    fig.suptitle('T010: '+scope+'; all fixed configurations')
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.savefig(args.study/'gate_tradeoff.png', dpi=180)
    fig.savefig(args.study/'gate_tradeoff.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
