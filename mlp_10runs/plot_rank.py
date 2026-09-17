#!/usr/bin/env python
"""Two-panel figure: (a) effective rank of dW and (b) ||dW||_F, per random mode.
Data (10 seeds per mode) is hard-coded from the analyze_rank.py output.
x-axis uses index 0/1/2/3; a 2x2 color legend is placed in panel (b).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# --- per-seed data (10 seeds each) ---------------------------------------
erank = {
    'PGU-Reuse': [61.20, 61.14, 61.45, 61.33, 61.75, 61.42, 61.67, 61.36, 61.69, 61.50],
    'PGU-XoR':   [120.07, 119.98, 120.38, 119.95, 119.94, 120.17, 120.19, 120.09, 119.99, 120.01],
    'Randint':   [119.89, 119.98, 120.21, 120.28, 120.28, 120.29, 120.09, 120.04, 120.32, 120.19],
    'Randn':     [120.06, 120.07, 120.09, 120.20, 119.89, 120.20, 120.14, 120.18, 120.22, 120.12],
}
dW_fro = {
    'PGU-Reuse': [1738.937, 1757.990, 1562.861, 1762.864, 1680.071, 1756.677, 1759.741, 1784.938, 1800.215, 1751.231],
    'PGU-XoR':   [1688.160, 1702.716, 1751.755, 1762.151, 1752.822, 1758.975, 1745.636, 1729.263, 1745.494, 1712.859],
    'Randint':   [1714.557, 1754.055, 1752.470, 1710.268, 1760.470, 1749.690, 1710.687, 1751.460, 1724.381, 1730.367],
    'Randn':     [1759.579, 1761.341, 1756.402, 1774.345, 1781.471, 1760.444, 1800.475, 1766.359, 1743.051, 1777.701],
}

labels = ['PGU-Reuse', 'PGU-XoR', 'Randint', 'Randn']
main_colors = [
    '#C0392B',  # deep brick red   -> PGU-Reuse
    '#2E7D6B',  # deep teal green  -> PGU-XoR
    '#1A5276',  # deep steel blue  -> Randint
    '#B7770D',  # deep gold        -> Randn
]

# --- font / style settings -----------------------------------------------
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(3.8, 2.2),
    gridspec_kw={'wspace': 0.3}
)
x = np.arange(len(labels))


def style_axis(ax):
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=10)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.tick_params(axis='both', length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)


# --- panel (a): effective rank ------------------------------------------
a_mean = [np.mean(erank[l]) for l in labels]
a_std  = [np.std(erank[l])  for l in labels]
ax_a.bar(x, a_mean, 0.52, color=main_colors, alpha=0.85,
         yerr=a_std,
         error_kw=dict(elinewidth=0.5, ecolor='#444444', capsize=2, capthick=0.2),
         zorder=3)
for xi, (m, s) in enumerate(zip(a_mean, a_std)):
    ax_a.text(xi, m + s + 2, f'{m:.1f}',
              ha='center', va='bottom', fontsize=8, color='#333333')
ax_a.set_ylim(0, 175)
ax_a.set_yticks([0, 40, 80, 120])
ax_a.set_ylabel('Effective rank of $\\Delta W$', fontsize=10)
ax_a.text(0, 1.1, '(a)', transform=ax_a.transAxes, fontsize=10, va='top', ha='left')
style_axis(ax_a)

# --- panel (b): Frobenius norm + 2x2 color legend -----------------------
b_mean = [np.mean(dW_fro[l]) / 1e3 for l in labels]   # in units of 10^3
b_std  = [np.std(dW_fro[l])  / 1e3 for l in labels]
ax_b.bar(x, b_mean, 0.52, color=main_colors, alpha=0.85,
         yerr=b_std,
         error_kw=dict(elinewidth=0.5, ecolor='#444444', capsize=2, capthick=0.2),
         zorder=3)
for xi, (m, s) in enumerate(zip(b_mean, b_std)):
    ax_b.text(xi, m + s + 0.03, f'{m:.2f}',
              ha='center', va='bottom', fontsize=8, color='#333333')
ax_b.set_ylim(0, 2.8)
ax_b.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax_b.set_ylabel(r'$\| \Delta W \|_F$ ($\times 10^{3}$)', fontsize=10)
ax_b.text(0, 1.1, '(b)', transform=ax_b.transAxes, fontsize=10, va='top', ha='left')
style_axis(ax_b)

# color legend split: 2 entries in panel (a), 2 in panel (b)
handles = [Patch(facecolor=c, alpha=0.85, label=l) for l, c in zip(labels, main_colors)]
leg_kw = dict(fontsize=7.5, frameon=True, framealpha=1, edgecolor='none',
              handlelength=0.5, handleheight=0.4, handletextpad=0.2,
              columnspacing=0.5, labelspacing=0.25, borderpad=0.2,
              bbox_to_anchor=(0.0, 1.0))   # raise y (e.g. 1.03/1.05) to move higher
ax_a.legend(handles=handles[:2], ncol=1, loc='upper left', **leg_kw)
ax_b.legend(handles=handles[2:], ncol=1, loc='upper left', **leg_kw)

# no tight_layout(): wspace is set via gridspec_kw and bbox_inches='tight' fits everything
fig.savefig('fig_rank.png', dpi=600, bbox_inches='tight')
print('saved fig_rank.png')
