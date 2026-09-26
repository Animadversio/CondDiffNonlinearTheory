"""Held-out loss vs noise for every class-conditional curve (and its unconditional twin).

Reads, all held-out (fit on CIFAR train[:10000], scored on the 10,000-image test split):
  tables/rf_cond_toll2d_heldout.npz              W_U, W_S, W_C and band-4 linear U / S / C
  tables/rf_pixel_band2d_heldout_t7.npz          circulant RF, 7x7 taps, c=128, B=4, unconditional
  tables/rf_pixel_band2d_cond_heldout_vu_t7_c128.npz   same RF, class-conditional (MODE=vu)
Writes figures/cond_heldout_loss_vs_sigma.{png,pdf} and a table twin .csv beside them.

    python scripts/plot_cond_heldout.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = lambda p: os.path.join(ROOT, 'tables', p)
OUT = os.path.join(ROOT, 'figures', 'cond_heldout_loss_vs_sigma')

toll = dict(np.load(T('rf_cond_toll2d_heldout.npz'), allow_pickle=True))
unc = dict(np.load(T('rf_pixel_band2d_heldout_t7.npz'), allow_pickle=True))
cnd = dict(np.load(T('rf_pixel_band2d_cond_heldout_vu_t7_c128.npz'), allow_pickle=True))
sig = [float(s) for s in np.asarray(toll['sigmas']).ravel()]
test = lambda v: float(np.atleast_2d(np.asarray(v, float))[:, 2].mean())   # RF rows: train, resid, test

curves = {   # name -> held-out test loss per sigma
    'rf_vu':   [test(cnd[f'{s}|128|4|t7|vu|g1']) for s in sig],
    'rf_unc':  [test(unc[f'{s}|128|4|t7']) for s in sig],
    'band_S':  list(toll['S|band2d|4'][:, 1]),
    'band_U':  list(toll['U|band2d|4'][:, 1]),
    'W_S':     list(toll['S|W'][:, 1]),
    'W_U':     list(toll['U|W'][:, 1]),
    'band_C':  list(toll['C|band2d|4'][:, 1]),
    'W_C':     list(toll['C|W'][:, 1]),
}
# the RF run stores its own copy of the conditional Wiener; it must be the same split
for s, a in zip(sig, curves['W_S']):
    assert abs(float(cnd[f'linS|{s}'][1]) - a) < 1e-9, 'RF run and toll table use different splits'

# entity -> fixed categorical slot (validated: dataviz validate_palette.js, light, adjacent)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
series = [  # key, label, color, marker, conditional?
    ('rf_vu',  'Circulant RF 7×7, c=128, B=4: class-conditional (vu)', BLUE,    'o', True),
    ('rf_unc', 'Circulant RF 7×7, c=128, B=4: unconditional',          BLUE,    'o', False),
    ('band_S', 'Band-4 linear + class bias (band_S)',                  ORANGE,  's', True),
    ('band_U', 'Band-4 linear, unconditional (band_U)',                ORANGE,  's', False),
    ('W_S',    'Wiener + class mean (W_S)',                            AQUA,    '^', True),
    ('W_U',    'Wiener, unconditional (W_U)',                          AQUA,    '^', False),
    ('band_C', 'Per-class band-4 linear (band_C)',                     YELLOW,  'D', True),
    ('W_C',    'Per-class Wiener (W_C)',                               MAGENTA, 'v', True),
]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), facecolor=SURF)
ref = np.array(curves['W_S'])
for ax, delta in zip(axes, (False, True)):
    ax.set_facecolor(SURF)
    for key, lab, col, mk, is_cond in series:
        y = np.array(curves[key]) - (ref if delta else 0.0)
        ax.plot(sig, y, color=col, lw=1.6, ls='-' if is_cond else (0, (4, 2.5)),
                marker=mk, ms=6 if mk != 'D' else 5, mfc=col if is_cond else SURF,
                mec=SURF if is_cond else col, mew=1.3, label=lab, zorder=3 if is_cond else 2,
                solid_capstyle='round', dash_capstyle='round')
    ax.set_xscale('log')
    ax.set_xticks(sig)
    ax.set_xticklabels([f'{s:g}' for s in sig], rotation=40, ha='right', rotation_mode='anchor')
    ax.minorticks_off()
    ax.set_xlabel('noise level σ')
    ax.grid(True, color=GRID, lw=0.7, ls='-')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].set_yscale('log')
axes[0].set_ylabel('held-out loss  E‖x₀ − D(y)‖²  (per image)')
axes[0].set_title('Held-out loss vs noise', loc='left')
axes[1].axhline(0, color=INK2, lw=0.9, zorder=1)
axes[1].set_ylabel('loss − W_S   (0 = class-conditional Wiener)')
axes[1].set_title('Relative to the class-conditional Wiener W_S', loc='left')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=8, handlelength=3.2,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('CIFAR-10, held out: fit on train[:10000], scored on the 10,000 test images '
             '(solid = uses the class label, dashed = unconditional twin)',
             x=0.01, ha='left', fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.14, 1, 0.95))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + '.png', dpi=200, facecolor=SURF)
fig.savefig(OUT + '.pdf', facecolor=SURF)

with open(OUT + '.csv', 'w') as f:   # table twin of the figure
    f.write('sigma,' + ','.join(k for k, *_ in series) + '\n')
    for i, s in enumerate(sig):
        f.write(f'{s:g},' + ','.join(f'{curves[k][i]:.4f}' for k, *_ in series) + '\n')
print('wrote', OUT + '.png', OUT + '.pdf', OUT + '.csv')
