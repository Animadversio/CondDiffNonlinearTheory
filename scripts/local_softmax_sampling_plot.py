"""Figures + summary table for scripts/local_softmax_sampling.py (Heun-30 samples, 2026-09-29).

    python scripts/local_softmax_sampling_plot.py --dataset cifar10     # or ffhq32, afhq32

Reads tables/local_softmax_samples_{ds}.npz (metrics) and the cached final samples in
STORE_DIR/CondDiffNonlinearTheory/local_softmax_samples/{ds}/N{n}_steps{s}_seed0/ (grid only).
Writes
  figures/local_softmax_samples_{ds}_grid.{png,pdf}     same 12 seeds, one row per denoiser
  figures/local_softmax_samples_{ds}_metrics.{png,pdf}  (a,b) on-trajectory R^2 vs sigma against
      each reference U-net, (c) final-sample R^2 against each U-net, (d) copying
  figures/local_softmax_samples_{ds}_summary.csv
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = sys.argv[sys.argv.index('--dataset') + 1] if '--dataset' in sys.argv else 'cifar10'
DSNAME = {'cifar10': 'CIFAR-10', 'ffhq32': 'FFHQ 32×32', 'afhq32': 'AFHQ 32×32'}[DS]
STORE = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
Z = dict(np.load(os.path.join(ROOT, 'tables', f'local_softmax_samples_{DS}.npz'), allow_pickle=True))
N, STEPS = int(Z['meta_NSAMP']), int(Z['meta_STEPS'])
CACHE = os.path.join(STORE, 'CondDiffNonlinearTheory', 'local_softmax_samples', DS,
                     f'N{N}_steps{STEPS}_seed0')
OUT = os.path.join(ROOT, 'figures', f'local_softmax_samples_{DS}')
METHODS = [str(m) for m in Z['meta_methods']]
UNETS = [str(u) for u in Z['meta_unets']]
sig_pix = Z['ts'] / 2                                  # EDM sigma -> [0,1]-pixel sigma

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
UNAME = {'unet-edm': 'EDM U-net (50k)', 'unet-10000_split1': 'U-net 10k (= pool)',
         'unet-full_longtrain': 'U-net 70k long', 'unet-full': 'U-net full',
         'unet-30000_split1': 'U-net 30k'}
STYLE = {  # name, color, marker, linestyle
    'wiener': ('Linear (Wiener)', MAGENTA, 'v', '-'),
    'ls': ('LS (Kamb & Ganguli)', ORANGE, 'o', '-'),
    'luk': ('Lukoianov', AQUA, '^', '-'),
    'global': ('Global softmax', YELLOW, 'D', '-'),
}
for i, u in enumerate(UNETS):
    STYLE[u] = (UNAME.get(u, u), BLUE, 's', ['-', (0, (4, 2))][i % 2])
ORDER = UNETS + ['wiener', 'ls', 'luk', 'global']
ORDER = [m for m in ORDER if m in METHODS]
# the "best" U-net is the reference in panel (a), the pool-matched one in (b)
REFS = sorted(UNETS, key=lambda u: '10000' in u)
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold'})


def se(v, axis=0):
    return np.std(v, axis=axis, ddof=1) / np.sqrt(v.shape[axis])


def style_ax(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------------- grid
NCOL = 12
fig, axes = plt.subplots(len(ORDER), 1, figsize=(NCOL * 0.62 + 1.6, len(ORDER) * 0.72 + 0.5),
                         facecolor=SURF)
for ax, m in zip(axes, ORDER):
    x = torch.load(os.path.join(CACHE, f'{m}.pt'), weights_only=False)['x'][:NCOL]
    x = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
    ax.imshow(np.concatenate(list(x), axis=1), interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel(STYLE[m][0], rotation=0, ha='right', va='center', fontsize=8, color=INK2)
fig.suptitle(f'{DSNAME}: Heun {STEPS} steps, same initial noise in every column',
             x=0.01, ha='left', fontsize=10, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT + '_grid.png', dpi=200, facecolor=SURF)
fig.savefig(OUT + '_grid.pdf', facecolor=SURF)
plt.close(fig)

# ------------------------------------------------------------------------------- metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), facecolor=SURF)
for ax, u in zip(axes[0], REFS):
    style_ax(ax)
    for m in ORDER:
        if m == u:
            continue
        v = Z[f'onr2:{m}|{u}']                          # (steps, N)
        mu, e = v.mean(1), se(v, 1)
        name, c, mk, ls = STYLE[m]
        ax.plot(sig_pix, mu, color=c, ls=ls, lw=1.6, label=name)
        ax.fill_between(sig_pix, mu - e, mu + e, color=c, alpha=0.18, lw=0)
    ax.set_xscale('log'); ax.set_ylim(-0.5, 1.02)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel('noise level σ along the U-net trajectory (pixel units)')
    ax.set_ylabel('R² of D(x_t) vs the U-net\'s D(x_t)')
    ax.set_title(f'On {UNAME.get(u, u)}\'s trajectory', loc='left')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')

ax = axes[1][0]; style_ax(ax)
xpos = np.arange(len(ORDER))
for j, u in enumerate(REFS):
    for i, m in enumerate(ORDER):
        if m == u:
            continue
        v = Z[f'finr2:{m}|{u}']
        off = (j - (len(REFS) - 1) / 2) * 0.22
        ax.errorbar(i + off, v.mean(), yerr=se(v), fmt='s' if j == 0 else 'o',
                    color=STYLE[m][1], mfc=STYLE[m][1] if j == 0 else SURF, ms=6, capsize=2,
                    label=f'vs {UNAME.get(u, u)}' if i == (1 if m == ORDER[0] else 0) or
                    (i == 0 and m != u) else None)
ax.set_xticks(xpos); ax.set_xticklabels([STYLE[m][0] for m in ORDER], rotation=25, ha='right')
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylabel('final-sample R² vs the U-net sample (same seed)')
ax.set_title('Final samples: agreement with each U-net', loc='left')
h = [plt.Line2D([], [], color=INK2, marker='s', ls='', ms=6, label=f'vs {UNAME.get(REFS[0], REFS[0])}  (filled)')]
if len(REFS) > 1:
    h.append(plt.Line2D([], [], color=INK2, marker='o', mfc=SURF, ls='', ms=6,
                        label=f'vs {UNAME.get(REFS[1], REFS[1])}  (hollow)'))
ax.legend(handles=h, frameon=False, fontsize=7.5, loc='lower left')

ax = axes[1][1]; style_ax(ax)
cats = ORDER + ['test']
for i, m in enumerate(cats):
    d1 = Z[f'nn1:{m}'] ** 2
    c = STYLE[m][1] if m in STYLE else MUTED
    q = np.percentile(d1, [25, 50, 75])
    ax.plot([i, i], [q[0], q[2]], color=c, lw=3, solid_capstyle='round', alpha=0.6)
    ax.plot(i, q[1], 'o', color=c, ms=7, mec=SURF)
    cp = 100 * np.mean(Z[f'nn1:{m}'] / Z[f'nn2:{m}'] < 1 / 3)
    ax.annotate(f'{cp:.0f}% copies', (i, q[2]), xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=7.5, color=INK2)
ax.axhline(np.median(Z['nn1:test'] ** 2), color=MUTED, lw=0.9, ls=(0, (2, 2)))
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([STYLE[m][0] if m in STYLE else 'held-out test images' for m in cats],
                   rotation=25, ha='right')
ax.set_ylabel('squared L2 distance to nearest pool image\n(pixel units; median, IQR)')
ax.set_title('Copying: distance to the 10k pool  (copy = d₁/d₂ < ⅓)', loc='left')
fig.suptitle(f'{DSNAME}: samples from analytic denoisers vs U-nets  (Heun {STEPS} steps, '
             f'{N} shared seeds; k, τ = MSE-optimal held-out schedule)',
             x=0.01, ha='left', fontsize=10.5, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT + '_metrics.png', dpi=180, facecolor=SURF)
fig.savefig(OUT + '_metrics.pdf', facecolor=SURF)

# ------------------------------------------------------------------------------ summary
with open(OUT + '_summary.csv', 'w') as f:
    cols = ['method', 'n', 'copy_pct', 'median_nn1_sq', 'iqr_nn1_sq_lo', 'iqr_nn1_sq_hi']
    for u in REFS:
        cols += [f'finR2_vs_{u}', f'finR2_se_vs_{u}', f'finMSE_vs_{u}',
                 f'onR2_mean_vs_{u}', f'onR2_sig0.1_vs_{u}', f'onR2_sig1_vs_{u}']
    f.write(','.join(cols) + '\n')
    i01, i1 = np.argmin(np.abs(np.log(sig_pix / 0.1))), np.argmin(np.abs(np.log(sig_pix / 1.0)))
    for m in cats:
        d1 = Z[f'nn1:{m}'] ** 2
        row = [m, len(d1), 100 * np.mean(Z[f'nn1:{m}'] / Z[f'nn2:{m}'] < 1 / 3),
               np.median(d1), *np.percentile(d1, [25, 75])]
        for u in REFS:
            if m in (u, 'test'):
                row += [''] * 6
                continue
            fr, on = Z[f'finr2:{m}|{u}'], Z[f'onr2:{m}|{u}']
            row += [fr.mean(), se(fr), Z[f'finmse:{m}|{u}'].mean(), on.mean(),
                    on[i01].mean(), on[i1].mean()]
        f.write(','.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in row) + '\n')
print(open(OUT + '_summary.csv').read())
print(f'wrote {OUT}_grid / _metrics / _summary.csv')
