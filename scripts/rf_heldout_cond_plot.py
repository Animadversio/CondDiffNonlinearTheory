"""LOSS vs NOISE, HELD OUT, CONDITIONAL vs UNCONDITIONAL: 7x7 band RF c=128 B=4 with and
without the label, against the unconditional Wiener and the per-class analytic Wiener.

    michimin, 2026-09-26: "plot heldout losses vs noises for conditional circulant 7x7 B=4
    c=128 vu, unconditional circulant 7x7 B=4 c=128, linear unconditional and conditional
    linear per class analytic. use the existing plot code used for the unconditional graph"

LAYOUT AND CONVENTIONS ARE COPIED FROM scripts/rf_heldout_band_plot.py (the unconditional
figure, figures/rf_heldout_band_vs_sigma.png): raw loss on the left, excess over the held-out
UNCONDITIONAL linear denoiser on the right, one legend on the right panel, crossing markers
only where the bracket is two ADJACENT grid points, curves broken at any missing sigma.
That script is not imported -- it has no __main__ guard and would redraw its own figure.

THE FOUR CURVES.  Every value is a TEST-column number on the identical 10,000 CIFAR test
images, every model fitted on CIFAR train[:10000]; nothing is hand-transcribed.
    W_U   Wiener, unconditional          rf_pixel_band2d_heldout_t7.npz          'linear|{s}'[1]
    W_C   per-class analytic Wiener      rf_cond_toll2d_heldout.npz              'C|W'[i, 1]
          (mean AND covariance per class from ~1,000 train images each, so rank <= ~1,000 in
          d=3072; test images are scored with their own labels)
    RF_U  band RF c=128 B=4, 7x7 taps     rf_pixel_band2d_heldout_t7.npz          '{s}|128|4|t7'[:, 2]
    RF_C  the same, MODE=vu (gamma^T U    rf_pixel_band2d_cond_heldout_vu_t7_c128.npz
          inside the relu + V U readout)                                         '{s}|128|4|t7|vu|g1'[:, 2]
W_U and W_C are each stored in more than one table; they are cross-checked, not assumed.

    python scripts/rf_heldout_cond_plot.py
    python scripts/rf_heldout_cond_plot.py --right-scale symlog
    python scripts/rf_heldout_cond_plot.py --right-scale percent
    python scripts/rf_heldout_cond_plot.py --left-zoom
    python scripts/rf_heldout_cond_plot.py --right-scale symlog --left-zoom
    python scripts/rf_heldout_cond_plot.py --right-scale symlog --left-scale log --left-zoom

The optional symlog view expands the region near zero without changing the losses.
It writes a separate *_symlog figure, shades the RF comparison, and moves the legend
below the panels. Its y-axis is linear within +/-0.5 loss units and logarithmic outside.
The percent view instead plots 100 * (loss - W_U) / W_U on a linear y-axis, so each
noise level is compared relative to its own baseline loss. A low-noise inset plots
the RF conditional-minus-unconditional contrast directly, in percentage points of
that same W_U baseline. It writes separate *_percent files.
The --left-zoom option changes only the left panel: magnified raw-loss comparisons
at the lowest and highest measured noise levels, plus a label for the small RF gap
at sigma=0.452. The overview retains its linear loss axis unless --left-scale log
is selected. A log overview expands low losses; the independent linear endpoint
zooms also resolve the high-loss gaps, which a log scale alone would compress.
The selected right panel
is retained exactly, including its shading. It writes separate *_leftzoom files.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGS = [0.127, 0.452, 0.621, 0.853, 1.172, 1.610, 2.212, 5.0]
OUT = 'figures/rf_heldout_cond_vs_sigma_c128B4.png'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--right-scale', choices=('linear', 'symlog', 'percent'), default='linear',
                    help='absolute excess loss on a linear/symlog axis, or percent excess over W_U')
parser.add_argument('--left-zoom', action='store_true',
                    help='magnify low/high-noise raw losses on the left; preserve the selected right panel')
parser.add_argument('--left-scale', choices=('linear', 'log'), default='linear',
                    help='y-axis scale for raw losses in the left overview only')
args = parser.parse_args()
EXPANDED = args.right_scale == 'symlog'
RELATIVE = args.right_scale == 'percent'
OUTSIDE_LEGEND = EXPANDED or RELATIVE
if OUTSIDE_LEGEND:
    OUT = OUT.replace('.png', f'_{args.right_scale}.png')
if args.left_scale != 'linear':
    OUT = OUT.replace('.png', f'_left{args.left_scale}.png')
if args.left_zoom:
    OUT = OUT.replace('.png', '_leftzoom.png')


def spellings(s):
    return [f'{s:g}', str(s), repr(s), f'{s:.3f}', f'{s:.2f}', f'{s:.4f}']


def get(store, fmt, s):
    """Key on the PHYSICAL sigma, never a row index (`5.0` is stored as '5.0', f'{5.0:g}'
    renders '5')."""
    for k in spellings(s):
        if fmt.format(k) in store:
            return np.asarray(store[fmt.format(k)], float)
    return None


def load(p):
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


U = load('tables/rf_pixel_band2d_heldout_t7.npz')                 # uncond band, 7x7 taps
C = load('tables/rf_pixel_band2d_cond_heldout_vu_t7_c128.npz')    # vu band + linU/linS/linC
T = load('tables/rf_cond_toll2d_heldout.npz')                     # U|W, S|W, C|W  [train, test]
assert U and C and T, 'a table is missing'
TS = [float(t) for t in T['sigmas']]

WU, WC, RU, RC = {}, {}, {}, {}
GAP = {}                      # own generalisation gaps, test - train(resid)
for s in SIGS:
    i = int(np.argmin([abs(np.log(t / s)) for t in TS]))
    assert abs(TS[i] / s - 1) < 1e-6, f'toll table has no sigma={s}'
    wu = get(U, 'linear|{}', s)
    for tag, o in (('vu linU', get(C, 'linU|{}', s)), ('toll U|W', T['U|W'][i])):
        assert abs(o[1] - wu[1]) < 1e-9, f'{tag} at sigma={s} disagrees: {o} vs {wu}'
    wc = T['C|W'][i]
    o = get(C, 'linC|{}', s)
    assert abs(o[1] - wc[1]) < 1e-9, f'vu linC at sigma={s} disagrees: {o} vs {wc}'
    WU[s], WC[s] = wu[1], wc[1]
    GAP[s] = {'W_U': wu[1] - wu[0], 'W_C': wc[1] - wc[0]}
    ru, rc = get(U, '{}|128|4|t7', s), get(C, '{}|128|4|t7|vu|g1', s)
    # A missing cell never enters the dict, so its curve is broken there (see segments()).
    if ru is not None:
        ru = np.atleast_2d(ru)
        RU[s], GAP[s]['RF_U'] = ru[:, 2].mean(), ru[:, 2].mean() - ru[:, 1].mean()
    if rc is not None:
        rc = np.atleast_2d(rc)
        RC[s], GAP[s]['RF_C'] = rc[:, 2].mean(), rc[:, 2].mean() - rc[:, 1].mean()

# Marker = model family (o linear, * band RF); line style = conditioning (solid = unconditional,
# dashed = label-conditioned).  Grey linear and the 7x7-band cyan are the colours those two
# curves already have in figures/rf_heldout_band_vs_sigma_c256B3.png.
series = [
    ('linear (Wiener) $W_U$, held out',                  WU, '#444444', 'o', '-',  2.0),
    ('per-class linear Wiener $W_C$ (analytic), held out', WC, '#8c564b', 'o', '--', 2.0),
    ('band RF  c=128  B=4 (7x7), unconditional',          RU, '#17becf', '*', '-',  1.8),
    ('band RF  c=128  B=4 (7x7), conditional (vu)',       RC, '#1f77b4', '*', '--', 1.8),
]
SHORT = {id(WC): 'per-class $W_C$', id(RU): 'uncond. band RF', id(RC): 'cond. band RF'}


def segments(ss):
    """Maximal runs of points ADJACENT on SIGS -- a line segment is a claim about what happens
    between its endpoints, so a curve with a hole is broken there, never bridged."""
    if not ss:
        return []
    idx = [SIGS.index(s) for s in ss]
    out, run = [], [ss[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            run.append(SIGS[b])
        else:
            out.append(run)
            run = [SIGS[b]]
    out.append(run)
    return out


def crossings(d, ref):
    """Every sigma where d - ref changes sign, log-interpolated, and ONLY across a bracket of
    two adjacent grid points (a wider bracket is not a location).  Unlike the unconditional
    figure the sign change can go either way here, so the direction is returned: -1 = d goes
    from above ref to below it as sigma grows (starts beating ref), +1 = the reverse."""
    x = [(s, d[s] - ref[s]) for s in SIGS if s in d and s in ref]
    out = []
    for (s0, e0), (s1, e1) in zip(x, x[1:]):
        if SIGS.index(s1) != SIGS.index(s0) + 1 or (e0 < 0) == (e1 < 0):
            continue
        xc = float(np.exp(np.log(s0) + (np.log(s1) - np.log(s0)) * (-e0) / (e1 - e0)))
        out.append((xc, -1 if e0 > 0 else +1, s0, s1))
    return out


def cell(d, s, ref=None, w=9):
    if s not in d:
        return ' ' * (w - 4) + '----'
    return f'{d[s] - ref[s]:+{w}.4f}' if ref is not None else f'{d[s]:{w}.4f}'


print(' sigma      W_U      W_C     RF_U     RF_C      (test column, 10k CIFAR test images)')
for s in SIGS:
    print(f' {s:5.3f} {cell(WU, s)}{cell(WC, s)}{cell(RU, s)}{cell(RC, s)}')
print('\n excess over the held-out W_U (negative = beats unconditional linear):')
print(' sigma      W_C     RF_U     RF_C  |  RF_C-RF_U  RF_C-W_C  (conditioning gain, cond RF vs cond linear)')
for s in SIGS:
    print(f' {s:5.3f} {cell(WC, s, WU)}{cell(RU, s, WU)}{cell(RC, s, WU)}  | '
          f'{cell(RC, s, RU, 10)}{cell(RC, s, WC, 10)}')
print('\n own generalisation gaps (test - train; RF uses train_resid).  W_C is the most overfit '
      'curve at low sigma:\n per-class covariance from ~1,000 images in d=3072.')
for s in SIGS:
    g = GAP[s]
    print(f' {s:5.3f}  ' + '  '.join(f'{k} {g[k]:+8.4f}' for k in ('W_U', 'W_C', 'RF_U', 'RF_C')
                                     if k in g))
print('\n sign changes (log-interpolated between adjacent grid points):')
for name, d, ref, rname in (('W_C ', WC, WU, 'W_U'), ('RF_U', RU, WU, 'W_U'),
                            ('RF_C', RC, WU, 'W_U'), ('RF_C', RC, RU, 'RF_U'),
                            ('RF_C', RC, WC, 'W_C')):
    xs = crossings(d, ref)
    if not xs:
        side = 'above' if all(d[s] > ref[s] for s in SIGS if s in d) else 'below'
        print(f'   {name} vs {rname}: no sign change on this grid ({side} it at every sigma)')
    for xc, dr, s0, s1 in xs:
        verb = 'starts beating' if dr < 0 else 'falls behind'
        print(f'   {name} {verb} {rname} at sigma = {xc:.3f}  (bracket {s0:g}-{s1:g}; margin at '
              f'{s1:g}: {d[s1] - ref[s1]:+.4f})')

def right_value(d, s):
    excess = d[s] - WU[s]
    if RELATIVE:
        assert WU[s] > 0, f'percentage excess needs a positive baseline at sigma={s}'
        return 100 * excess / WU[s]
    return excess


fig, ax = plt.subplots(1, 2, figsize=(13.2, 6.1 if OUTSIDE_LEGEND else 5.2))
for lab, d, col, mk, ls, lw in series:
    ms = 8 if mk == '*' else 5
    for k, seg in enumerate(segments([s for s in SIGS if s in d])):
        ax[0].plot(seg, [d[s] for s in seg], ls, color=col, marker=mk, ms=ms, lw=lw)
        ax[1].plot(seg, [right_value(d, s) for s in seg], ls, color=col, marker=mk, ms=ms, lw=lw,
                   label=lab if k == 0 else None)

ax[0].set_xscale('log')
ax[0].set_xlabel(r'pixel noise $\sigma$')
ax[0].set_ylabel(r'held-out loss  $\mathbb{E}\,\|x_0-\hat{x}_0\|^2$')
ax[0].set_title('raw held-out loss vs noise\n(all fitted on 10k train, scored on the 10k CIFAR test set)',
                fontsize=10)
left_legend_note = ax[0].annotate(
    'the four curves are labelled ' + ('below' if OUTSIDE_LEGEND else 'on the right panel'),
    (0.025, 0.955), xycoords='axes fraction', fontsize=7.5, color='#444444', va='top')
ax[0].grid(alpha=0.3)

EX = [right_value(d, s) for _, d, *_ in series for s in SIGS if s in d]
YLO, YHI = np.floor(min(EX)) - 0.8, max(EX) + 1.9
if RELATIVE:
    YLO, YHI = np.floor(min(EX)) - 2, max(EX) + 3
ax[1].axhline(0, color='#444444', lw=2)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'pixel noise $\sigma$')
ax[1].set_ylabel('loss $-$ held-out linear $W_U$')
ax[1].set_title('excess over the held-out unconditional linear denoiser\n(same test set both sides, '
                'so the train/test trace offset cancels)', fontsize=10)
ax[1].grid(alpha=0.3)
ax[1].axhspan(YLO, 0, color='#2ca02c', alpha=0.05, zorder=0)
ax[1].set_ylim(YLO, YHI)
if EXPANDED:
    ax[1].set_yscale('symlog', linthresh=0.5, linscale=1.0)
    ax[1].set_ylim(min(YLO, -8), max(YHI, 6))
    ticks = [-6, -3, -1, -0.5, 0, 0.5, 1, 2]
    ax[1].set_yticks(ticks, labels=[f'{t:g}' for t in ticks])
    ax[1].set_ylabel('loss $-$ held-out linear $W_U$ (symmetric-log scale)')
    ax[1].set_title('Excess loss: expanded near zero\n'
                    'linear within $\\pm 0.5$; logarithmic outside', fontsize=10)
if OUTSIDE_LEGEND:
    # Use the same measured points as the lines and retain any missing-data breaks.
    for seg in segments([s for s in SIGS if s in RU and s in RC]):
        ax[1].fill_between(seg, [right_value(RU, s) for s in seg],
                           [right_value(RC, s) for s in seg],
                           color='#1f77b4', alpha=0.12, zorder=1)
if RELATIVE:
    ax[1].set_title('Relative excess loss vs noise\n'
                    r'$100\,(L-L_{W_U})/L_{W_U}$; baseline evaluated at each $\sigma$', fontsize=10)
    ax[1].set_ylabel('excess loss relative to linear $W_U$ (%)')
    ax[1].set_yticks([-5, 0, 5, 10, 15, 20, 25])
    ax[1].set_xticks([0.127, 0.452, 1, 2.212, 5], labels=['0.127', '0.452', '1', '2.212', '5'])
    ax[1].tick_params(axis='x', labelsize=9)
    for _, d, col, *_ in series[1:]:
        s = SIGS[0]
        if s in d:
            value = right_value(d, s)
            ax[1].annotate(f'{value:.2f}%', (s, value),
                           xytext=(9, 10 if d is RU else 3), textcoords='offset points',
                           color=col, fontsize=9, fontweight='bold',
                           bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))

    # A direct contrast reveals small RF differences without needing to resolve two
    # nearly coincident curves. It uses the SAME denominator as the main panel.
    detail_sigs = [s for s in SIGS if 0.45 <= s <= 0.86 and s in RU and s in RC]
    if detail_sigs:
        inset = ax[1].inset_axes([0.43, 0.44, 0.53, 0.43])
        inset.set_facecolor('#fbfcfe')
        inset.set_title('Low-noise RF gap\nconditional $-$ unconditional', fontsize=8.5, pad=6)
        inset.axhline(0, color='#666666', lw=1)
        for seg in segments(detail_sigs):
            delta = [100 * (RC[s] - RU[s]) / WU[s] for s in seg]
            inset.plot(seg, delta, '-D', color='#5b4a82', ms=4.5, lw=1.6)
            for s, value in zip(seg, delta):
                inset.annotate(f'{value:+.2f}', (s, value), xytext=(0, 9),
                               textcoords='offset points', ha='center', fontsize=8,
                               color='#5b4a82', fontweight='bold')
        inset.set_xscale('log')
        inset.set_xlim(0.41, 0.94)
        inset.set_ylim(-1.5, 0.6)
        inset.set_xticks(detail_sigs, labels=[f'{s:g}' for s in detail_sigs])
        inset.set_yticks([-1, -0.5, 0, 0.5])
        inset.minorticks_off()
        inset.set_xlabel(r'pixel noise $\sigma$', fontsize=8, labelpad=3)
        inset.set_ylabel('RF gap (percentage points)', fontsize=8, labelpad=3)
        inset.tick_params(labelsize=7.5)
        inset.grid(alpha=0.18)
        inset.set_axisbelow(True)
        for spine in inset.spines.values():
            spine.set_color('#9aa8bc')
    ax[1].annotate('below zero = beats linear $W_U$', (0.025, 0.035),
                   xycoords='axes fraction', fontsize=8, color='#2ca02c')
# Crossing markers against the zero line, staggered in y in the headroom above the curves.
# The percent view omits these: interpolating normalized endpoints would give
# different crossing estimates than the original absolute-loss interpolation.
n = 0
for lab, d, col, mk, ls, lw in ([] if RELATIVE else series[1:]):
    for xc, dr, s0, s1 in crossings(d, WU):
        if EXPANDED:
            # Mixed coordinates keep annotation headroom independent of the nonlinear axis.
            yfrac = 0.97 - 0.05 * n
            n += 1
            right = xc > 1.5
            ax[1].annotate('', xy=(xc, 0), xycoords='data',
                           xytext=(xc, yfrac), textcoords=('data', 'axes fraction'),
                           arrowprops=dict(arrowstyle='-', color=col, ls=':', lw=1.1))
            ax[1].annotate(f'{SHORT[id(d)]} crosses at $\\sigma$={xc:.3f}',
                           (xc, yfrac), xycoords=('data', 'axes fraction'),
                           xytext=(-8 if right else 8, 0), textcoords='offset points',
                           ha='right' if right else 'left', va='center', fontsize=7.5,
                           color=col, zorder=6,
                           bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.2))
            continue
        ylab = YHI - 0.45 - 0.5 * n
        n += 1
        # From the zero line (where the crossing IS) up to its label -- drawn from the bottom
        # it ran through the green annotation.
        ax[1].axvline(xc, color=col, ls=':', lw=1.1, alpha=0.8, ymin=(0 - YLO) / (YHI - YLO),
                      ymax=(ylab - YLO) / (YHI - YLO))
        right = xc > 1.5          # near the right edge the label goes on the left of its line
        ax[1].annotate(f'{SHORT[id(d)]} crosses at $\\sigma$={xc:.3f}', (xc, ylab),
                       xytext=(-8 if right else 8, 0), textcoords='offset points',
                       ha='right' if right else 'left', va='center', fontsize=7.5, color=col,
                       zorder=6, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))
# To the RIGHT of the lower-left legend (at sigma=0.6 it sat underneath it).
if EXPANDED:
    ax[1].annotate('below zero = beats linear $W_U$', (0.025, 0.035),
                   xycoords='axes fraction', fontsize=8, color='#2ca02c')
    ax[1].annotate('blue shading = gap between RF curves', (0.025, 0.095),
                   xycoords='axes fraction', fontsize=8, color='#1f77b4')
    # Quantify the endpoint comparison in the original loss units.
    if SIGS[-1] in RU and SIGS[-1] in RC:
        s = SIGS[-1]
        lo, hi = RC[s] - WU[s], RU[s] - WU[s]
        ax[1].annotate('', xy=(s * 1.04, lo), xytext=(s * 1.04, hi),
                       arrowprops=dict(arrowstyle='|-|', color='#1f77b4', lw=1.2))
        ax[1].annotate(f'RF gap at $\\sigma=5$:\n{hi - lo:.2f} loss units', (0.82, 0.61),
                       xycoords='axes fraction', ha='center', fontsize=8,
                       color='#1f77b4', bbox=dict(fc='white', ec='none', alpha=0.9, pad=2))
if OUTSIDE_LEGEND:
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.01), frameon=False,
               title='solid = unconditional,  dashed = label-conditioned', title_fontsize=8)
else:
    ax[1].annotate('beats held-out linear $W_U$', (1.12, YLO + 0.3), fontsize=8.5, color='#2ca02c')
    ax[1].legend(fontsize=8.5, loc='lower left',
                 title='solid = unconditional,  dashed = label-conditioned', title_fontsize=7.5)

fig.suptitle('Held-out denoising loss vs noise level, label-conditioned vs unconditional, CIFAR-10 '
             'raw pixels (d=3072, train[:10000] -> 10,000 test images)', fontsize=11)
fig.tight_layout(rect=(0, 0.14 if OUTSIDE_LEGEND else 0, 1, 0.95))
if args.left_scale == 'log':
    # Change only the left axis AFTER layout; the right panel stays pixel-identical.
    ax[0].set_yscale('log')
    ax[0].set_yticks([8, 10, 20, 40, 80, 140], labels=['8', '10', '20', '40', '80', '140'])
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[0].set_ylabel(r'held-out loss  $\mathbb{E}\,\|x_0-\hat{x}_0\|^2$ (log scale)')
    ax[0].set_title('raw held-out loss vs noise (log y-axis)\n'
                    '(all fitted on 10k train, scored on the 10k CIFAR test set)', fontsize=10)
if args.left_zoom:
    # Apply after layout so the selected right panel keeps its exact position,
    # size, labels, and rendering. All values remain in raw-loss units.
    ax[0].set_xticks([0.127, 0.452, 1, 2.212, 5], labels=['0.127', '0.452', '1', '2.212', '5'])
    ax[0].tick_params(axis='x', labelsize=9)
    left_legend_note.set_visible(False)
    overview_title = ('raw held-out loss (log y-axis) with endpoint zooms' if args.left_scale == 'log'
                      else 'raw held-out loss with magnified endpoint comparisons')
    ax[0].set_title(overview_title + '\n'
                    '(all fitted on 10k train, scored on the 10k CIFAR test set)', fontsize=10)

    # Independent, explicitly labelled linear scales make BOTH endpoint gaps
    # readable. Model rows separate nearly coincident marks without offsetting
    # any measured loss along its numerical axis.
    left_details = []
    low_bounds = ([0.10, 0.65, 0.37, 0.25] if args.left_scale == 'log'
                  else [0.12, 0.60, 0.38, 0.28])
    for s, name, bounds in (
            (SIGS[0], 'Low noise', low_bounds),
            (SIGS[-1], 'High noise', [0.61, 0.115, 0.35, 0.28])):
        if not all(s in d for d in (WU, WC, RU, RC)):
            continue
        left_detail = ax[0].inset_axes(bounds)
        left_details.append((s, left_detail))
        left_detail.set_in_layout(False)
        left_detail.set_facecolor('#fbfcfe')
        left_detail.set_title(f'{name}: $\\sigma={s:g}$\n'
                              f'$RF_C - RF_U = {RC[s] - RU[s]:+.4f}$',
                              fontsize=8, pad=7)
        rows = [series[0], series[2], series[3], series[1]]
        for row, (_, d, color, marker, _, _) in enumerate(rows):
            left_detail.plot(d[s], row, marker=marker, color=color,
                             ms=6.5 if marker == '*' else 4.5, ls='none')
            left_detail.annotate(f'{d[s]:.4f}', (d[s], row), xytext=(6, 0),
                                 textcoords='offset points', va='center',
                                 fontsize=7.5, color=color)
        left_detail.set_yticks(range(4), labels=[r'$W_U$', r'$RF_U$', r'$RF_C$', r'$W_C$'])
        left_detail.set_ylim(3.55, -0.55)
        values = [d[s] for _, d, *_ in rows]
        span = max(values) - min(values)
        pad = max(span, 0.01)
        left_detail.set_xlim(min(values) - 0.1 * pad, max(values) + 0.6 * pad)
        left_detail.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        left_detail.set_xlabel('held-out loss (linear zoom)', fontsize=7.5, labelpad=3)
        left_detail.tick_params(labelsize=7.5)
        left_detail.grid(axis='x', alpha=0.2)
        left_detail.set_axisbelow(True)
        for spine in left_detail.spines.values():
            spine.set_color('#a1a8b3')
    s = 0.452
    if s in RU and s in RC:
        ax[0].annotate(f'At $\\sigma={s:g}$:\n'
                       f'$RF_C - RF_U = {RC[s] - RU[s]:+.4f}$',
                       xy=(s, (RC[s] + RU[s]) / 2), xycoords='data',
                       xytext=(0.10, 0.47) if args.left_scale == 'log' else (0.13, 0.39),
                       textcoords='axes fraction',
                       ha='left', va='center', fontsize=8, color='#1f77b4',
                       arrowprops=dict(arrowstyle='->', color='#7f8a96', lw=0.9),
                       bbox=dict(fc='white', ec='none', alpha=0.95, pad=3))
os.makedirs('figures', exist_ok=True)
fig.savefig(OUT, dpi=160)
fig.savefig(OUT.replace('.png', '.pdf'))
print(f'\nwrote {OUT} (+ .pdf)')
