"""LOSS vs NOISE, HELD OUT, CONDITIONAL vs UNCONDITIONAL: 7x7 band RF c=128 B=4 with and
without the label, conditional dense RF k/d=8, and unconditional/conditional Wiener baselines.

    michimin, 2026-09-26: "plot heldout losses vs noises for conditional circulant 7x7 B=4
    c=128 vu, unconditional circulant 7x7 B=4 c=128, linear unconditional and conditional
    linear per class analytic. use the existing plot code used for the unconditional graph"

LAYOUT AND CONVENTIONS ARE COPIED FROM scripts/rf_heldout_band_plot.py (the unconditional
figure, figures/rf_heldout_band_vs_sigma.png): raw loss on the left, excess over the held-out
UNCONDITIONAL linear denoiser W_U on the right, one legend, crossing markers
only where the bracket is two ADJACENT grid points, curves broken at any missing sigma.
That script is not imported -- it has no __main__ guard and would redraw its own figure.

THE SIX CURVES.  The computed curves use TEST-column numbers on the identical
10,000 CIFAR test images, every model fitted on CIFAR train[:10000]:
    W_U   Wiener, unconditional          rf_pixel_band2d_heldout_t7.npz          'linear|{s}'[1]
    W_S   shared-slope conditional       rf_cond_toll2d_heldout.npz              'S|W'[i, 1]
    W_C   per-class analytic Wiener      rf_cond_toll2d_heldout.npz              'C|W'[i, 1]
          (mean AND covariance per class from ~1,000 train images each, so rank <= ~1,000 in
          d=3072; test images are scored with their own labels)
    RF_U  band RF c=128 B=4, 7x7 taps     rf_pixel_band2d_heldout_t7.npz          '{s}|128|4|t7'[:, 2]
    RF_C  the same, MODE=vu (gamma^T U    rf_pixel_band2d_cond_heldout_vu_t7_c128.npz
          inside the relu + V U readout)                                         '{s}|128|4|t7|vu|g1'[:, 2]
    RF_DC dense RF k/d=8, conditional VU rf_pixel_dense_cond_heldout_vu_reported.json
          user-supplied held-out losses rounded to four decimals (2026-09-26).
          No train loss, seed count, or uncertainty is inferred for this curve.
W_U is the right-panel zero reference. W_S is an additional comparison curve on both panels.
W_U, W_C and W_S are cross-checked across the toll and conditional RF source tables.

    python scripts/rf_heldout_cond_plot.py
    python scripts/rf_heldout_cond_plot.py --right-scale symlog
    python scripts/rf_heldout_cond_plot.py --right-scale percent
    python scripts/rf_heldout_cond_plot.py --right-scale symlog --left-scale log

The optional symlog view expands the region near zero without changing the losses.
It writes a separate *_symlog figure. Its y-axis is linear within +/-0.5 loss units
and logarithmic outside. All views shade the RF comparison and put the legend below.
The percent view instead plots 100 * (loss - W_U) / W_U on a linear y-axis, so each
noise level is compared relative to its own baseline loss. It writes separate *_percent files.
The --left-scale log option expands low losses without changing the data. No inset zooms
or explanatory callouts are drawn. Crossings with W_U are log-interpolated from adjacent
measurements, projected down to the right panel's x-axis, and labelled there numerically.
"""
import argparse
import json
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
parser.add_argument('--left-scale', choices=('linear', 'log'), default='linear',
                    help='y-axis scale for raw losses in the left overview only')
args = parser.parse_args()
EXPANDED = args.right_scale == 'symlog'
RELATIVE = args.right_scale == 'percent'
if args.right_scale != 'linear':
    OUT = OUT.replace('.png', f'_{args.right_scale}.png')
if args.left_scale != 'linear':
    OUT = OUT.replace('.png', f'_left{args.left_scale}.png')


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
with open('tables/rf_pixel_dense_cond_heldout_vu_reported.json') as f:
    dense_report = json.load(f)
assert dense_report['k_over_d'] == 8
assert dense_report['sigmas'] == SIGS, 'dense report has a different noise grid'
assert len(dense_report['test_loss']) == len(SIGS)
assert np.isfinite(dense_report['test_loss']).all(), 'nonfinite dense loss'
RDC = dict(zip(SIGS, dense_report['test_loss']))

WU, WS, WC, RU, RC = {}, {}, {}, {}, {}
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
    ws = T['S|W'][i]
    o = get(C, 'linS|{}', s)
    assert abs(o[1] - ws[1]) < 1e-9, f'vu linS at sigma={s} disagrees: {o} vs {ws}'
    WU[s], WS[s], WC[s] = wu[1], ws[1], wc[1]
    GAP[s] = {'W_U': wu[1] - wu[0], 'W_S': ws[1] - ws[0], 'W_C': wc[1] - wc[0]}
    ru, rc = get(U, '{}|128|4|t7', s), get(C, '{}|128|4|t7|vu|g1', s)
    # A missing cell never enters the dict, so its curve is broken there (see segments()).
    if ru is not None:
        ru = np.atleast_2d(ru)
        RU[s], GAP[s]['RF_U'] = ru[:, 2].mean(), ru[:, 2].mean() - ru[:, 1].mean()
    if rc is not None:
        rc = np.atleast_2d(rc)
        RC[s], GAP[s]['RF_C'] = rc[:, 2].mean(), rc[:, 2].mean() - rc[:, 1].mean()

# Marker = model family (o linear, * band RF, D dense RF); line style = conditioning (solid = unconditional,
# dashed = label-conditioned).  Grey linear and the 7x7-band cyan are the colours those two
# curves already have in figures/rf_heldout_band_vs_sigma_c256B3.png.
series = [
    ('Linear, unconditional',                          WU, '#444444', 'o', '-',  2.0),
    ('Linear, conditional (per-class slopes)',          WC, '#8c564b', 'o', '--', 2.0),
    ('Circulant RF, unconditional',                    RU, '#17becf', '*', '-',  1.8),
    ('Circulant RF, conditional',                      RC, '#1f77b4', '*', '--', 1.8),
    ('Dense RF, conditional (8× input width)',          RDC, '#9467bd', 'D', '--', 1.8),
    ('Linear, conditional (shared slope)',             WS, '#e67e22', 'o', '--', 2.0),
]


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


print(' sigma      W_U      W_S      W_C     RF_U     RF_C    RF_DC      (held-out losses)')
for s in SIGS:
    print(f' {s:5.3f} {cell(WU, s)}{cell(WS, s)}{cell(WC, s)}{cell(RU, s)}{cell(RC, s)}{cell(RDC, s)}')
print('Dense VU uses the user-supplied four-decimal losses; no train/seed statistics supplied.')
print('\n excess over the held-out W_U (negative = beats unconditional linear):')
print(' sigma      W_S      W_C     RF_U     RF_C    RF_DC  |  RF_C-RF_U  RF_C-W_C  RF_DC-RF_C')
for s in SIGS:
    print(f' {s:5.3f} {cell(WS, s, WU)}{cell(WC, s, WU)}{cell(RU, s, WU)}{cell(RC, s, WU)}{cell(RDC, s, WU)}  | '
          f'{cell(RC, s, RU, 10)}{cell(RC, s, WC, 10)}{cell(RDC, s, RC, 10)}')
print('\n own generalisation gaps (test - train; RF uses train_resid).  W_C is the most overfit '
      'curve at low sigma:\n per-class covariance from ~1,000 images in d=3072.')
for s in SIGS:
    g = GAP[s]
    print(f' {s:5.3f}  ' + '  '.join(f'{k} {g[k]:+8.4f}' for k in ('W_U', 'W_S', 'W_C', 'RF_U', 'RF_C')
                                     if k in g))
print('\n sign changes (log-interpolated between adjacent grid points):')
for name, d, ref, rname in (('W_S', WS, WU, 'W_U'), ('W_C ', WC, WU, 'W_U'),
                            ('RF_U', RU, WU, 'W_U'), ('RF_C', RC, WU, 'W_U'),
                            ('RF_C', RC, RU, 'RF_U'), ('RF_C', RC, WC, 'W_C'),
                            ('RF_DC', RDC, WU, 'W_U'),
                            ('RF_DC', RDC, RC, 'RF_C')):
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


fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.8))
for lab, d, col, mk, ls, lw in series:
    ms = 8 if mk == '*' else 5
    for k, seg in enumerate(segments([s for s in SIGS if s in d])):
        ax[0].plot(seg, [d[s] for s in seg], ls, color=col, marker=mk, ms=ms, lw=lw)
        ax[1].plot(seg, [right_value(d, s) for s in seg], ls, color=col, marker=mk, ms=ms, lw=lw,
                   label=lab if k == 0 else None)

for panel in ax:
    panel.set_xscale('log')
    panel.set_xlabel('Noise level', labelpad=28)
    panel.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    panel.grid(alpha=0.22)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=9)

ax[0].set_title('Held-out loss', fontsize=12)
ax[0].set_ylabel('Loss' + (' (log scale)' if args.left_scale == 'log' else ''))
ax[0].set_xticks([0.127, 0.452, 1, 2.212, 5], labels=['0.127', '0.452', '1', '2.212', '5'])
if args.left_scale == 'log':
    ax[0].set_yscale('log')
    ax[0].set_yticks([8, 10, 20, 40, 80, 140], labels=['8', '10', '20', '40', '80', '140'])
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

EX = [right_value(d, s) for _, d, *_ in series for s in SIGS if s in d]
YLO, YHI = np.floor(min(EX)) - 0.8, max(EX) + 0.7
if RELATIVE:
    YLO, YHI = np.floor(min(EX)) - 2, max(EX) + 3
ax[1].set_ylim(YLO, YHI)
ax[1].set_title('Difference from unconditional linear', fontsize=12)
ax[1].set_ylabel('Loss difference')
ax[1].axhline(0, color='#444444', lw=2)
if EXPANDED:
    ax[1].set_yscale('symlog', linthresh=0.5, linscale=1.0)
    ax[1].set_ylim(min(YLO, -8), max(YHI, 4))
    ticks = [-6, -3, -1, -0.5, 0, 0.5, 1, 2, 4]
    ax[1].set_yticks(ticks, labels=[f'{t:g}' for t in ticks])
    ax[1].set_ylabel('Loss difference (symmetric log scale)')
if RELATIVE:
    ax[1].set_ylabel('Relative loss difference (%)')
ax[1].axhspan(ax[1].get_ylim()[0], 0, color='#2ca02c', alpha=0.05, zorder=0)
for seg in segments([s for s in SIGS if s in RU and s in RC]):
    ax[1].fill_between(seg, [right_value(RU, s) for s in seg],
                       [right_value(RC, s) for s in seg],
                       color='#1f77b4', alpha=0.12, zorder=1)

# Keep ordinary ticks clear of the crossing labels near 1 and 2.2.
# Crossing positions are estimates from raw loss differences, not measured extra points.
ax[1].set_xticks([0.127, 0.452, 5], labels=['0.127', '0.452', '5'])
crossing_marks = sorted((xc, col) for _, d, col, *_ in series[1:]
                        for xc, *_ in crossings(d, WU))
previous_x = None
row = 0
for xc, col in crossing_marks:
    # Draw downward from the zero crossing to the actual bottom x-axis.
    ax[1].vlines(xc, ax[1].get_ylim()[0], 0, colors=col,
                 linestyles='--', linewidth=1.1, alpha=0.85, zorder=2)
    ax[1].plot([xc, xc], [0, -0.014], transform=ax[1].get_xaxis_transform(),
               color=col, lw=1.1, clip_on=False)
    # The band and dense RF crossings are only about 0.06 apart; stagger their
    # numeric labels vertically while retaining their exact horizontal positions.
    row = 1 - row if previous_x is not None and np.log(xc / previous_x) < 0.14 else 0
    ax[1].annotate(f'{xc:.3f}', (xc, 0), xycoords=ax[1].get_xaxis_transform(),
                   xytext=(0, -9 - 14 * row), textcoords='offset points',
                   ha='center', va='top', color=col, fontsize=9,
                   annotation_clip=False)
    previous_x = xc

handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=9, loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, 0.015), frameon=False)
fig.tight_layout(rect=(0, 0.16, 1, 1))
os.makedirs('figures', exist_ok=True)
fig.savefig(OUT, dpi=160)
fig.savefig(OUT.replace('.png', '.pdf'))
print(f'\nwrote {OUT} (+ .pdf)')
