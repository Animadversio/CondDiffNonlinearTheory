"""LOSS vs NOISE, HELD OUT: linear / EDM / plain 2-D c=3072 / band c=512 B=1 and B=2.

    michimin, 2026-09-23 21:06: "graph loss against noise for held out linear, test edm,
    c=3072 plain 2-d, c=512 B=1 and c=512 B=2"

EVERY CURVE IS A TEST-COLUMN NUMBER ON THE IDENTICAL 10,000 CIFAR TEST IMAGES, so the
comparison between any two of them is exact -- no trace offset, no scale mixing (the offset
Tr(S_test) = 189.794 vs Tr(S_train) = 191.522 is common to all five and cancels in every
difference plotted here).  Nothing is hand-transcribed; every value is read from the npz.

    python scripts/rf_heldout_band_plot.py

THE ONE ASYMMETRY, AND IT IS NOT FIXABLE FROM THESE TABLES: the four RF/linear curves are
fitted on CIFAR train[:10000]; EDM saw all 50,000.  It is the analogue of the linear-on-50k
row, a 5:1 data handicap in its favour, and its own memorisation gap (train-seen minus
test-unseen) peaks at +1.76 near sigma=0.85 -- larger than the free Wiener's +0.465 there.
Read it as "what a well-trained nonlinear denoiser achieves", not as a matched competitor.

WHY TWO PANELS.  Raw loss vs sigma is what was asked for and is the left panel, but over
8 <= L <= 137 the five curves sit within ~8 of each other at every sigma and the eye cannot
separate them; the excess-over-Wiener panel on the right is where every statement in the
caption is actually visible.  Same reason figures/rf_heldout_vs_sigma.png is two-panelled.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGS = [0.127, 0.452, 0.621, 0.853, 1.172, 1.610, 2.212, 5.0]
OUT = 'figures/rf_heldout_band_vs_sigma.png'


def spellings(s):
    return [f'{s:g}', str(s), repr(s), f'{s:.3f}', f'{s:.2f}', f'{s:.4f}']


def get(store, fmt, s):
    """Key on the PHYSICAL sigma, never on a row index -- the tables disagree on spelling
    (`5.0` is stored as '5.0' but f'{5.0:g}' renders '5') and on grid order."""
    for k in spellings(s):
        if fmt.format(k) in store:
            return np.asarray(store[fmt.format(k)], float)
    return None


def load(p):
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


H = load('tables/rf_pixel_heldout.npz')          # plain arms + Wiener
E = load('tables/edm_pixel_heldout.npz')         # EDM U-net, NREP=6 sweeps
B = load('tables/rf_pixel_band2d_heldout.npz')   # band, `{sigma}|{c}|{B}`
D = load('tables/rf_pixel_dense_heldout.npz')    # dense, `{sigma}|{k/d}|dense`, k/d a FLOAT
                                                 # string ('8.0', not '8') -- check before
                                                 # indexing, the spelling differs from every
                                                 # other table in the project.

sig = np.array(SIGS)
W, EDM, EDMSE, P2, B1, B2, DN = ({} for _ in range(7))
NS = {}
for s in SIGS:
    w = get(H, 'linear|{}', s)
    # The SAME Wiener row is stored in all four tables (linear_split has no RNG).  Verify
    # rather than assume -- a mismatch would mean the splits moved underneath one of them.
    for tag, z in (('band', B), ('edm', E), ('dense', D)):
        o = get(z, 'linear|{}', s)
        assert o is None or abs(o[1] - w[1]) < 1e-9, f'{tag} linear|{s} disagrees: {o} vs {w}'
    e = get(E, 'uncond-ve|{}', s)
    p = np.atleast_2d(get(H, '{}|3072|2d', s))
    b1 = np.atleast_2d(get(B, '{}|512|1', s))
    b2 = np.atleast_2d(get(B, '{}|512|2', s))
    dn = np.atleast_2d(get(D, '{}|8.0|dense', s))
    W[s], EDM[s], EDMSE[s] = w[1], e[2], e[3]
    P2[s], B1[s], B2[s] = p[:, 2].mean(), b1[:, 2].mean(), b2[:, 2].mean()
    DN[s] = dn[:, 2].mean()
    NS[s] = (len(p), len(b1), len(b2), len(dn))

series = [
    ('linear (Wiener), held out', W,   '#444444', 'o', '-',  2.0),
    ('EDM U-net (trained on 50k)', EDM, '#d62728', 'D', '--', 2.0),
    ('dense RF  k/d=8',           DN,  '#ff7f0e', 'P', '-.', 1.8),
    ('plain 2-D RF  c=3072',      P2,  '#1f77b4', 's', '-',  1.8),
    ('band RF  c=512  B=1',       B1,  '#2ca02c', '^', '-',  1.8),
    ('band RF  c=512  B=2',       B2,  '#9467bd', 'v', '-',  1.8),
]

print(' sigma   W_test    EDM_te  (se)  dense k/d=8   2D c=3072   B=1 c=512   B=2 c=512'
      '   seeds(2D,B1,B2,dense)')
for s in SIGS:
    print(f' {s:5.3f} {W[s]:8.4f}  {EDM[s]:8.4f} {EDMSE[s]:.4f} '
          f'{DN[s]:12.4f} {P2[s]:11.4f} {B1[s]:11.4f} {B2[s]:11.4f}       {NS[s]}')
print('\n excess over the HELD-OUT Wiener (negative = beats linear on the same 10k images):')
for s in SIGS:
    print(f' {s:5.3f}  EDM {EDM[s]-W[s]:+8.4f}   dense {DN[s]-W[s]:+8.4f}   '
          f'2D {P2[s]-W[s]:+8.4f}   B=1 {B1[s]-W[s]:+8.4f}   B=2 {B2[s]-W[s]:+8.4f}')

# *** k/d=8 IS DENSE AT ITS WORST WIDTH AT LOW sigma. ***  Held out, dense is NON-MONOTONE in
# k (in sample it is monotone -- that turnaround IS the overfitting), with a minimum at
# k/d=4 at sigma=0.127.  Quoting only k/d=8 there makes the comparison look rigged, so the
# best-over-width column is printed alongside it and belongs in any caption.
KDS = ['0.5', '1.0', '2.0', '3.0', '4.0', '6.0', '8.0']
print('\n dense held out across widths (test), and where its minimum sits:')
for s in SIGS:
    row = {k: np.atleast_2d(get(D, '{}|' + k + '|dense', s))[:, 2].mean() for k in KDS}
    kb = min(row, key=row.get)
    print(f' {s:5.3f}  ' + '  '.join(f'{k}:{row[k]:8.4f}' for k in KDS)
          + f'   | best k/d={kb} ({row[kb]:.4f}, {row[kb]-W[s]:+.4f} vs Wiener)')


def crossing(d):
    """sigma at which a curve crosses the held-out Wiener, log-interpolated on the grid."""
    x = [(s, d[s] - W[s]) for s in SIGS]
    for (s0, e0), (s1, e1) in zip(x, x[1:]):
        if e0 < 0 <= e1:
            return float(np.exp(np.log(s0) + (np.log(s1) - np.log(s0)) * (-e0) / (e1 - e0)))
    return None


fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.2))

for lab, d, col, mk, ls, lw in series:
    y = [d[s] for s in SIGS]
    ax[0].plot(sig, y, ls, color=col, marker=mk, ms=5, lw=lw, label=lab)
    ax[1].plot(sig, [d[s] - W[s] for s in SIGS], ls, color=col, marker=mk, ms=5, lw=lw,
               label=lab)

ax[0].set_xscale('log')
ax[0].set_xlabel(r'pixel noise $\sigma$')
ax[0].set_ylabel(r'held-out loss  $\mathbb{E}\,\|x_0-\hat{x}_0\|^2$')
ax[0].set_title('raw held-out loss vs noise\n(all fitted on 10k train, scored on the 10k CIFAR test set)',
                fontsize=10)
ax[0].legend(fontsize=8.5, loc='upper left')
ax[0].grid(alpha=0.3)

ax[1].axhline(0, color='#444444', lw=2)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'pixel noise $\sigma$')
ax[1].set_ylabel('loss $-$ held-out linear')
ax[1].set_title('excess over the held-out linear denoiser\n(same test set both sides, so the '
                'train/test trace offset cancels)', fontsize=10)
ax[1].grid(alpha=0.3)
ax[1].axhspan(-11, 0, color='#2ca02c', alpha=0.05, zorder=0)
ax[1].set_ylim(-11, 8.6)
# Crossing markers, staggered in y so the three labels (0.503 / 0.523 / 0.571) do not
# collide -- they are within 14% of each other in sigma.
for i, (lab, d, col, mk, ls, lw) in enumerate(series[1:]):
    xc = crossing(d)
    if xc is None:
        continue
    ax[1].axvline(xc, color=col, ls=':', lw=1.1, alpha=0.8, ymax=0.62)
    short = {'plain 2-D RF  c=3072': 'plain 2-D c=3072',
             'band RF  c=512  B=1': 'band c=512 B=1',
             'band RF  c=512  B=2': 'band c=512 B=2'}.get(lab, lab)
    ax[1].annotate(f'{short} crosses at $\\sigma$={xc:.3f}',
                   (xc, 7.4 - 0.95 * i), xytext=(8, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=7.5, color=col, zorder=6,
                   bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))
ax[1].annotate('RF beats held-out linear', (0.128, -10.4), fontsize=8.5, color='#2ca02c')
ax[1].legend(fontsize=8.5, loc='lower right')

fig.suptitle('Held-out denoising loss vs noise level, CIFAR-10 raw pixels '
             '(d=3072, train[:10000] -> 10,000 test images)', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs('figures', exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f'\nwrote {OUT}')

print('\n crossings of the held-out Wiener (log-interpolated):')
for lab, d, *_ in series[1:]:
    xc = crossing(d)
    print(f'   {lab:32s} {"sigma = %.3f" % xc if xc else "never crosses on this grid"}')
print('\n fraction of the linear->EDM gap closed (positive = the RF captures part of it):')
for s in SIGS:
    g = W[s] - EDM[s]
    print(f' {s:5.3f}  gap {g:6.3f}   dense {100*(W[s]-DN[s])/g:+7.1f}%   '
          f'2D {100*(W[s]-P2[s])/g:+7.1f}%   '
          f'B=1 {100*(W[s]-B1[s])/g:+7.1f}%   B=2 {100*(W[s]-B2[s])/g:+7.1f}%')

# Where dense k/d=8 overtakes the best structured arm.  Below this sigma the band is the
# best non-EDM curve on the plot; above it dense is, and it never stops winning.
print('\n dense k/d=8 (75,497,472 params) vs band B=2 c=512 (39,321,600):')
for s in SIGS:
    print(f' {s:5.3f}  B=2 - dense {B2[s]-DN[s]:+8.4f}   '
          f'{"band wins" if B2[s] < DN[s] else "DENSE wins"}')
x = [(s, B2[s] - DN[s]) for s in SIGS]
for (s0, e0), (s1, e1) in zip(x, x[1:]):
    if e0 < 0 <= e1:
        xc = float(np.exp(np.log(s0) + (np.log(s1) - np.log(s0)) * (-e0) / (e1 - e0)))
        print(f' => dense k/d=8 overtakes band B=2 at sigma ~ {xc:.3f}')
print('\n own generalisation gaps (test - train_resid) -- dense is the most overfit RF here:')
for s in SIGS:
    dn = np.atleast_2d(get(D, '{}|8.0|dense', s))
    b2 = np.atleast_2d(get(B, '{}|512|2', s))
    p = np.atleast_2d(get(H, '{}|3072|2d', s))
    wtr = get(H, 'linear|{}', s)
    print(f' {s:5.3f}  dense k/d=8 {dn[:,2].mean()-dn[:,1].mean():+8.4f}   '
          f'B=2 {b2[:,2].mean()-b2[:,1].mean():+8.4f}   '
          f'2D c=3072 {p[:,2].mean()-p[:,1].mean():+8.4f}   '
          f'Wiener {wtr[1]-wtr[0]:+8.4f}')
