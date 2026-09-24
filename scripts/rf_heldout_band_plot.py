"""LOSS vs NOISE, HELD OUT: linear / EDM / plain 2-D c=3072 / band c=512 B=1, B=2 and B=3.

    michimin, 2026-09-23 21:06: "graph loss against noise for held out linear, test edm,
    c=3072 plain 2-d, c=512 B=1 and c=512 B=2"

    michimin, 2026-09-23 22:04: "don't add the held out bayes oracle just add the old
    bayes_uncond we already had"  =>  the Bayes curve on the left panel is `bayes_uncond`
    from tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz.  It is the ONLY curve
    here that is in-sample (its atoms are its targets); see the note by BAYES below.

    michimin, 2026-09-24 01:58: "add B=3 curve to the loss vs noise graph"
    *** ⚠ B=3 IS A SHORT CURVE AND MUST BE DRAWN AS ONE.  Job 48071356 was only ever asked
    for four sigma (0.127 / 0.452 / 1.610 / 2.212) against every other curve's eight, so it
    is plotted on whatever subset of SIGS is actually IN THE TABLE -- never interpolated onto
    the full grid, never extended past its last measured point. ***  Same rule the
    `bayes_uncond` curve gets for its off-grid sigma=5.  B3SIGS is rebuilt from the npz on
    every run, so re-running this script after a cell lands extends the curve automatically.

EVERY OTHER CURVE IS A TEST-COLUMN NUMBER ON THE IDENTICAL 10,000 CIFAR TEST IMAGES, so the
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
O = load('tables/bayes_oracle_heldout.npz')      # empirical-prior Bayes, `oracle|{sigma}` =
                                                 # [test, se, in_sample, N_eff_te, N_eff_tr].
                                                 # PRINTED ONLY -- see the note by BAYES below.
M = load('tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz')

sig = np.array(SIGS)
W, EDM, EDMSE, P2, B1, B2, B3, DN, BO, BO50 = ({} for _ in range(10))
NS = {}
for s in SIGS:
    w = get(H, 'linear|{}', s)
    # The SAME Wiener row is stored in all five tables (linear_split has no RNG).  Verify
    # rather than assume -- a mismatch would mean the splits moved underneath one of them.
    for tag, z in (('band', B), ('edm', E), ('dense', D), ('oracle', O)):
        o = get(z, 'linear|{}', s)
        assert o is None or abs(o[1] - w[1]) < 1e-9, f'{tag} linear|{s} disagrees: {o} vs {w}'
    e = get(E, 'uncond-ve|{}', s)
    p = np.atleast_2d(get(H, '{}|3072|2d', s))
    b1 = np.atleast_2d(get(B, '{}|512|1', s))
    b2 = np.atleast_2d(get(B, '{}|512|2', s))
    # *** B=3 IS ABSENT AT FOUR OF THE EIGHT SIGMA AND THAT IS NOT AN ERROR -- see the
    # header.  `get` returns None on a miss, so the key simply never enters B3 and the curve
    # is drawn over B3SIGS instead of SIGS. ***
    b3 = get(B, '{}|512|3', s)
    dn = np.atleast_2d(get(D, '{}|8.0|dense', s))
    bo = get(O, 'oracle|{}', s)
    BO50[s] = get(O, 'oracle50|{}', s)[0]
    W[s], EDM[s], EDMSE[s] = w[1], e[2], e[3]
    P2[s], B1[s], B2[s] = p[:, 2].mean(), b1[:, 2].mean(), b2[:, 2].mean()
    if b3 is not None:
        b3 = np.atleast_2d(b3)
        B3[s] = b3[:, 2].mean()
    DN[s] = dn[:, 2].mean()
    BO[s] = bo[0]
    NS[s] = (len(p), len(b1), len(b2), len(dn), len(b3) if b3 is not None else 0)

B3SIGS = [s for s in SIGS if s in B3]

NNSQ = float(O['nn_sq_test_to_train'][0])     # mean squared nearest-neighbour distance,
NNSQ50 = float(O['nn_sq_test_to_train50'][0])  # test -> train = the sigma->0 limit of BO

# Last element is the curve's OWN sigma grid.  Everything but B=3 is measured at all eight.
series = [
    ('linear (Wiener), held out', W,   '#444444', 'o', '-',  2.0, SIGS),
    ('EDM U-net (trained on 50k)', EDM, '#d62728', 'D', '--', 2.0, SIGS),
    ('dense RF  k/d=8',           DN,  '#ff7f0e', 'P', '-.', 1.8, SIGS),
    ('plain 2-D RF  c=3072',      P2,  '#1f77b4', 's', '-',  1.8, SIGS),
    ('band RF  c=512  B=1',       B1,  '#2ca02c', '^', '-',  1.8, SIGS),
    ('band RF  c=512  B=2',       B2,  '#9467bd', 'v', '-',  1.8, SIGS),
    (f'band RF  c=512  B=3  ({len(B3SIGS)} $\\sigma$, line broken at gaps)',
                                  B3,  '#17becf', '*', '-',  1.8, B3SIGS),
]


def segments(ss):
    """Split a curve's sigma grid into maximal runs of points ADJACENT on the full grid.

    *** A LINE SEGMENT IS A CLAIM ABOUT WHAT HAPPENS BETWEEN ITS ENDPOINTS, AND B=3 HAS NO
    RIGHT TO MAKE ONE ACROSS FOUR SIGMA IT WAS NEVER RUN AT. ***  Joining 0.127 straight to
    1.61 draws a confident chord over 0.452 / 0.621 / 0.853 / 1.172 -- exactly the octave
    where every other curve has a measured point, where the linear->EDM gap peaks, and where
    on the left panel that chord rides ABOVE every other model and reads as "B=3 is terrible
    in the middle".  It is an artefact of two endpoints and a straight line.
    So the curve is BROKEN at its holes: markers at every measured sigma, line segments only
    between neighbours on SIGS.  Same convention as printing an absent cell as `----` instead
    of dropping the column -- a hole has to be VISIBLE, not smoothed over.
    Everything else passes SIGS itself and comes back as one unbroken run.
    """
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

# *** WHICH BAYES CURVE GOES ON THE PLOT -- michimin, 2026-09-23 22:04: "don't add the held
# out bayes oracle just add the old bayes_uncond we already had". ***  So the curve drawn is
# `bayes_uncond` from tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz, the same one
# in figures/dnn_feature_mmse_*.png.  The held-out oracle stays in the script and the table
# and is PRINTED below, but is not drawn.
#
# UNITS CHECK (done before drawing, not assumed): that table's `linear_uncond` at sigma=0.127
# is 7.9228 against our in-sample Wiener 7.8997 => raw pixels, d=3072, N=10^4, so it is on the
# same axis as everything else here.  ⚠ ONE ASYMMETRY TO KEEP IN MIND: it is an IN-SAMPLE
# quantity (atoms = targets) while every other curve on this figure is a test-column number.
#
# SIGMA GRID: the sigma30 file's grid contains 7 of our 8 sigma to <=0.08% (0.127 / 0.452 /
# 0.621 / 0.853 / 1.172 / 1.610 / 2.212) but NOT 5.0 -- the nearest stored point is 5.7362,
# 14.7% away.  So it is drawn on ITS OWN grid clipped to this figure's x-range rather than
# interpolated onto ours; that is why it stops at sigma=4.175.  (The sigma40 companion file
# misses every one of our sigma by 3-11%; do not use it.)
BSIG, BUNC = (np.asarray(M['sigma'], float), np.asarray(M['bayes_uncond'], float)) if M \
    else (np.array([]), np.array([]))
_bm = (BSIG >= sig.min() * 0.999) & (BSIG <= sig.max() * 1.001)
BAYES = ('Bayes under the empirical prior (`bayes_uncond`)', '#8c564b', 'X', (0, (4, 2)), 2.0)

# *** ABSENT B=3 CELLS RENDER AS AN EXPLICIT `----`, NEVER AS A DROPPED COLUMN. ***  Same
# convention as rf_heldout_report.py section B: a hole in a grid has to be VISIBLE in the
# output, because a table that silently omits the sigma it lacks reads as complete.
def b3(s, w=11):
    return f'{B3[s]:{w}.4f}' if s in B3 else '-' * (w - 4) + '----'


print(' sigma   W_test    EDM_te  (se)  dense k/d=8   2D c=3072   B=1 c=512   B=2 c=512'
      '   B=3 c=512   seeds(2D,B1,B2,dense,B3)')
for s in SIGS:
    print(f' {s:5.3f} {W[s]:8.4f}  {EDM[s]:8.4f} {EDMSE[s]:.4f} '
          f'{DN[s]:12.4f} {P2[s]:11.4f} {B1[s]:11.4f} {B2[s]:11.4f} {b3(s)}       {NS[s]}')
_b3list = ', '.join('%g' % s for s in B3SIGS)
print(f'  => B=3 covers {len(B3SIGS)} of the {len(SIGS)} sigma on this plot ({_b3list}); '
      f'job 48071356 was asked for four (0.127/0.452/1.61/2.212) and the other four were '
      f'never requested.  The curve is drawn on that subgrid, NOT interpolated.')
print('\n excess over the HELD-OUT Wiener (negative = beats linear on the same 10k images):')
for s in SIGS:
    e3 = f'{B3[s]-W[s]:+8.4f}' if s in B3 else '    ----'
    print(f' {s:5.3f}  EDM {EDM[s]-W[s]:+8.4f}   dense {DN[s]-W[s]:+8.4f}   '
          f'2D {P2[s]-W[s]:+8.4f}   B=1 {B1[s]-W[s]:+8.4f}   B=2 {B2[s]-W[s]:+8.4f}'
          f'   B=3 {e3}   | oracle {BO[s]-W[s]:+9.4f}')

# *** WHY ONLY THE HELD-OUT ORACLE IS PLOTTABLE. ***  The in-sample column is the retracted
# `bayes_uncond` curve of figures/dnn_feature_mmse_*.png: its atoms ARE its targets, so at
# small sigma the posterior is a point mass on the image that generated y and the loss is
# literally 0.  N_eff = exp(H(w)) is 1.00 in BOTH columns at low sigma -- the difference is
# that in sample that single atom is the right answer and held out it is the wrong one.
print('\n the empirical-prior Bayes oracle, and why the in-sample version is not on the plot:')
print(f'  sigma   held out   vs Wiener |   IN SAMPLE   N_eff(test)  N_eff(train)')
for s in SIGS:
    bo = get(O, 'oracle|{}', s)
    print(f'  {s:5.3f} {bo[0]:10.4f} {bo[0]-W[s]:+10.4f} | {bo[2]:11.4f} {bo[3]:12.2f} '
          f'{bo[4]:13.2f}')
print(f'  => the held-out curve flattens onto the mean squared NN distance {NNSQ:.4f} as '
      f'sigma->0 (sigma=0.127 gives {BO[SIGS[0]]:.4f}, agreeing to {abs(BO[SIGS[0]]-NNSQ):.4f})')

# THE CURVE THAT IS ACTUALLY DRAWN, on its own sigma grid (see the note by BAYES above).
print('\n `bayes_uncond` as drawn (dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz):')
for a, b in zip(BSIG[_bm], BUNC[_bm]):
    near = min(SIGS, key=lambda s: abs(np.log(s / a)))
    flag = 'ours' if abs(near / a - 1) < 1e-3 else f'({100*abs(near/a-1):.1f}% off {near:g})'
    print(f'  {a:8.4f} {b:10.4f}   {flag}')
print(f'  => grid stops at {BSIG[_bm][-1]:.4f}; the nearest stored point to sigma=5.0 is '
      f'{BSIG[BSIG > 5.0][0]:.4f} ({100*(BSIG[BSIG > 5.0][0]/5.0 - 1):.1f}% off), NOT snapped')

# *** EDM vs ITS OWN MATCHED-PRIOR ORACLE.  This is the strongest single statement the oracle
# curve licenses, and it runs the OPPOSITE way to the worry it was built to test. ***  EDM saw
# all 50k, so the 50k-atom oracle is exactly "what a model that reproduced its training prior
# perfectly would score on these test images".  EDM beats it by more than an order of
# magnitude at low sigma => whatever EDM is doing, it is not reproducing its empirical prior.
print('\n EDM vs the 50k-atom oracle matched to its own training set:')
print('  sigma   EDM_test   oracle50    ratio   | oracle10k   Wiener')
for s in SIGS:
    print(f'  {s:5.3f} {EDM[s]:10.4f} {BO50[s]:10.4f} {BO50[s]/EDM[s]:8.2f}x | '
          f'{BO[s]:10.4f} {W[s]:8.4f}')

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


def crossing(d, ss=None):
    """sigma at which a curve crosses the held-out Wiener, log-interpolated on the grid.

    ⚠ ONLY AS GOOD AS THE BRACKET IT IS INTERPOLATED ACROSS.  For the eight-sigma curves the
    two straddling points are ADJACENT grid points (0.452 and 0.621, 14% apart), so the
    crossing is pinned.  B=3 skips 0.621 / 0.853 / 1.172 entirely, so its bracket is 3.6x
    wide and the interpolated value is a bracket, not a location -- which is why it is
    printed with its endpoints and NOT drawn as a marker on the figure.
    """
    x = [(s, d[s] - W[s]) for s in (ss or SIGS)]
    for (s0, e0), (s1, e1) in zip(x, x[1:]):
        if e0 < 0 <= e1:
            return float(np.exp(np.log(s0) + (np.log(s1) - np.log(s0)) * (-e0) / (e1 - e0)))
    return None


fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.2))

lab, col, mk, ls, lw = BAYES
ax[0].plot(BSIG[_bm], BUNC[_bm], color=col, ls=ls, marker=mk, ms=5, lw=lw, label=lab)
# One neutral label, not an argument: the number is measured (posterior_neff() in
# scripts/rf_kstar_vs_sigma.py) and is what makes the low-sigma end of this curve read as 0.
# Placed in the empty top-left quadrant (every curve is below 30 out to sigma=0.6) rather than
# next to the curve itself, which runs along y=0 straight through the legend.
ax[0].annotate('posterior over the $10^4$ training atoms, evaluated on\n'
               'those same atoms; $N_{\\mathrm{eff}}=1.00$ at every $\\sigma\\leq1.61$.\n'
               'Drawn on its own $\\sigma$ grid (7 of our 8 exact, none at $\\sigma$=5).',
               (0.025, 0.63), xycoords='axes fraction',
               fontsize=7.2, color=col, va='top')

for lab, d, col, mk, ls, lw, ss in series:
    # ss is the curve's OWN grid: B=3 stops at its last measured sigma instead of being
    # stretched across the four it was never run at.
    ms = 8 if mk == '*' else 5
    # NO label on the left panel -- the model curves are identical in colour and marker on
    # both panels, so one legend (on the right, where there is empty space below the zero line)
    # serves both.  A second copy here had to sit on top of the sigma=0.5-1.2 octave, which is
    # exactly the part of the left panel worth looking at.
    # One plot() call per unbroken run; a run of length 1 still renders its marker.  The
    # label goes on the first run only, so a broken curve gets ONE legend entry.
    for k, seg in enumerate(segments(ss)):
        ax[0].plot(seg, [d[s] for s in seg], ls, color=col, marker=mk, ms=ms, lw=lw)
        ax[1].plot(seg, [d[s] - W[s] for s in seg], ls, color=col, marker=mk, ms=ms, lw=lw,
                   label=lab if k == 0 else None)

ax[0].set_xscale('log')
ax[0].set_xlabel(r'pixel noise $\sigma$')
ax[0].set_ylabel(r'held-out loss  $\mathbb{E}\,\|x_0-\hat{x}_0\|^2$')
ax[0].set_title('raw held-out loss vs noise\n(all fitted on 10k train, scored on the 10k CIFAR test set)',
                fontsize=10)
ax[0].legend(fontsize=8.0, loc='upper left',
             title='the six model curves are labelled on the right panel',
             title_fontsize=7.5)
ax[0].grid(alpha=0.3)

ax[1].axhline(0, color='#444444', lw=2)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'pixel noise $\sigma$')
ax[1].set_ylabel('loss $-$ held-out linear')
ax[1].set_title('excess over the held-out linear denoiser\n(same test set both sides, so the '
                'train/test trace offset cancels)', fontsize=10)
ax[1].grid(alpha=0.3)
ax[1].axhspan(-11, 0, color='#2ca02c', alpha=0.05, zorder=0)
# Headroom above the highest curve (+7.27) is deliberate: the three crossing labels sit in it
# so they clear both the curves and the off-scale oracle note below them.
YLO, YHI = -11.0, 11.4
ax[1].set_ylim(YLO, YHI)
# Crossing markers, staggered in y so the three labels (0.503 / 0.523 / 0.571) do not
# collide -- they are within 14% of each other in sigma.
for i, (lab, d, col, mk, ls, lw, ss) in enumerate(series[1:]):
    # *** NO CROSSING MARKER FOR B=3, DELIBERATELY. ***  Every other curve's crossing is
    # bracketed by ADJACENT grid points 14% apart; B=3's bracket skips 0.621 / 0.853 / 1.172
    # and is 3.6x wide, so a vertical line would claim a precision the grid does not have.
    # The number is still PRINTED below, with its endpoints.
    if d is B3:
        continue
    xc = crossing(d, ss)
    if xc is None:
        continue
    ylab = 10.6 - 0.95 * i
    ax[1].axvline(xc, color=col, ls=':', lw=1.1, alpha=0.8,
                  ymax=(ylab - YLO) / (YHI - YLO))
    short = {'plain 2-D RF  c=3072': 'plain 2-D c=3072',
             'band RF  c=512  B=1': 'band c=512 B=1',
             'band RF  c=512  B=2': 'band c=512 B=2'}.get(lab, lab)
    ax[1].annotate(f'{short} crosses at $\\sigma$={xc:.3f}',
                   (xc, ylab), xytext=(8, 0), textcoords='offset points',
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
for lab, d, col, mk, ls, lw, ss in series[1:]:
    xc = crossing(d, ss)
    note = ''
    if xc is not None and d is B3:
        lo = max([s for s in ss if s < xc])
        hi = min([s for s in ss if s >= xc])
        skipped = ', '.join('%g' % s for s in SIGS if lo < s < hi)
        note = (f'   ⚠ BRACKET {lo:g}-{hi:g} ({hi/lo:.1f}x wide, skipping '
                f'{skipped}) -- not drawn on the figure')
    print(f'   {lab:38s} '
          f'{"sigma = %.3f" % xc if xc else "never crosses on this grid"}{note}')
print('\n fraction of the linear->EDM gap closed (positive = the RF captures part of it):')
for s in SIGS:
    g = W[s] - EDM[s]
    print(f' {s:5.3f}  gap {g:6.3f}   dense {100*(W[s]-DN[s])/g:+7.1f}%   '
          f'2D {100*(W[s]-P2[s])/g:+7.1f}%   '
          f'B=1 {100*(W[s]-B1[s])/g:+7.1f}%   B=2 {100*(W[s]-B2[s])/g:+7.1f}%   '
          f'B=3 {f"{100*(W[s]-B3[s])/g:+7.1f}%" if s in B3 else "   ----"}')

# Where dense k/d=8 overtakes the best structured arm.  Below this sigma the band is the
# best non-EDM curve on the plot; above it dense is, and it never stops winning.
print('\n dense k/d=8 (75,497,472 params) vs band B=2 c=512 (39,321,600) and B=3 (77,070,336):')
for s in SIGS:
    # B=3 is the closest thing on the plot to a parameter-matched rival for dense k/d=8
    # (77.07M vs 75.50M, within 2%), so this column is the fairest dense-vs-band comparison
    # the grid contains -- and dense still wins it at every sigma where B=3 exists above 0.127.
    d3 = (f'   |  B=3 - dense {B3[s]-DN[s]:+8.4f}  '
          f'{"band wins" if B3[s] < DN[s] else "DENSE wins"}') if s in B3 else ''
    print(f' {s:5.3f}  B=2 - dense {B2[s]-DN[s]:+8.4f}   '
          f'{"band wins" if B2[s] < DN[s] else "DENSE wins"}{d3}')
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
    g3 = get(B, '{}|512|3', s)
    g3 = (f'{np.atleast_2d(g3)[:,2].mean()-np.atleast_2d(g3)[:,1].mean():+8.4f}'
          if g3 is not None else '    ----')
    # ⚠ AT sigma >= 1.61 EVERY own gap is NEGATIVE -- that is the train/test TRACE OFFSET
    # (Tr(Sigma_test) = 189.794 vs Tr(Sigma_train) = 191.522, a 0.9% deficit that passes
    # ~1:1 into the loss at large sigma), NOT better generalisation.  Read gaps RELATIVE to
    # the common offset of about -0.15, never as absolute numbers.
    print(f' {s:5.3f}  dense k/d=8 {dn[:,2].mean()-dn[:,1].mean():+8.4f}   '
          f'B=2 {b2[:,2].mean()-b2[:,1].mean():+8.4f}   B=3 {g3}   '
          f'2D c=3072 {p[:,2].mean()-p[:,1].mean():+8.4f}   '
          f'Wiener {wtr[1]-wtr[0]:+8.4f}')
