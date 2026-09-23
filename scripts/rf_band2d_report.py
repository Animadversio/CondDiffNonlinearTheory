"""BAND B=1 NONLINEAR RF vs PLAIN 2-D vs the best LINEAR map in each class, held out.

Regenerates the band comparison table from the npz files -- no number is hand-transcribed.
Run before editing any doc or posting any band table.

    C=512 python scripts/rf_band2d_report.py

DEFINITIONS
  gain(B) = L[best LINEAR map in the class] - L[the relu RF in the same class]
  retain  = gain(B=1) / gain(B=0)   -- "did the 9x larger linear class absorb what relu bought?"

TWO INDEXING TRAPS THIS FILE EXISTS TO AVOID.
  (1) tables/rf_band_relaxation_heldout*.npz store one ROW PER SIGMA under a single key
      `band2d|B`, with the sigma grid in a separate `sigmas` array -- and the two files have
      DIFFERENT grids ([0.127 0.452 1.61 5.0] and [2.212]).  Matching by row index across
      files silently compares different noise levels.  We build {float(sigma): row} per file
      and match on the VALUE.
  (2) The PAIRED differential band-minus-plain is only paired if the two sides use the SAME
      Theta draws.  The plain table has 2 seeds at some cells and the band table has 1, so we
      truncate BOTH to min(len) -- averaging all of plain against one band seed would mix a
      paired differential with an unpaired seed-mean difference.
"""
import numpy as np, os, sys

BAND = 'tables/rf_pixel_band2d_heldout.npz'
PLAIN = 'tables/rf_pixel_heldout.npz'
LINS = ['tables/rf_band_relaxation_heldout.npz',
        'tables/rf_band_relaxation_heldout_sig2212.npz',
        'tables/rf_band_relaxation_heldout_sigmid.npz']
C = int(os.environ.get('C', '512'))
SIGS = [float(x) for x in os.environ.get(
    'SIGS', '0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0').split(',')]
TOL = 1e-9


def spellings(s):
    return [f'{s:g}', str(s), repr(s), f'{s:.3f}', f'{s:.2f}', f'{s:.4f}']


def get(store, fmt, s):
    for k in spellings(s):
        if fmt.format(k) in store:
            return np.asarray(store[fmt.format(k)])
    return None


def col(a, c):
    if a is None:
        return float('nan')
    return float(np.mean(a[:, c])) if a.ndim == 2 else float(a.ravel()[c])


band = dict(np.load(BAND, allow_pickle=True)) if os.path.exists(BAND) else {}
plain = dict(np.load(PLAIN, allow_pickle=True)) if os.path.exists(PLAIN) else {}

# --- linear band classes: {B: {sigma_value: test_loss}} keyed on the PHYSICAL sigma ---
linmap = {}
for f in LINS:
    if not os.path.exists(f):
        continue
    z = dict(np.load(f, allow_pickle=True))
    sg = np.asarray(z['sigmas']).ravel()
    for key, v in z.items():
        if not key.startswith('band2d|'):
            continue
        B = int(key.split('|')[1])
        v = np.asarray(v)
        for i, s in enumerate(sg):
            linmap.setdefault(B, {})[float(s)] = float(v[i, 1])   # col 1 = test


def linlook(B, s):
    for k, val in linmap.get(B, {}).items():
        if abs(k - s) < 1e-6:
            return val
    return float('nan')


print(f'BAND B=1 NONLINEAR RF vs PLAIN 2-D, c={C}, HELD OUT, paired on identical Theta draws')
print(' sigma  W_test  lin B=0  lin B=1 | plain2D  band B=1  paired d | '
      'gain B=0 gain B=1 retain  seeds')
for s in SIGS:
    w = get(band, 'linear|{}', s)
    if w is None:
        w = get(plain, 'linear|{}', s)
    wte = col(w, 1) if w is not None else float('nan')

    b = get(band, '{}|' + f'{C}|1', s)
    p = get(plain, '{}|' + f'{C}|2d', s)
    L0, L1 = linlook(0, s), linlook(1, s)

    if b is None:
        print(f' {s:5.3f} {wte:7.4f} {L0:8.4f} {L1:8.4f} |'
              f'   -- band cell not present yet --')
        continue

    # PAIRED: truncate both sides to the seeds they share.
    if p is not None:
        n = min(len(np.atleast_2d(b)), len(np.atleast_2d(p)))
        bb, pp = np.atleast_2d(b)[:n], np.atleast_2d(p)[:n]
        pte = float(pp[:, 2].mean())
        dpair = float((bb[:, 2] - pp[:, 2]).mean())
    else:
        n, pte, dpair = 0, float('nan'), float('nan')
    bte = col(b, 2)
    g0, g1 = L0 - pte, L1 - bte
    ret = 100 * g1 / g0 if g0 == g0 and g0 != 0 else float('nan')
    print(f' {s:5.3f} {wte:7.4f} {L0:8.4f} {L1:8.4f} | {pte:8.4f} {bte:9.4f} '
          f'{dpair:+9.4f} | {g0:8.4f} {g1:8.4f} {ret:6.1f}%  {n}')

print()
print('own generalisation gaps (test - train_resid), band vs plain at this c:')
print(' sigma   band      plain    [NEGATIVE at high sigma = the train/test TRACE OFFSET,')
print('                             Tr(S_test)=189.794 vs Tr(S_train)=191.522, NOT a gap]')
for s in SIGS:
    b = get(band, '{}|' + f'{C}|1', s)
    p = get(plain, '{}|' + f'{C}|2d', s)
    if b is None:
        continue
    gb = col(b, 2) - col(b, 1)
    gp = (col(p, 2) - col(p, 1)) if p is not None else float('nan')
    print(f' {s:5.3f} {gb:+8.4f}  {gp:+8.4f}')

miss = [s for s in SIGS if linlook(1, s) != linlook(1, s)]
if miss:
    print(f'\n*** MISSING LINEAR BAND COLUMN at sigma = {miss} -- gain/retain cannot be '
          f'computed there.  Fill with:\n    SIGS={",".join(str(x) for x in miss)} '
          f'python scripts/rf_band_relaxation_heldout.py  (~1 min on GPU)')
