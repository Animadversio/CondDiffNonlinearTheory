"""Merge a side-table of band2d held-out cells into the canonical band table.

    python scripts/merge_band2d_tables.py            # dry run, prints the plan
    APPLY=1 python scripts/merge_band2d_tables.py    # actually writes MAIN

WHY THIS EXISTS.  scripts/rf_pixel_band2d_heldout.py reads its output npz ONCE at startup
and rewrites the WHOLE dict after each finished cell.  Two concurrent jobs on one file
therefore clobber each other -- the second to save writes its startup snapshot plus its own
cell, deleting whatever the other finished in between.  So concurrent jobs must use separate
OUT files and be merged here afterwards.

THE SAFETY PROPERTY.  Shared keys are not overwritten, they are VERIFIED.  Both jobs
recompute the `linear|{sigma}` Wiener rows for their own sigma, and the sigma grids of the
two jobs overlap in the table only where a previous run already stored a value -- so any
shared key is an independent recomputation of the same quantity and MUST agree.  A
disagreement means the two jobs did not run the same experiment and the merge is refused.
"""
import numpy as np, os, sys

MAIN = os.environ.get('MAIN', 'tables/rf_pixel_band2d_heldout.npz')
SIDE = os.environ.get('SIDE', 'tables/rf_pixel_band2d_heldout_B2rest.npz')
TOL = float(os.environ.get('TOL', '1e-10'))
APPLY = os.environ.get('APPLY', '') == '1'

for f in (MAIN, SIDE):
    if not os.path.exists(f):
        sys.exit(f'missing {f}')

main = {k: v for k, v in np.load(MAIN, allow_pickle=True).items()}
side = {k: v for k, v in np.load(SIDE, allow_pickle=True).items()}
print(f'MAIN {MAIN}: {len(main)} keys')
print(f'SIDE {SIDE}: {len(side)} keys')

shared = sorted(set(main) & set(side))
new = sorted(set(side) - set(main))

bad = []
for k in shared:
    a, b = np.asarray(main[k], dtype=float), np.asarray(side[k], dtype=float)
    if a.shape != b.shape:
        bad.append((k, f'shape {a.shape} vs {b.shape}'))
        continue
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not (d <= TOL):
        bad.append((k, f'max|delta| = {d:.3e}'))

print(f'\nshared keys: {len(shared)}  (independent recomputations -- must agree)')
for k in shared:
    a, b = np.asarray(main[k], dtype=float), np.asarray(side[k], dtype=float)
    if a.shape == b.shape and a.size:
        print(f'  {k:28s} max|delta| = {float(np.max(np.abs(a - b))):.3e}')

if bad:
    print('\n*** REFUSING TO MERGE -- shared keys disagree: ***')
    for k, why in bad:
        print(f'  {k}: {why}')
    sys.exit(1)

print(f'\nnew keys to add: {len(new)}')
for k in new:
    v = np.atleast_2d(np.asarray(side[k], dtype=float))
    tail = f'  test={v[:, 2].mean():.4f}' if v.shape[1] >= 3 else ''
    print(f'  {k}{tail}')

if not new:
    print('\nnothing to add.')
    sys.exit(0)

if not APPLY:
    print('\ndry run -- rerun with APPLY=1 to write.')
    sys.exit(0)

main.update({k: side[k] for k in new})
np.savez(MAIN, **main)
chk = {k: v for k, v in np.load(MAIN, allow_pickle=True).items()}
assert set(chk) == set(main), 'readback key mismatch'
for k in new:
    assert np.allclose(np.asarray(chk[k], dtype=float),
                       np.asarray(side[k], dtype=float)), f'readback mismatch {k}'
print(f'\nwrote {MAIN}: {len(chk)} keys ({len(new)} added), readback verified.')
