"""Held-out dense conditional RF at k/d=8 (MODE=vu).

The model is relu(Theta y + Gamma U), followed by a dense readout plus a free
class-specific intercept VU.  Equivalently, train and test moments are centred by
the TRAIN mean of their class.  Noise is integrated analytically.  One row per
seed stores [train, train_resid, test], and the NPZ is rewritten after every cell
so interrupted runs resume at the next seed.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.rf_pixel_heldout import load, DEV, DT, d
from scripts.rf_pixel_dense_heldout import split_losses
from scripts.rf_cond_toll2d_heldout import load_labels, conditional_families, NCLS

LAM = float(os.environ.get('LAM', '1e-6'))
J = float(os.environ.get('J', '8'))
SIGS = [float(x) for x in os.environ.get(
    'SIGS', '0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
NSEED = int(os.environ.get('NSEED', '1'))
GSCALE = float(os.environ.get('GSCALE', '1'))
OUT = os.environ.get('OUT', 'tables/rf_pixel_dense_cond_heldout_vu_j8.npz')


def key(sg):
    return f'{sg}|{J:g}|dense|vu|g{GSCALE:g}'


def main():
    Xtr = load(True, NIMG).reshape(NIMG, d)
    Xte = load(False, NTEST).reshape(NTEST, d)
    ytr = load_labels(True, NIMG, Xtr)
    yte = load_labels(False, NTEST, Xte)
    k = int(round(J * d))
    store = ({q: v for q, v in np.load(OUT, allow_pickle=True).items()}
             if os.path.exists(OUT) else {})
    print(f'DENSE CONDITIONAL RF MODE=vu d={d} k={k} k/d={J:g} '
          f'train={NIMG} test={NTEST} seeds={NSEED} -> {OUT}', flush=True)

    # W_S is the matching linear family: shared slope plus free class intercept.
    bas = conditional_families(Xtr, ytr, Xte, yte, NCLS, SIGS, [], verbose=False)
    for sg in SIGS:
        lk = f'linS|{sg}'
        lv = np.asarray(bas['S'][('W', sg)], float)
        if lk in store:
            assert np.max(np.abs(np.asarray(store[lk]) - lv)) < 1e-9
        store[lk] = lv
        q = key(sg)
        old = np.atleast_2d(store[q]) if q in store else np.empty((0, 3))
        if len(old) >= NSEED:
            print(f'  sigma={sg}: already done, {len(old)} seed(s), test {old[:,2].mean():.4f}',
                  flush=True)
            continue
        rows = old.tolist()
        for s in range(len(rows), NSEED):
            # Theta is paired with the unconditional dense driver.  Gamma has its own RNG,
            # hence adding conditioning never changes Theta.
            Th = torch.as_tensor(
                np.random.default_rng(3 + s).standard_normal((k, d)) / np.sqrt(d),
                device=DEV, dtype=DT)
            gg = torch.Generator(device=DEV).manual_seed(7700 + 11 * s + k)
            Ga = torch.randn(k, NCLS, generator=gg, device=DEV, dtype=DT)
            Ga *= GSCALE / np.sqrt(NCLS)
            t0 = time.time()
            r = split_losses(Xtr, Xte, Th, sg, lam=LAM, lab=ytr, gam=Ga,
                             lab_test=yte, class_centre=True)
            rows.append(list(r))
            print(f'  sigma={sg} seed={s}: train {r[0]:.4f} resid {r[1]:.4f} '
                  f'test {r[2]:.4f} vs W_S {r[2]-lv[1]:+.4f} '
                  f'[{time.time()-t0:.0f}s, peak '
                  f'{torch.cuda.max_memory_allocated()/2**30:.1f} GB]', flush=True)
            del Th, Ga
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            store[q] = np.asarray(rows)
            np.savez(OUT, **store)
    print('done', flush=True)


if __name__ == '__main__':
    main()
