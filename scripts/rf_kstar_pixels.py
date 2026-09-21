"""k_* (dense-RF-over-linear threshold) on noised raw CIFAR-10 pixels, on the SIGMA GRID
of the matched-free-parameter experiment, compared against the crossing we MEASURED.

Why this script exists alongside rf_kstar_vs_sigma.py
-----------------------------------------------------
That script already works on raw pixels (d=3072, T.ToTensor(), uncentred [0,1]) and is the
authority for the defect machinery -- this one imports it rather than reimplementing.  What
it does NOT do is evaluate on the sigma grid of the featmatch experiments, or compare the
threshold to the width at which dense actually overtook the linear denoiser, which we now
have from tables/rf_pixel_dense_sweep.npz.  That comparison is the point here.

The threshold, and what tau means
---------------------------------
Theorem thm:prob in the constant-complete (Cov(r), shrinkage-free) form:

    L^lin_sigma - L^RF_sigma <= tau * d        for every   k <= tau * d / (4 check_eps_w^2)

so tau is a PER-COORDINATE gap, and the quoted k(gap<=tau) is linear in tau.  The writeup's
k_* = d / hat_eps_w^2 is the same statement with the Xi = 1/gamma conversion dropped; it is
reported as a reference, not as the operational number (see rem:xi-sigma).

Reading it against the experiment: our dense sweep attains an ABSOLUTE gap
L^lin - L^dense = g at width k, i.e. a per-coordinate tau = g/d.  The theorem then forbids
that gap for any k <= tau*d/(4 check_eps_w^2) = g/(4 check_eps_w^2).  So

    k_min(g) := g / (4 * check_eps_w^2)

is the smallest width at which a gap of g is permitted, and the test is whether the k at
which we measured g exceeds it.

Sample size
-----------
The defect is a functional of the distribution.  We report it on N=10,000 -- the SAME
empirical measure the featmatch losses are computed on, so the comparison is like-for-like
-- and on N=50,000 as the closer-to-population value.  If they disagree materially, the
threshold is being read off a sample-dependent object and that has to be said out loud.

    python scripts/rf_kstar_pixels.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import scripts.rf_kstar_vs_sigma as K

SIGS = [0.127, 0.452, 1.61, 2.459, 3.756, 5.0]
N_THETA = int(os.environ.get('N_THETA', '2048'))
NIMGS = [int(x) for x in os.environ.get('NIMGS', '10000,50000').split(',')]
OUT = 'tables/rf_kstar_pixels.npz'


def measured():
    """(sigma -> {k: L_dense}) and the linear baseline, from the featmatch tables."""
    D, LIN = {}, {}
    for f in ('tables/rf_pixel_featmatch2.npz', 'tables/rf_pixel_dense_sweep.npz'):
        for key, v in np.load(f, allow_pickle=True).items():
            if key.startswith('linear'):
                LIN[float(key.split('|')[1])] = float(v); continue
            sg, j, w = key.split('|')
            if w == 'dense':
                D.setdefault(float(sg), {})[int(round(float(j) * 3072))] = float(np.mean(v))
    return D, LIN


def main():
    D, LIN = measured()
    store = {}
    for nimg in NIMGS:
        K.NIMG = nimg
        X = K.load('CIFAR-10')[:nimg]
        d = X.shape[1]
        g = torch.Generator(device=K.DEV); g.manual_seed(0)
        Theta = torch.randn(N_THETA, d, device=K.DEV, dtype=K.DT, generator=g) / np.sqrt(d)
        hat, chk, gam = K.defect_curves(X, Theta, SIGS)
        store[f'{nimg}|hat'] = hat; store[f'{nimg}|chk'] = chk; store[f'{nimg}|gam'] = gam

        print(f"\n{'='*104}\nCIFAR-10 raw pixels  d={d}  N={nimg}  n_theta={N_THETA}"
              f"\n{'='*104}")
        print(f"{'sigma':>7} {'gamma':>10} {'chk_eps_w^2':>12} {'hat_eps_w^2':>12} "
              f"{'d/hat^2':>10} {'k(tau=1e-3)':>12} {'k(tau=1e-4)':>12}")
        for i, s in enumerate(SIGS):
            print(f"{s:>7} {gam[i]:>10.3e} {chk[i]:>12.3e} {hat[i]:>12.3e} "
                  f"{d/hat[i]:>10.3e} {1e-3*d/(4*chk[i]):>12.3e} {1e-4*d/(4*chk[i]):>12.3e}")

        print(f"\n  MEASURED gap vs the width the theorem permits it at"
              f"   [k_min(g) = g / (4 chk_eps_w^2)]")
        print(f"  {'sigma':>7} {'linear':>8} {'k meas':>8} {'k/d':>5} {'L_dense':>9} "
              f"{'gap g':>8} {'tau=g/d':>10} {'k_min(g)':>11} {'k/k_min':>8}")
        for i, s in enumerate(SIGS):
            if s not in D:
                continue
            for k in sorted(D[s]):
                gp = LIN[s] - D[s][k]
                if gp <= 0:
                    continue
                kmin = gp / (4 * chk[i])
                print(f"  {s:>7} {LIN[s]:>8.3f} {k:>8,} {k/3072:>5.3g} {D[s][k]:>9.3f} "
                      f"{gp:>8.3f} {gp/d:>10.3e} {kmin:>11.3e} {k/kmin:>8.2f}"
                      f"{'   <-- VIOLATES' if k < kmin else ''}")
        del X, Theta; torch.cuda.empty_cache()
    np.savez(OUT, sigmas=np.array(SIGS), **store)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
