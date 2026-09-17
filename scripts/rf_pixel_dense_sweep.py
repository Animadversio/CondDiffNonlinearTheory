"""Dense-only extension of the raw-pixel feature-match sweep.

WHY DENSE-ONLY IS A LEGITIMATE THING TO PLOT HERE
-------------------------------------------------
The circulant side of the matched-free-parameter comparison has been measured flat: at
sigma=0.127 it is 8.0768 / 8.0501 / 8.0291 at k/d = 0.5 / 1 / 2, i.e. it moves 0.048 across
a 4x range in free parameters, against dense's 9.93. It is sitting on its class floor. So
the interesting remaining question is not "what does the circulant do at k/d=3" -- it does
8.02 -- but WHERE DENSE CROSSES THAT FLOOR, and dense is cheap: a k x k Stein build at
k=6144 takes 1 s, while the matched circulant at c=6144 takes 2561 s.

This sweeps dense alone to large k/d so the crossing can be read off against the circulant
floor. The circulant value at those k/d is an EXTRAPOLATION of a measured-flat curve, not a
computed point, and must be labelled as such wherever it is plotted.

Memory is the limit: Sigma_phi is dense k x k and the Stein path needs rho, rho^2, rho^3 and
three C_n Grams alongside it, ~10 arrays of k^2 fp64:
    k/d=3  k=9216    0.68 GB each   ~7 GB
    k/d=4  k=12288   1.21 GB each  ~12 GB
    k/d=6  k=18432   2.72 GB each  ~27 GB
    k/d=8  k=24576   4.83 GB each  ~48 GB   <- tried last, may OOM
Largest k/d is attempted last so a failure there does not cost the smaller points.

    python scripts/rf_pixel_dense_sweep.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t
from scripts.rf_pixel_featmatch2 import load, DEV, DT, d, LAM

JS = [float(x) for x in os.environ.get('JS', '3,4,6,8').split(',')]
SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610').split(',')]
NSEED = int(os.environ.get('NSEED', '2'))
OUT = os.environ.get('OUT', 'tables/rf_pixel_dense_sweep.npz')


def main():
    X = load(); N = X.shape[0]
    Xc = X - X.mean(0); ev = torch.linalg.eigvalsh((Xc.T @ Xc) / N)
    U = torch.zeros(N, 1, dtype=DT, device=DEV)
    print(f"DENSE SWEEP raw pixels d={d} N={N} Tr(Sigma)={float(ev.sum()):.3f} "
          f"seeds={NSEED}", flush=True)

    store = {}
    if os.path.exists(OUT):
        store = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}

    for sg in SIGS:
        lin = float((sg ** 2 * ev / (ev + sg ** 2)).sum())
        store[f'linear|{sg}'] = np.array(lin)
        print(f"\n=== sigma={sg}   linear={lin:.4f} ===", flush=True)
        for j in JS:
            key = f'{sg}|{j}|dense'
            if key in store:
                print(f"  j={j}: already done", flush=True)
                continue
            k = int(round(j * d))
            t0 = time.time(); dv = []
            try:
                for s in range(NSEED):
                    Th = np.random.default_rng(3 + s).standard_normal((k, d)) / np.sqrt(d)
                    G = torch.zeros(k, 1, dtype=DT, device=DEV)
                    dv.append(float(stein_finiteN_mmse_t(X, U, Th, G, sg, LAM,
                                                         conditional=False, device=DEV,
                                                         dtype=DT)))
                    del G, Th; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  j={j} k={k}: OOM, stopping this sigma", flush=True)
                torch.cuda.empty_cache()
                break
            store[key] = np.array(dv)
            np.savez(OUT, **store)
            print(f"  j={j} k={k} ({k*d:,} params): dense = {np.mean(dv):.4f}"
                  f"+-{np.std(dv, ddof=1) if NSEED > 1 else 0:.4f}  "
                  f"[{time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB]",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()
    print("\ndone", flush=True)


if __name__ == '__main__':
    main()
