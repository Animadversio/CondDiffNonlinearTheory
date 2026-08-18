"""Circulant with MORE ROWS than dense at the same x = k/d, but FEWER free parameters.

At dense width k = j*d (x-axis position j = k/d), run the circulant with c blocks so that
its row count c*d is a multiple m of the dense row count:  c = m*j,  m in {2,4,8}.
Reported alongside dense so the trade is explicit: the circulant has m x the rows and
d/m x FEWER free parameters (c*d against k*d).

Also emits the fixed-c reading (c literally 2/4/8) for comparison; note that one has fewer
rows than dense once k/d > c, so it does not match the "total rows larger" description.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scripts.rf_circulant_dnn_layers import extract, standardize, DEV, DT, LAM
from core.rf_circulant_struct import circulant_rf_mmse_struct

NIMG = int(os.environ.get('NIMG', '10000'))
JS   = [int(x) for x in os.environ.get('JS', '1,2,4,8,16').split(',')]
MS   = [int(x) for x in os.environ.get('MS', '2,4,8').split(',')]
SIGS = [float(x) for x in os.environ.get('SIGS', '0.010,0.036,0.127,0.452,2.212,7.880').split(',')]
MODE = os.environ.get('MODE', 'A')        # A: c = m*j   B: c = m (fixed)
NSEED = int(os.environ.get('NSEED', '2'))

X = standardize(extract('CIFAR-10', n_img=NIMG)['avgpool'].to(DEV, DT))
N, d = X.shape
print(f"CIFAR-10 avgpool: N={N}, d={d}   mode={MODE}  (A: c=m*k/d, B: c=m fixed)\n")
res = {}
for j in JS:
    for m in MS:
        c = m * j if MODE == 'A' else m
        for sg in SIGS:
            vals = []
            for sd in range(NSEED):
                g = torch.Generator(device=DEV); g.manual_seed(500 + 31 * sd + c)
                h = torch.randn(c, d, generator=g, device=DEV, dtype=DT) / np.sqrt(d)
                t0 = time.time()
                vals.append(circulant_rf_mmse_struct(X, h, sg, lam=LAM, device=DEV,
                                                     block_chunk=min(c, 32)))
                del h; torch.cuda.empty_cache()
            v = np.array(vals)
            res[(j, m, sg)] = (v.mean(), v.std(ddof=1) if NSEED > 1 else 0.0)
            print(f"  k/d={j:>2} m={m} c={c:>4} rows={c*d:>7} params={c*d:>7} "
                  f"(dense rows={j*d:>6}, params={j*d*d:>9})  sigma={sg:<6} "
                  f"L={v.mean():>9.4f}+-{v.std(ddof=1) if NSEED>1 else 0:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            np.savez(f'tables/rf_circulant_rowmult_{MODE}.npz',
                     **{f'{k[0]}|{k[1]}|{k[2]}': np.array(v) for k, v in res.items()})
print("done")
