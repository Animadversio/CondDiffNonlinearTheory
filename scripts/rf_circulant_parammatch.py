"""Free-parameter-matched circulant curve, added to the loss-vs-k/d figure.

At x-axis position k/d = c0 the dense model has k = c0*d features and k*d free parameters
in each of Theta and W. The parameter-matched circulant uses c = k BLOCKS, so it also has
c*d = k*d free parameters -- and therefore c*d = k*d TOTAL features (262,144 at c0=1,
d=512). L^circ is evaluated with core.rf_circulant_struct, which never forms the K x K
covariance and is validated against the reference build to ~1e-15.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scripts.rf_circulant_dnn_layers import extract, standardize, DEV, DT, LAM
from core.rf_circulant_struct import circulant_rf_mmse_struct

NIMG   = int(os.environ.get('NIMG', '10000'))
C_LIST = [int(x) for x in os.environ.get('C_LIST', '1,2').split(',')]
BC     = int(os.environ.get('BC', '32'))
SIGS   = [float(x) for x in os.environ.get('SIGS', '0.010,0.127,0.452,2.212').split(',')]

X = standardize(extract('CIFAR-10', n_img=NIMG)['avgpool'].to(DEV, DT))
N, d = X.shape
print(f"CIFAR-10 avgpool: N={N}, d={d}\n")
out = {}
for c0 in C_LIST:
    c = c0 * d                                  # c = k blocks  =>  k*d total features
    g = torch.Generator(device=DEV); g.manual_seed(100 + c0)
    h = torch.randn(c, d, generator=g, device=DEV, dtype=DT) / np.sqrt(d)
    for sg in SIGS:
        t0 = time.time()
        L = circulant_rf_mmse_struct(X, h, sg, lam=LAM, device=DEV, block_chunk=BC)
        out[(c0, sg)] = L
        print(f"  k/d={c0:>2} c={c:>5} K={c*d:>9} sigma={sg:<6} L^circ_pm={L:>10.4f} "
              f"({time.time()-t0:.0f} s)", flush=True)
        np.savez('tables/rf_circulant_parammatch.npz',
                 **{f'{k[0]}|{k[1]}': v for k, v in out.items()})
    del h; torch.cuda.empty_cache()
print("done")
