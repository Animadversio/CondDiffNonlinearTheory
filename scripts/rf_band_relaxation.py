"""BAND vs COMB: the readout relaxation `rf_equivariance_toll.py` §4 did not price.

Context
-------
`docs/rf_blockdiag_derivation_and_toll.md` §5 prices relaxing the equivariant readout along
ONE axis: period-P sharing, `A` commutes with `S^P`, i.e. the filter may depend on position
mod P.  The verdict was "no cheap interior point: to remove the toll you have to break
equivariance almost completely".

That verdict is about the axis, not about the toll.  Period-P is a COMB in frequency --
it couples output f to inputs f + j*(d/P), i.e. it lets the filter vary at spatial
period P.  Image non-stationarity is the opposite: smooth, whole-image scale (objects
centred, borders unlike interiors, sky above).  §5 says as much and then prices the wrong
thing, because the search was over SUBGROUPS of Z_d and the relaxation that matches smooth
non-stationarity is not a subgroup.

The matching relaxation is a BAND:

    A = sum_{s in band} diag(e_s) * BCCB(a_s),     e_s[p] = exp(2*pi*i*<s,p>/32)

"a convolution whose kernel varies smoothly across the image, band-limited to |s|<=B
spatial modes".  In frequency, output f reads inputs f+Delta for |Delta|<=B.

Why it keeps the fast path: each parameter `a_s[g]` enters exactly one output frequency
f = g+s, so the normal equations still decouple into 1024 independent blocks -- of size
3(2B+1)^2 instead of 3.  Nothing else in the pipeline changes.  (Equivariance gives
decoupling by symmetry, block-diagonality by direct sum; this is a third route -- the
parameters are partitioned by f regardless of any group.)

Self-check: B=0 is exactly `period2d(p=1)`, and the script asserts it.

Run
---
    python scripts/rf_band_relaxation.py              # ~90 s, one GPU
    SIGS=0.127,0.452,1.610,5.0 BS=0,1,2,3 python scripts/rf_band_relaxation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.rf_pixel_featmatch2 import load, DEV, DT, d
from scripts.rf_equivariance_toll import period2d

H = Wd = 32
SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610,5.0').split(',')]
BS = [int(x) for x in os.environ.get('BS', '0,1,2,3,4').split(',')]
OUT = 'tables/rf_band_relaxation.npz'
CIRC_RF = {0.127: 8.029, 0.452: 31.479, 1.610: 80.511, 5.0: 139.276}


def fourier_cov(Xc):
    """Chat[a,b] = E[ xhat_a conj(xhat_b) ] over the unitary 2-D DFT, index (ch, fr, fc)."""
    N = Xc.shape[0]
    Xh = torch.fft.fft2(Xc.reshape(N, 3, H, Wd), norm='ortho').reshape(N, d)
    return (Xh.conj().T @ Xh).conj() / N


def band_index(B):
    """win (1024, 3nb) input coords, out (1024, 3) output coords, nb = (2B+1)^2."""
    offs = torch.tensor([(a, b) for a in range(-B, B + 1) for b in range(-B, B + 1)],
                        device=DEV)
    nb = offs.shape[0]
    f = torch.arange(H * Wd, device=DEV)
    fr, fc = f // Wd, f % Wd
    base = ((fr[:, None] + offs[None, :, 0]) % H) * Wd + (fc[:, None] + offs[None, :, 1]) % Wd
    ch = torch.arange(3, device=DEV) * (H * Wd)
    return ((base[:, :, None] + ch).reshape(H * Wd, 3 * nb),
            (fr * Wd + fc)[:, None] + ch, nb)


def band2d(Chat, TR, B, sig, chunk=128):
    """Best linear denoiser whose kernel varies smoothly (band-limited to B modes) in space,
    with free 3-channel mixing.  B=0 == strictly equivariant == period2d(p=1)."""
    win, out, nb = band_index(B)
    m = 3 * nb
    I = (sig * sig) * torch.eye(m, device=DEV, dtype=Chat.dtype)
    red = 0.0
    for lo in range(0, H * Wd, chunk):
        w, o = win[lo:lo + chunk], out[lo:lo + chunk]
        P = Chat[w[:, :, None], w[:, None, :]] + I
        q = Chat[w[:, :, None], o[:, None, :]]
        red += float(torch.einsum('fam,fam->', q.conj(), torch.linalg.solve(P, q)).real)
    return TR - red, 3 * H * Wd * m, nb


def main():
    X = load()
    Xc = X - X.mean(0)
    N = Xc.shape[0]
    Sig = (Xc.T @ Xc) / N
    TR = float(torch.diagonal(Sig).sum())
    ev = torch.linalg.eigvalsh(Sig)
    lin = {s: float((s ** 2 * ev / (ev + s ** 2)).sum()) for s in SIGS}
    Chat = fourier_cov(Xc)
    assert abs(float(Chat.diagonal().sum().real) - TR) < 1e-6 * TR, "Parseval check failed"

    store = {'sigmas': np.array(SIGS), 'linear': np.array([lin[s] for s in SIGS])}
    hdr = "".join(f"{f'sg={s}':>18}" for s in SIGS)
    print(f"CIFAR-10 raw pixels  d={d}  N={N}  Tr(Sigma)={TR:.3f}")
    print(f"\n{'model':>26} {'W params':>12} |{hdr}")
    print(f"{'free Wiener':>26} {d*d:>12,} |"
          + "".join(f"{lin[s]:>10.3f}{0.0:>+8.3f}" for s in SIGS))
    print(f"{'block-circ RF (measured)':>26} {'':>12} |"
          + "".join(f"{CIRC_RF.get(s, float('nan')):>10.3f}"
                    f"{CIRC_RF.get(s, float('nan'))-lin[s]:>+8.3f}" for s in SIGS))

    print(f"\n  COMB -- period-p (the axis already priced)")
    ref = None
    for p in (1, 2, 4, 8, 16):
        L = [period2d(Xc, TR, p, s) for s in SIGS]
        store[f'period2d|{p}'] = np.array(L)
        if p == 1:
            ref = L
        print(f"{f'period2d p={p}':>26} {3*d*p*p:>12,} |"
              + "".join(f"{a:>10.3f}{a-lin[s]:>+8.3f}" for s, a in zip(SIGS, L)))

    print(f"\n  BAND -- smoothly space-varying kernel (the proposal)")
    for B in BS:
        r = [band2d(Chat, TR, B, s) for s in SIGS]
        L = [x[0] for x in r]
        npar, nb = r[0][1], r[0][2]
        store[f'band2d|{B}'] = np.array(L)
        print(f"{f'band2d B={B} ({nb} modes)':>26} {npar:>12,} |"
              + "".join(f"{a:>10.3f}{a-lin[s]:>+8.3f}" for s, a in zip(SIGS, L)))
        if B == 0:
            err = max(abs(a - b) for a, b in zip(L, ref))
            print(f"{'':>26} {'':>12} |  SELFTEST max|band B=0 - period p=1| = {err:.2e}"
                  f"   {'OK' if err < 1e-8 else 'MISMATCH'}")
            assert err < 1e-8

    print(f"\n{'='*104}\nTOLL REMOVED PER MILLION READOUT PARAMETERS "
          f"(higher = better use of the relaxation budget)\n{'='*104}")
    print(f"{'model':>26} {'W params':>12} |{hdr}")
    base = store['period2d|1']
    for key, lab, npar in ([(f'period2d|{p}', f'period2d p={p}', 3 * d * p * p)
                            for p in (2, 4, 8, 16)]
                           + [(f'band2d|{B}', f'band2d B={B}', 3 * H * Wd * 3 * (2 * B + 1) ** 2)
                              for B in BS if B > 0]):
        if key not in store:
            continue
        print(f"{lab:>26} {npar:>12,} |"
              + "".join(f"{(b - a) / (npar / 1e6):>18.2f}"
                        for a, b in zip(store[key], base)))

    os.makedirs('tables', exist_ok=True)
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
