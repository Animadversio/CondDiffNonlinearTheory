"""Relaxing the equivariant readout by letting the kernel vary SMOOTHLY across the image.

The question (michimin, 2026-09-22)
-----------------------------------
`scripts/rf_equivariance_toll.py` priced one relaxation of the equivariance constraint:
period-`P` sharing, where the filter depends on position `u mod P`.  It was nearly flat
where it was affordable, which suggested the non-stationarity of CIFAR lives at the scale
of the whole 32x32 image rather than at any short period.  This script prices the
relaxation that hypothesis actually points at: let the kernel vary **smoothly** with
position, i.e. be band-limited in `u`.

The class
---------
Write a general linear denoiser with a position-dependent kernel

    (A y)_{c,u} = sum_{c',v}  K_{c,c'}(u, v)  y_{c', u-v},          u in Z_32 x Z_32

and constrain the `u`-dependence to a set `G` of modulation frequencies:

    K(u, v) = sum_{g in G}  exp(2 pi i g.u / 32)  K_g(v),      K real  =>  G = -G.

`G = {0}` is exact equivariance (a real CNN layer, free 3-channel mixing).  Two families:

    band B    G = box(B)        = { g : |g_1|, |g_2| <= B }          |G| = (2B+1)^2
    period p  G = sublattice(p) = { g : g = (32/p) * integer }       |G| = p^2

**They are the same family with a different G, and the same parameter budget:**

    W trained params  =  9 (channels) * 1024 (taps) * |G|  =  9216 |G|

so period-p costs 9216 p^2 (reproducing the period2d row of rf_equivariance_toll.py) and
band-B costs 9216 (2B+1)^2.  The only difference is WHICH modulation frequencies are
bought: band-B buys the |G| lowest (smooth variation across the image), period-p buys a
spread-out sublattice.  Note period-2's sublattice is {0,16}^2 -- it spends its 4x
parameters on the Nyquist modulation, i.e. checkerboard variation.

Why it still decouples
----------------------
In Fourier the class couples output frequency `f` to input frequency `f-g`, `g in G`:

    (A y)^(f)  =  sum_{g in G}  K_g^(f-g)  y^(f-g).

The parameter `K_g^(h)` feeds output frequency `h+g` and no other, so the normal equations
split over OUTPUT FREQUENCY into 1024 least squares of size 3|G|:

    P_f = E[ y_G(f) y_G(f)^H ] = Sx_G(f) + s^2 I   (3|G| x 3|G|),
    C_f = E[ x^(f) y_G(f)^H ]                      (3 x 3|G|),
    L   = sum_f [ Tr Sx(f) - Tr( C_f P_f^{-1} C_f^H ) ].

B=3 is 147x147 over 1024 frequencies: seconds.

Reality
-------
K real <=> K_g^(h)* = K_{-g}^(-h), which links output frequency f to -f.  For f != -f the
two blocks are independent conjugate copies, so the unconstrained complex solve at each is
exact.  At the 4 SELF-CONJUGATE frequencies (0,0), (0,16), (16,0), (16,16) the constraint
bites *within* the block and an unconstrained complex solve is optimistic; those are done
as a real constrained solve instead (`_selfconj_block`).  `STRICT=0` disables the
correction so its size can be measured.

Validation (all asserted, the run stops if any fails)
    1. band B=0          == period2d(p=1)   of rf_equivariance_toll.py
    2. sublattice(p)     == period2d(p)     for p = 2, 4, 8   -- the strong version:
                                            same solver, non-trivial G, reproduces the
                                            independently-written polyphase code
    3. SELFTEST=1        == brute force over an explicit REAL parameterization at
                                            3 channels x 4 x 4, data non-Gaussian and
                                            non-stationary

    python scripts/rf_band_relaxation_atlas.py
    SELFTEST=1 python scripts/rf_band_relaxation_atlas.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.rf_pixel_featmatch2 import load, DEV, DT, d
from scripts.rf_equivariance_toll import period2d

SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610,5.0').split(',')]
BANDS = [int(x) for x in os.environ.get('BANDS', '0,1,2,3').split(',')]
PERIODS = [int(x) for x in os.environ.get('PERIODS', '1,2,4,8,16').split(',')]
STRICT = os.environ.get('STRICT', '1') != '0'
FCHUNK = int(os.environ.get('FCHUNK', '256'))
OUT = 'tables/rf_band_relaxation_atlas.npz'

NS, NCH = 32, 3                                   # spatial side, colour channels
CPLX = torch.complex128


# ----------------------------------------------------------------- modulation-frequency sets

def box(B, n=NS):
    """the (2B+1)^2 LOWEST modulation frequencies -- kernel varies smoothly across u."""
    assert 2 * B + 1 <= n, f"box(B={B}) wraps at n={n}"
    r = [g % n for g in range(-B, B + 1)]
    return [(a, b) for a in r for b in r]


def sublattice(p, n=NS):
    """the p^2 multiples of n/p -- kernel depends on u mod p.  This IS period-p."""
    assert n % p == 0
    m = n // p
    return [(i * m, j * m) for i in range(p) for j in range(p)]


def nparams(G, nch=NCH, n=NS):
    return nch * nch * n * n * len(G)


# ----------------------------------------------------------------------------- the estimator

def _selfconj_block(Z, f0, G, s, n, nch):
    """Exact constrained solve at a self-conjugate output frequency f0 (f0 == -f0).

    There the reality constraint K_g^(h)* = K_{-g}^(-h) links g to -g INSIDE the block, so
    the complex coefficients are not free.  Pairing g with -g, and using
    y^(f0+g) = conj(y^(f0-g)), the pair contributes

        M_g y^(f0-g) + conj(M_g) conj(y^(f0-g)) = 2 Re[M_g y^(f0-g)]
                                                = A_g (2 Re y) + B_g (-2 Im y)

    with A_g, B_g free REAL 3x3.  So the block is a real least squares whose regressors are
    2 Re y^(f0-g) and -2 Im y^(f0-g) (noise variance 2 s^2 each, since Var Re z^ = 1/2),
    plus y^(f0-g) itself for the self-conjugate g (real, noise variance s^2).
    """
    a0, b0 = f0
    seen, cols, nvar = set(), [], []
    for g in G:
        if g in seen:
            continue
        gm = ((-g[0]) % n, (-g[1]) % n)
        v = Z[:, :, ((a0 - g[0]) % n) * n + ((b0 - g[1]) % n)]        # (N, nch) complex
        if gm == g:                                                    # h = f0-g self-conj
            seen.add(g); cols.append(v.real); nvar += [s * s] * nch
        else:
            seen.update((g, gm))
            cols += [2 * v.real, -2 * v.imag]; nvar += [2 * s * s] * (2 * nch)
    U = torch.cat(cols, 1)                                             # (N, 3|G|) real
    x = Z[:, :, a0 * n + b0].real                                      # (N, nch) real
    N = U.shape[0]
    P = (U.T @ U) / N + torch.diag(torch.tensor(nvar, device=U.device, dtype=U.dtype))
    C = (x.T @ U) / N
    Sx = (x.T @ x) / N
    return float(torch.diagonal(Sx).sum() - torch.einsum('ab,ab->', C, torch.linalg.solve(P, C.T).T))


def band_loss(Z, G, s, n=NS, nch=NCH, strict=STRICT, fchunk=FCHUNK):
    """min over the class of E||x0 - A y||^2, y = x0 + s z.  Z = unitary fft2 of centred x0."""
    N, F, ng = Z.shape[0], n * n, len(G)
    Zf = Z.reshape(N, nch * F)
    R = ((Zf.T @ Zf.conj()) / N).reshape(nch, F, nch, F).permute(1, 3, 0, 2).contiguous()
    fa, fb = torch.arange(F, device=Z.device) // n, torch.arange(F, device=Z.device) % n
    IDX = torch.stack([((fa - g[0]) % n) * n + ((fb - g[1]) % n) for g in G], 1)   # (F, ng)
    I = torch.eye(ng * nch, device=Z.device, dtype=CPLX) * (s * s)

    tot = 0.0
    for lo in range(0, F, fchunk):
        hi = min(lo + fchunk, F)
        ix, nf = IDX[lo:hi], hi - lo
        P = R[ix.unsqueeze(2), ix.unsqueeze(1)]                # (nf, ng, ng, nch, nch)
        P = P.permute(0, 1, 3, 2, 4).reshape(nf, ng * nch, ng * nch) + I
        C = R[torch.arange(lo, hi, device=Z.device).unsqueeze(1), ix]   # (nf, ng, nch, nch)
        C = C.permute(0, 2, 1, 3).reshape(nf, nch, ng * nch)
        Sx = R[torch.arange(lo, hi, device=Z.device), torch.arange(lo, hi, device=Z.device)]
        tot += float((torch.einsum('fcc->', Sx)
                      - torch.einsum('fab,fab->', C, torch.linalg.solve(P, C.conj().transpose(1, 2))
                                     .transpose(1, 2))).real)

    if strict:                       # replace the 4 optimistic blocks with the exact ones
        for f0 in [(a, b) for a in (0, n // 2) for b in (0, n // 2)]:
            ff = f0[0] * n + f0[1]
            Sx = R[ff, ff]
            ix = IDX[ff]
            P = R[ix.unsqueeze(1), ix.unsqueeze(0)].permute(0, 2, 1, 3).reshape(ng * nch, ng * nch) + I
            C = R[ff, ix].permute(1, 0, 2).reshape(nch, ng * nch)
            loose = float((torch.diagonal(Sx).sum()
                           - torch.einsum('ab,ab->', C, torch.linalg.solve(P, C.conj().T).T)).real)
            tot += _selfconj_block(Z, f0, G, s, n, nch) - loose
    return tot


def to_freq(Xc, n=NS, nch=NCH):
    """unitary 2-D DFT over space; Parseval => sum_f Tr Sx(f) = Tr Sigma."""
    return torch.fft.fft2(Xc.reshape(-1, nch, n, n), norm='ortho').reshape(-1, nch, n * n)


# ------------------------------------------------------------------------ brute-force check

def selftest(n=4, nch=3, B=1, N=40000, s=0.7, seed=0):
    """Explicit REAL parameterization, solved directly.  Catches any reality-constraint slip.

    Basis of the class: for each real modulation ψ with spectral support in G (|G| of them),
    each shift v, and each (c,c'), the operator (E y)_{c,u} = ψ(u) y_{c',u-v}.  That is
    exactly nch^2 * n^2 * |G| real parameters, and A = Σ θ_i E_i spans the class.
    """
    torch.manual_seed(seed)
    dd, F = nch * n * n, n * n
    # non-Gaussian AND non-stationary: smooth random field * position-dependent gain, skewed
    W = torch.randn(dd, dd, device=DEV, dtype=DT) / np.sqrt(dd)
    Zr = torch.randn(N, dd, device=DEV, dtype=DT)
    X = (Zr @ W) + 0.6 * (Zr[:, :dd] ** 2 - 1.0) @ W.abs()
    gain = (1.0 + torch.arange(n * n, device=DEV, dtype=DT) / (n * n)).repeat(nch)
    X = (X * gain); X = X - X.mean(0)

    G = box(B, n)
    # real modulation basis: cos for self-conjugate g, (cos, sin) for each +-g pair
    u1, u2 = torch.arange(n, device=DEV, dtype=DT).repeat_interleave(n), \
             torch.arange(n, device=DEV, dtype=DT).repeat(n)
    seen, psis = set(), []
    for g in G:
        if g in seen:
            continue
        gm = ((-g[0]) % n, (-g[1]) % n)
        ph = 2 * np.pi * (g[0] * u1 + g[1] * u2) / n
        if gm == g:
            seen.add(g); psis.append(torch.cos(ph))
        else:
            seen.update((g, gm)); psis += [torch.cos(ph), torch.sin(ph)]
    assert len(psis) == len(G), (len(psis), len(G))

    E = []
    for psi in psis:
        for v in range(F):
            Sh = torch.zeros(F, F, device=DEV, dtype=DT)          # y_{u-v} <- cyclic shift
            src = ((torch.arange(n, device=DEV).repeat_interleave(n) - v // n) % n) * n \
                  + ((torch.arange(n, device=DEV).repeat(n) - v % n) % n)
            Sh[torch.arange(F, device=DEV), src] = 1.0
            DS = psi[:, None] * Sh                                 # (F, F)
            for c in range(nch):
                for cp in range(nch):
                    M = torch.zeros(dd, dd, device=DEV, dtype=DT)
                    M[c * F:(c + 1) * F, cp * F:(cp + 1) * F] = DS
                    E.append(M)
    E = torch.stack(E)                                             # (P, dd, dd)
    assert E.shape[0] == nparams(G, nch, n), (E.shape[0], nparams(G, nch, n))

    Sx = (X.T @ X) / N
    Sy = Sx + s * s * torch.eye(dd, device=DEV, dtype=DT)
    Ef = E.reshape(E.shape[0], -1)
    H = (E @ Sy).reshape(E.shape[0], -1) @ Ef.T
    b = Ef @ Sx.T.reshape(-1)
    L_brute = float(torch.diagonal(Sx).sum() - b @ torch.linalg.solve(H, b))
    L_form = band_loss(to_freq(X, n, nch), G, s, n, nch)
    print(f"  brute force {L_brute:.14f}   formula {L_form:.14f}   diff {abs(L_brute-L_form):.3e}")
    assert abs(L_brute - L_form) < 1e-8, "band-limited loss formula FAILED"
    print("  SELFTEST PASSED")


# --------------------------------------------------------------------------------- main

def main():
    if os.environ.get('SELFTEST'):
        print("brute-force check, 3 channels x 4 x 4, non-Gaussian non-stationary data")
        selftest(); return

    X = load(); Xc = X - X.mean(0)
    Sig = (Xc.T @ Xc) / X.shape[0]
    ev = torch.linalg.eigvalsh(Sig)
    TR = float(torch.diagonal(Sig).sum())
    lin = {s: float((s ** 2 * ev / (ev + s ** 2)).sum()) for s in SIGS}
    Z = to_freq(Xc)
    print(f"CIFAR-10 raw pixels  d={d}  N={X.shape[0]}  Tr(Sigma)={TR:.3f}")
    print(f"Parseval check: sum_f Tr Sx(f) = {float((Z.abs()**2).mean(0).sum()):.6f} vs {TR:.6f}")
    hdr = "".join(f"{f'sg={s}':>18}" for s in SIGS)

    # ---- validation 1+2: the general-G solver must reproduce the polyphase period2d code
    print(f"\n{'='*92}\nVALIDATION  general-G solver vs period2d() of rf_equivariance_toll.py\n{'='*92}")
    worst = 0.0
    for p in (1, 2, 4, 8):
        for s in SIGS:
            a = band_loss(Z, sublattice(p), s)
            b_ = period2d(Xc, TR, p, s)
            worst = max(worst, abs(a - b_))
            print(f"  p={p:<3} sg={s:<6} G-solver {a:12.6f}   period2d {b_:12.6f}   diff {abs(a-b_):.2e}")
    assert worst < 1e-6, f"general-G solver does NOT reproduce period2d (worst {worst:.3e}) -- STOP"
    print(f"  PASSED (worst |diff| = {worst:.2e});  in particular band B=0 == period2d p=1")

    store = {'sigmas': np.array(SIGS), 'linear': np.array([lin[s] for s in SIGS])}
    print(f"\n{'='*92}\nTOLL vs PARAMETERS   (L, and excess over the free Wiener denoiser)\n{'='*92}")
    print(f"{'class':>16}{'|G|':>6}{'W params':>12} |" + hdr)
    rows = ([(f'band B={B}', box(B)) for B in BANDS]
            + [(f'period p={p}', sublattice(p)) for p in PERIODS])
    for nm, G in rows:
        L = [band_loss(Z, G, s) for s in SIGS]
        key = f"band|{nm.split('=')[1]}" if nm.startswith('band') else f"period|{nm.split('=')[1]}"
        store[key] = np.array(L); store[key + '|np'] = np.array([nparams(G)])
        print(f"{nm:>16}{len(G):>6}{nparams(G):>12,} |"
              + "".join(f"{a:>10.3f}{a-lin[s]:>+8.3f}" for s, a in zip(SIGS, L)))

    if STRICT:                        # how much the 4 self-conjugate blocks actually matter
        print(f"\n  (self-conjugate correction, band B=3: "
              + ", ".join(f"{band_loss(Z, box(3), s, strict=False) - band_loss(Z, box(3), s):+.2e}"
                          for s in SIGS) + " -- STRICT=0 is optimistic by this much)")

    os.makedirs('tables', exist_ok=True)
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
