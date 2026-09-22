"""What the circulant model's structure costs, measured EXACTLY in the linear world.

Motivation
----------
The matched-free-parameter tables say the block-circulant RF denoiser loses to the plain
linear (Wiener) denoiser at every sigma, by +0.13 / +2.93 / +8.32 / +8.97 at
sigma = 0.127 / 0.452 / 1.610 / 5.0.  Two hypotheses for where that goes:

  (A) TAP STARVATION.  A banded block-circulant Theta draws only c*t random numbers
      (49,152 at c=6144, t=8) versus dense's k*d = 18.9M -- 384x fewer.  Fix: renew taps.
  (B) READOUT TOLL.  A block-circulant W forces the whole predictor to commute with the
      cyclic shift.  CIFAR is not translation-stationary, so that constraint costs
      something no feature map can recover.

These are separable WITHOUT running any RF model, because hypothesis (B) already bites on
the LINEAR denoiser, where every quantity is a closed-form generalized eigenproblem:

    L_free       = min over all A in R^{dxd}            = the Wiener denoiser
    L_equivariant= min over A commuting with the shift  = per-character 1x1 (or mxm) solves
    toll         = L_equivariant - L_free

The toll is computed over ALL equivariant linear maps at once -- it is an exact minimum, so
it has no taps to starve.  If the toll alone accounts for the RF model's deficit, (A) is
not the binding constraint and renewing taps cannot fix it.

What is computed
----------------
1. TILED    x_j = A_j y_j on non-overlapping b-blocks.  This is michimin's proposed
            "Theta is 8x8 circulant along the diagonal, W the same" in its linear shadow.
            Also its BAYES floor sum_j MMSE(x0_j | y_j) -- the best ANY nonlinear tiled
            model can do -- by exact posterior weights over the empirical measure.
            (In b=8 dims with 10^4 atoms the posterior is well sampled; N_eff is reported
            so this is not the N_eff=1 memorization artifact of the d=3072 case.)
2. SLIDING  x_i from a width-w stride-1 window -- isolates OVERLAP from RECEPTIVE FIELD.
3. CONV2D   x_i from a true ks x ks spatial patch, with and without 3-channel mixing.
            The raster layout is channel-major, so 8 contiguous coords are 8 HORIZONTAL
            pixels of ONE colour channel; a fair CNN comparison needs real 2-D patches.
4. PERIOD-P A commutes with S^P (1-D) or with spatial shifts by p (2-D).  Interpolates
            equivariant (P=1) to free (P=d).  This is the payoff curve for relaxing the
            readout, priced before any RF code is written.

    python scripts/rf_equivariance_toll.py            # table + tables/rf_equivariance_toll.npz
    NO_BAYES=1 python scripts/rf_equivariance_toll.py # skip the (slower) tiled Bayes floor
    SELFTEST=1 python scripts/rf_equivariance_toll.py # brute-force check of the block-diag formula
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.rf_pixel_featmatch2 import load, DEV, DT, d

SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610,5.0').split(',')]
NT = int(os.environ.get('NT', '2000'))          # test draws per block for the Bayes floor
OUT = 'tables/rf_equivariance_toll.npz'

# measured block-circulant RF losses at matched free parameters (c=6144, or the largest c
# computed at that sigma).  Sources: tables/rf_pixel_featmatch2.npz, logs/featmatch_sig161.log.
CIRC_RF = {0.127: 8.029, 0.452: 31.479, 1.610: 80.511, 5.0: 139.276}


def setup():
    X = load()
    Xc = X - X.mean(0)
    Sig = (Xc.T @ Xc) / X.shape[0]
    ev = torch.linalg.eigvalsh(Sig)
    lin = {s: float((s ** 2 * ev / (ev + s ** 2)).sum()) for s in SIGS}
    return X, Xc, Sig, float(torch.diagonal(Sig).sum()), lin


def tiled(Sig, TR, b, s):
    M = d // b
    idx = torch.arange(d, device=DEV).reshape(M, b)
    S = torch.stack([Sig[i][:, i] for i in idx])
    I = torch.eye(b, device=DEV, dtype=DT).expand(M, b, b)
    A = torch.linalg.solve(S + s * s * I, S)
    return float(torch.einsum('jii->', S) - torch.einsum('jab,jab->', A, S))


def sliding(Sig, TR, w, s):
    off = torch.arange(w, device=DEV) - w // 2
    idx = (torch.arange(d, device=DEV)[:, None] + off[None, :]) % d
    S = Sig[idx[:, :, None], idx[:, None, :]]
    sv = Sig[torch.arange(d, device=DEV)[:, None], idx]
    I = torch.eye(w, device=DEV, dtype=DT).expand(d, w, w)
    a = torch.linalg.solve(S + s * s * I, sv.unsqueeze(-1)).squeeze(-1)
    return TR - float((a * sv).sum())


def conv2d(Sig, TR, ks, allch, s):
    """x0_i from a ks x ks spatial patch (cyclic), all 3 channels if allch."""
    ch, rw, cl = torch.meshgrid(torch.arange(3), torch.arange(32), torch.arange(32),
                                indexing='ij')
    ch, rw, cl = ch.reshape(-1).to(DEV), rw.reshape(-1).to(DEV), cl.reshape(-1).to(DEV)
    o = torch.arange(ks, device=DEV) - ks // 2
    dr, dc = torch.meshgrid(o, o, indexing='ij')
    r = (rw[:, None] + dr.reshape(-1)[None, :]) % 32
    c = (cl[:, None] + dc.reshape(-1)[None, :]) % 32
    if allch:
        idx = (torch.arange(3, device=DEV)[None, :, None] * 1024
               + (r * 32 + c)[:, None, :]).reshape(d, -1)
    else:
        idx = ch[:, None] * 1024 + (r * 32 + c)
    w = idx.shape[1]
    S = Sig[idx[:, :, None], idx[:, None, :]]
    sv = Sig[torch.arange(d, device=DEV)[:, None], idx]
    I = torch.eye(w, device=DEV, dtype=DT).expand(d, w, w)
    a = torch.linalg.solve(S + s * s * I, sv.unsqueeze(-1)).squeeze(-1)
    return w, TR - float((a * sv).sum())


def period1d(Xc, TR, P, s):
    """best linear A commuting with S^P.  polyphase (N, d/P, P), DFT over the coarse axis."""
    N, M = Xc.shape[0], d // P
    F = torch.fft.fft(Xc.reshape(N, M, P), dim=1) / np.sqrt(M)
    Pm = torch.einsum('nma,nmb->mab', F, F.conj()) / N
    I = torch.eye(P, device=DEV, dtype=Pm.dtype)[None]
    return TR - float(torch.einsum('mab,mba->', Pm, torch.linalg.solve(Pm + s * s * I, Pm)).real)


def period2d(Xc, TR, p, s):
    """A equivariant to spatial shifts by p on both axes, free 3-channel mixing."""
    N, m, B = Xc.shape[0], 32 // p, 3 * p * p
    V = Xc.reshape(N, 3, m, p, m, p).permute(0, 2, 4, 1, 3, 5).reshape(N, m, m, B)
    F = (torch.fft.fft2(V, dim=(1, 2)) / m).reshape(N, m * m, B).permute(1, 0, 2)
    Pm = torch.einsum('fna,fnb->fab', F, F.conj()) / N
    I = torch.eye(B, device=DEV, dtype=Pm.dtype)[None]
    return TR - float(torch.einsum('fab,fba->', Pm, torch.linalg.solve(Pm + s * s * I, Pm)).real)


def tiled_bayes(X, b, s, nt):
    """sum_j MMSE(x0_j | y_j) on the empirical measure, plus the median posterior N_eff."""
    N, M = X.shape[0], d // b
    idx = torch.arange(d, device=DEV).reshape(M, b)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    sel = torch.randperm(N, device=DEV, generator=g)[:nt]
    tot, neff = 0.0, []
    for j in range(M):
        A = X[:, idx[j]]
        y = A[sel] + s * torch.randn(nt, b, device=DEV, dtype=DT, generator=g)
        w = torch.softmax(-(torch.cdist(y, A) ** 2) / (2 * s * s), dim=1)
        tot += float(((w @ A - A[sel]) ** 2).sum()) / nt
        if j % 64 == 0:
            H = -(w * torch.log(w.clamp_min(1e-30))).sum(1)
            neff.append(float(torch.exp(H).median()))
    return tot, float(np.median(neff))


def selftest(dd=32, b=4, c=2, N=40000, sig=0.7, seed=0):
    """Brute-force check of  L = Tr(Sig_p0) - sum_{j,f} q^H P^-1 q  for the model
    'Theta = c blocks of dxd, each block-diagonal with bxb circulant blocks; W the same'.

    Deliberately run on data with STRONG cross-chunk correlation and non-Gaussian
    marginals, because the claim under test is precisely that cross-chunk feature moments
    are nonzero yet do not enter the optimum.  Also reports the analytic noise
    cross-covariance sig^2 (Theta_a Theta_b^T)_[j,j'], which IS exactly zero.
    """
    g = torch.Generator().manual_seed(seed)
    M = dd // b
    rows = (torch.arange(b)[:, None] - torch.arange(b)[None, :]) % b
    A = torch.randn(dd, dd, dtype=torch.float64, generator=g)
    Lc = torch.linalg.cholesky(A @ A.T / dd + 0.3)
    x0 = (torch.randn(N, dd, dtype=torch.float64, generator=g) @ Lc.T) ** 2 * 0.4
    x0 = x0 - x0.mean(0)
    y = x0 + sig * torch.randn(N, dd, dtype=torch.float64, generator=g)
    h = torch.randn(c, M, b, dtype=torch.float64, generator=g) / np.sqrt(b)
    Th = torch.zeros(c, dd, dd, dtype=torch.float64)
    for a in range(c):
        for j in range(M):
            Th[a, j*b:(j+1)*b, j*b:(j+1)*b] = h[a, j][rows]
    phi = torch.relu(torch.einsum('aij,nj->nai', Th, y))

    xch = max(float(((sig**2) * (Th[a] @ Th[bb].T))[j*b:(j+1)*b, jp*b:(jp+1)*b].abs().max())
              for a in range(c) for bb in range(c)
              for j in range(M) for jp in range(M) if j != jp)
    P4 = (phi - phi.mean(0)).reshape(N, c, M, b)
    fx = max(float(torch.einsum('na,nb->ab', P4[:, a, j], P4[:, bb, jp]).abs().max()) / N
             for a in range(c) for bb in range(c)
             for j in range(M) for jp in range(M) if j != jp)

    cols = []
    for a in range(c):
        for j in range(M):
            pj = phi[:, a, j*b:(j+1)*b]
            for m in range(b):
                col = torch.zeros(N, dd, dtype=torch.float64)
                for r in range(b):
                    col[:, j*b+r] = pj[:, (r-m) % b]
                cols.append(col)
    Xd = torch.stack(cols, -1)
    Xd = Xd - Xd.mean(0)                                   # free per-coordinate bias
    Gm = torch.einsum('ndp,ndq->pq', Xd, Xd) / N
    rv = torch.einsum('ndp,nd->p', Xd, x0) / N
    w = torch.linalg.solve(Gm, rv)
    L_brute = float(((x0 - torch.einsum('ndp,p->nd', Xd, w)) ** 2).sum(1).mean())

    Fb = torch.fft.fft(torch.eye(b, dtype=torch.complex128), dim=0) / np.sqrt(b)
    red = 0.0
    for j in range(M):
        PH = torch.einsum('fr,ncr->ncf', Fb, phi[:, :, j*b:(j+1)*b].to(torch.complex128))
        XH = torch.einsum('fr,nr->nf', Fb, x0[:, j*b:(j+1)*b].to(torch.complex128))
        PH, XH = PH - PH.mean(0), XH - XH.mean(0)
        for f in range(b):
            Pm = torch.einsum('na,nb->ab', PH[:, :, f], PH[:, :, f].conj()) / N
            q = torch.einsum('n,na->a', XH[:, f].conj(), PH[:, :, f]) / N
            red += float((q.conj() @ torch.linalg.solve(Pm, q)).real)
    L_form = float((x0 ** 2).sum(1).mean()) - red

    print(f"SELFTEST  d={dd} b={b} c={c} N={N} sigma={sig}")
    print(f"  max |cross-chunk FEATURE covariance|           = {fx:.6f}   (nonzero -- real)")
    print(f"  max |cross-chunk NOISE covariance|, analytic   = {xch:.3e}   (exactly 0)")
    print(f"  brute-force constrained optimum                = {L_brute:.14f}")
    print(f"  formula Tr(Sig) - sum_j,f q^H P^-1 q           = {L_form:.14f}")
    print(f"  abs diff                                       = {abs(L_brute-L_form):.3e}")
    assert abs(L_brute - L_form) < 1e-9, "block-diagonal loss formula FAILED"
    print("  OK: the nonzero cross-chunk feature moments do not enter the optimum.")


def main():
    if os.environ.get('SELFTEST'):
        selftest(); return
    X, Xc, Sig, TR, lin = setup()
    store = {'sigmas': np.array(SIGS)}
    sg = lambda f: [f(s) for s in SIGS]
    hdr = lambda: "".join(f"{f'sg={s}':>20}" for s in SIGS)

    print(f"CIFAR-10 raw pixels  d={d}  N={X.shape[0]}  Tr(Sigma)={TR:.3f}")
    print(f"\nfree linear (Wiener):        " + "".join(f"{lin[s]:>20.3f}" for s in SIGS))
    print(f"block-circulant RF, matched: "
          + "".join(f"{CIRC_RF.get(s, float('nan')):>20.3f}" for s in SIGS))
    store['linear'] = np.array([lin[s] for s in SIGS])
    store['circ_rf'] = np.array([CIRC_RF.get(s, np.nan) for s in SIGS])

    print(f"\n{'='*96}\n1+2. TILED b-blocks vs SLIDING width-w windows  (L, and excess over Wiener)"
          f"\n{'='*96}\n{'w = b':>7} |" + hdr())
    for b in (8, 16, 32, 64, 128):
        t_ = sg(lambda s: tiled(Sig, TR, b, s)); sl = sg(lambda s: sliding(Sig, TR, b, s))
        store[f'tiled|{b}'] = np.array(t_); store[f'sliding|{b}'] = np.array(sl)
        print(f"{b:>7} |" + "".join(f"{a:>10.3f}{a-lin[s]:>+7.3f} /{c-a:>+5.2f}"
                                    for s, a, c in zip(SIGS, t_, sl)))
    print("  (third number = sliding - tiled: overlap's contribution alone)")

    if not os.environ.get('NO_BAYES'):
        print(f"\n{'='*96}\n1b. BAYES floor of the tiled model, b=8 -- the best ANY nonlinear "
              f"tiled denoiser can do\n{'='*96}")
        print(f"{'sigma':>7} {'Wiener':>9} {'tiled lin':>10} {'tiled BAYES':>12} "
              f"{'BAYES-Wiener':>13} {'med N_eff':>10}")
        bay, nef = [], []
        for s in SIGS:
            tb, ne = tiled_bayes(X, 8, s, NT); bay.append(tb); nef.append(ne)
            print(f"{s:>7} {lin[s]:>9.3f} {tiled(Sig,TR,8,s):>10.3f} {tb:>12.3f} "
                  f"{tb-lin[s]:>+13.3f} {ne:>10.1f}")
        store['tiled_bayes|8'] = np.array(bay); store['tiled_bayes_neff|8'] = np.array(nef)
        print("  BAYES-Wiener > 0 => a PERFECT tiled nonlinear denoiser still loses to linear.")

    print(f"\n{'='*96}\n3. True 2-D receptive fields\n{'='*96}\n"
          f"{'patch':>8} {'chans':>6} {'inputs':>7} |" + hdr())
    for ks in (3, 5, 7, 9):
        for allch in (False, True):
            r = [conv2d(Sig, TR, ks, allch, s) for s in SIGS]
            w = r[0][0]; L = [x[1] for x in r]
            store[f'conv2d|{ks}|{int(allch)}'] = np.array(L)
            print(f"{f'{ks}x{ks}':>8} {3 if allch else 1:>6} {w:>7} |"
                  + "".join(f"{a:>12.3f}{a-lin[s]:>+8.3f}" for s, a in zip(SIGS, L)))

    print(f"\n{'='*96}\n4. THE READOUT TOLL: relaxing equivariance to period P"
          f"\n{'='*96}\n{'P':>6} {'W params':>12} |" + hdr())
    for P in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3072):
        L = sg(lambda s: period1d(Xc, TR, P, s)); store[f'period1d|{P}'] = np.array(L)
        print(f"{P:>6} {d*P:>12,} |" + "".join(f"{a:>12.3f}{a-lin[s]:>+8.3f}"
                                               for s, a in zip(SIGS, L)))
    print(f"\n  2-D spatial period p (free 3-channel mixing); p=1 is a real CNN layer:")
    print(f"{'p':>6} {'W params':>12} |" + hdr())
    for p in (1, 2, 4, 8, 16, 32):
        L = sg(lambda s: period2d(Xc, TR, p, s)); store[f'period2d|{p}'] = np.array(L)
        print(f"{p:>6} {3*d*p*p:>12,} |" + "".join(f"{a:>12.3f}{a-lin[s]:>+8.3f}"
                                                   for s, a in zip(SIGS, L)))

    print(f"\n{'='*96}\nVERDICT\n{'='*96}")
    print(f"{'sigma':>7} {'circ RF':>9} {'equiv LINEAR':>13} {'toll':>8} "
          f"{'RF - equivLin':>14}   reading")
    for i, s in enumerate(SIGS):
        el = store['period1d|1'][i]; rf = CIRC_RF.get(s, np.nan)
        print(f"{s:>7} {rf:>9.3f} {el:>13.3f} {el-lin[s]:>+8.3f} {rf-el:>+14.3f}   "
              f"{'nonlinearity buys ' + f'{el-rf:.2f}' + ' < toll ' + f'{el-lin[s]:.2f}'}")
    os.makedirs('tables', exist_ok=True)
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
