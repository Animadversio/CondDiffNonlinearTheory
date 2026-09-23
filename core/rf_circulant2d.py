"""L^circ for a Z_H x Z_W equivariant random-feature denoiser on multichannel images.

WHY THIS FILE EXISTS
--------------------
`core.rf_circulant_struct` computes L^circ for the group Z_d acting on the RASTER-FLATTENED
image, i.e. it treats a 3 x 32 x 32 CIFAR image as a ring of 3072 scalars.  That is the
wrong symmetry for a CNN: a shift by one in Z_3072 moves a pixel one step along the raster
and wraps colour planes into each other, and the "local" 8-tap filter reads 8 horizontally
adjacent pixels of ONE colour plane with no vertical or cross-channel extent at all.

The honest CNN group is G = Z_32 x Z_32 acting on (3, 32, 32) with the 3 channels FREE
(mixed by a dense 3 x 3 matrix at every frequency, not acted on by the group).  Every finite
abelian group has a DFT, so the whole machinery carries over verbatim with `rfft` replaced by
`rfft2` and the frequency index f in Z_d replaced by f = (f1, f2) in Z_32 x Z_32.

WHAT THE MODEL IS
-----------------
    x0   in R^{Cin x H x W},   y = x0 + sigma Z
    phi_a[u] = relu( sum_{ch, v} h_a[ch, v] y[ch, u + v] ),   a = 1..c,  u in G
               h_a spatially supported on [0,t) x [0,t) in each of the Cin channels
               => t^2 * Cin random numbers per feature plane (27 at t=3, Cin=3)
    D(y)[ch, u] = sum_a sum_v W[ch, a, v] phi_a[u + v]  +  beta[ch, u],   beta free

so the readout is block-circulant with circulant blocks (BCCB) on G with FREE Cin-channel
mixing: Cin * c * |G| trained parameters.  At Cin=3, H=W=32 that is 3072 * c -- EXACTLY the
c*d of the Z_3072 model and the k*d of the dense model at k = c, so the free-parameter
matching used throughout `rf_pixel_featmatch2.py` is preserved and `c` maps onto the same
axis.  The feature count is c * |G| = 1024c (vs 3072c in 1-D): fewer features, same
parameters, a better group.

THE ESTIMATOR
-------------
Identical in form to the 1-D one:

    L = Tr(Sigma_p0) - sum_{f in Ghat} Tr( q_f^H P_f^{-1} q_f ),
    P_f (c x c)  = E[ uhat[f] uhat[f]^H ],    q_f (c x Cin) = E[ uhat[f] vhat[f]^H ],

with uhat_a[f] = |G|^{-1/2} sum_u (phi_a[u] - E phi_a[u]) e^{-2 pi i <f,u>},
     vhat_ch[f] = |G|^{-1/2} sum_u (x0[ch,u] - E x0[ch,u]) e^{-2 pi i <f,u>}.

Only the SAME-frequency blocks appear.  That is a property of the MODEL (a BCCB readout
cannot couple frequency f to g), not of p(x0): the cross-frequency moments are nonzero on
CIFAR, they simply never enter.  Nothing here assumes stationarity.

Sigma_phi uses the same Stein/Mehler expansion as the 1-D path, so the two are comparable
cell for cell:  Cov(phi) = Cov_x0(G) + E_x0[Cov(phi | x0)], with
Cov(phi|x0)_{ij} ~= sum_{n=1..3} coef_n rho_ij^n C_{n,i} C_{n,j} and the diagonal replaced by
the exact conditional variance.

THE ONE STEP WORTH VALIDATING is the noise assembly.  rho is the row-correlation matrix,
rho_{(a,u),(b,u')} = psi_ab(u - u'), a G-convolution because both rows are translates of the
same pair of filters.  Its n-th Hadamard power times a Gram M gives, at frequency f,

    (1/|G|) sum_{u, u'} psi_ab(u-u')^n M_{(a,u),(b,u')} e^{-2 pi i <f, u-u'>}
  = (1/|G|) sum_m psi_ab(m)^n e^{-2 pi i <f, m>} S^n_ab(m),
    S^n_ab(m) := (1/N) sum_{samples} sum_u C_n[a, u+m] C_n[b, u],

i.e. a plain lag sum, NOT a cyclic correlation over frequencies -- exactly the restructuring
`circulant_rf_mmse_lag2` uses in 1-D.  psi is supported on the (2t-1)^2 lags m in
[-(t-1), t-1]^2 (25 at t=3, against 15 for the 1-D t=8 model), and lag symmetry
S_ab(-m) = S_ba(m), psi_ab(-m) = psi_ba(m) halves the Grams that must be formed.

`selftest2d` checks the whole thing against a brute-force reference that materialises the
K x K feature covariance (K = c|G|) and solves the constrained least squares over an
EXPLICIT REAL parameterization of the BCCB readout -- so it validates the frequency
decoupling, the lag assembly and the Hermitian fold together, not just the algebra.

HERMITIAN FOLD.  phi and x0 are real, so P_{-f} = conj(P_f) and q_{-f} = conj(q_f) and the
two frequencies contribute the same real part.  `rfft2` returns f2 = 0..W/2; within that
half-plane the pairs (f1, f2) and (-f1, f2) are conjugate when f2 in {0, W/2}, so those
columns take weight 1 and every other column weight 2.  (32*2)*1 + (32*15)*2 = 1024.  At the
four self-conjugate frequencies f = -f the data uhat[f], vhat[f] are real, P and q are real,
and the unconstrained complex solve is automatically the real one -- no special case needed.
"""

import numpy as np
import torch

from core.rf_gmm_estimators_torch import _ndtr, _npdf


def lag_reps(t):
    """One representative of each {+m, -m} pair among the lags where psi can be nonzero.

    psi_ab(m) = sum_{ch,v} h_a[ch,v] h_b[ch,v+m] is nonzero only if the supports overlap
    after the shift, i.e. m in [-(t-1), t-1]^2.  Since psi_ab(-m) = psi_ba(m) and
    S_ab(-m) = S_ba(m), only the half m1 > 0 or (m1 == 0 and m2 >= 0) has to be formed.
    """
    return [(m1, m2)
            for m1 in range(-(t - 1), t)
            for m2 in range(-(t - 1), t)
            if (m1 > 0) or (m1 == 0 and m2 >= 0)]


def lag_full(t):
    return [(m1, m2) for m1 in range(-(t - 1), t) for m2 in range(-(t - 1), t)]


def circulant2d_rf_mmse(x0, h, sigma, t_band, lam=1e-6, device='cuda',
                        dtype=torch.float64, sample_chunk=None, freq_chunk=32,
                        super_chunk=2048, verbose=False):
    """L^circ on G = Z_H x Z_W with free Cin-channel mixing.

    x0 : (N, Cin, H, W) real images (NOT flattened).
    h  : (c, Cin, H, W) filters, spatially supported on [0,t) x [0,t); one feature PLANE per
         a, so the feature count is c * H * W and the readout has Cin * c * H * W parameters.
    sigma, lam, chunking : as in `circulant_rf_mmse_lag2`.
    """
    x0 = x0.to(device=device, dtype=dtype)
    h = h.to(device=device, dtype=dtype)
    N, Cin, H, Wd = x0.shape
    c = h.shape[0]
    if h.shape[1:] != (Cin, H, Wd):
        raise ValueError(f"h must be (c, {Cin}, {H}, {Wd}), got {tuple(h.shape)}")
    t = int(t_band)
    if 2 * t - 1 > min(H, Wd):
        raise ValueError("lag set wraps: need 2t-1 <= min(H, W)")
    D = H * Wd                      # |G|
    Wh = Wd // 2 + 1
    F = H * Wh                      # frequencies kept by rfft2
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    sq = D ** 0.5

    if sample_chunk is None:        # keep the (c, nb, D) fp64 feature tensor near 2 GB
        sample_chunk = max(1, min(N, int(2.0e9 / (c * D * 8))))
    nb = int(sample_chunk)

    mu = x0.mean(0)
    X0c = x0 - mu
    trace_p0 = float((X0c ** 2).sum() / max(N, 1))

    Xr = torch.fft.rfft2(x0, dim=(-2, -1)).reshape(N, Cin, F)          # (N, Cin, F)
    Hc = torch.fft.rfft2(h, dim=(-2, -1)).reshape(c, Cin, F)
    HpT = Hc.conj().permute(2, 1, 0).contiguous()                      # (F, Cin, c)
    vh = torch.fft.rfft2(X0c, dim=(-2, -1)).reshape(N, Cin, F) / sq
    del Hc, X0c

    nrm = torch.linalg.norm(h.reshape(c, -1), dim=1)
    s = sigma * nrm

    reps = lag_reps(t)
    full = lag_full(t)
    pos = {m: i for i, m in enumerate(full)}
    nl, nL = len(reps), len(full)

    psi = torch.stack([torch.einsum('acij,bcij->ab', h,
                                    torch.roll(h, shifts=(-m1, -m2), dims=(2, 3)))
                       for (m1, m2) in reps], dim=-1)                  # (c, c, nl)
    psi /= (nrm.view(-1, 1, 1) * nrm.view(1, -1, 1))

    def feats(n0, n1):
        """M[a, n, u] = sum_{ch,v} h_a[ch,v] x[n,ch,u+v], plus the pointwise Stein pieces."""
        pr = torch.matmul(Xr[n0:n1].permute(2, 0, 1), HpT)             # (F, nb, c)
        Mf = pr.permute(2, 1, 0).reshape(c, n1 - n0, H, Wh)
        del pr
        M = torch.fft.irfft2(Mf, s=(H, Wd), dim=(-2, -1)).reshape(c, n1 - n0, D)
        del Mf
        sa = s.view(-1, 1, 1)
        z = M / sa
        Phi = _ndtr(z)
        ph = _npdf(z)
        G = M * Phi + sa * ph
        return M, Phi, ph, G, sa

    # ---- pass 1: feature mean, exact-diagonal correction, the lag Grams -----------------
    gmean = torch.zeros(c, D, dtype=dtype, device=device)
    R = [torch.zeros(c, c, nl, dtype=dtype, device=device) for _ in range(3)]
    diag_corr = torch.zeros(c, dtype=dtype, device=device)
    for n0 in range(0, N, nb):
        n1 = min(n0 + nb, N)
        M, Phi, ph, G, sa = feats(n0, n1)
        gmean += G.sum(1)
        C = [sa * Phi, sa * ph / 2.0, -M * ph / 6.0]
        de = ((M ** 2 + sa ** 2) * Phi + M * sa * ph - G ** 2).sum(1)
        fo = (C[0] ** 2).sum(1) + 2.0 * (C[1] ** 2).sum(1) + 6.0 * (C[2] ** 2).sum(1)
        # a diagonal correction diag(delta_{a,u}) contributes (1/|G|) sum_u delta[a,u] to
        # EVERY P_f, so only its mean over u is ever needed
        diag_corr += (de - fo).sum(1) / D
        del M, Phi, ph, G, de, fo
        for i in range(3):
            Ci = C[i].reshape(c, n1 - n0, H, Wd)
            flat = C[i].reshape(c, -1)
            for li, (m1, m2) in enumerate(reps):
                R[i][:, :, li].addmm_(
                    torch.roll(Ci, shifts=(-m1, -m2), dims=(2, 3)).reshape(c, -1), flat.T)
            del Ci, flat
        del C
    gmean /= N
    for i in range(3):
        R[i] /= N
    diag_corr /= N

    # Real lag tensor with the Hermite coefficients folded in and the negative lags
    # materialised as transposes, contiguous in the lag axis so the whole per-frequency
    # noise assembly is two (c^2, 3nL) x (3nL, nf) matmuls.
    coef = (1.0, 2.0, 6.0)
    B = torch.zeros(c, c, 3 * nL, dtype=dtype, device=device)
    for i in range(3):
        for li, (m1, m2) in enumerate(reps):
            Am = (psi[:, :, li] ** (i + 1)) * R[i][:, :, li]
            B[:, :, i * nL + pos[(m1, m2)]] = coef[i] * Am
            if (m1, m2) != (0, 0):
                B[:, :, i * nL + pos[(-m1, -m2)]] = coef[i] * Am.T
            del Am
    del R, psi

    lm1 = torch.tensor([m[0] for m in full] * 3, device=device, dtype=dtype)
    lm2 = torch.tensor([m[1] for m in full] * 3, device=device, dtype=dtype)
    eye = torch.eye(c, dtype=cdt, device=device)
    dgc = torch.diag_embed(diag_corr.to(cdt))

    f1i = (torch.arange(F, device=device, dtype=dtype) // Wh)
    f2i = (torch.arange(F, device=device, dtype=dtype) % Wh)
    selfconj = (f2i == 0) | ((Wd % 2 == 0) & (f2i == Wd // 2))
    wgt_all = torch.where(selfconj, 1.0, 2.0).to(dtype)

    # ---- pass 2: stream frequencies, accumulate P and q over sample super-chunks --------
    expl = 0.0
    fc = max(1, int(freq_chunk))
    NS = max(nb, int(super_chunk))
    for fa in range(0, F, fc):
        fb = min(fa + fc, F)
        nf = fb - fa
        P = torch.zeros(nf, c, c, dtype=cdt, device=device)
        q = torch.zeros(nf, c, Cin, dtype=cdt, device=device)
        for ns0 in range(0, N, NS):
            ns1 = min(ns0 + NS, N)
            buf = torch.zeros(nf, ns1 - ns0, c, dtype=cdt, device=device)
            for n0 in range(ns0, ns1, nb):
                n1 = min(n0 + nb, ns1)
                M, Phi, ph, G, sa = feats(n0, n1)
                Gc = (G - gmean.unsqueeze(1)).reshape(c, n1 - n0, H, Wd)
                U = torch.fft.rfft2(Gc, dim=(-2, -1)).reshape(c, n1 - n0, F)[:, :, fa:fb] / sq
                buf[:, n0 - ns0:n1 - ns0, :] = U.permute(2, 1, 0)
                del M, Phi, ph, G, Gc, U
            P += buf.transpose(1, 2) @ buf.conj()
            q += buf.transpose(1, 2) @ vh[ns0:ns1, :, fa:fb].conj().permute(2, 0, 1)
            del buf
        P /= N
        q /= N
        ang = (-2.0 * np.pi) * (torch.outer(f1i[fa:fb], lm1) / H
                                + torch.outer(f2i[fa:fb], lm2) / Wd)     # (nf, 3nL)
        Pv = torch.view_as_real(P)
        for part, trig in ((0, torch.cos), (1, torch.sin)):
            Pv[..., part] += torch.einsum('fm,abm->fab', trig(ang), B) / D
        del Pv, ang
        P += dgc.unsqueeze(0) + lam * eye.unsqueeze(0)
        sol = torch.linalg.solve(P, q)                                   # (nf, c, Cin)
        term = torch.real(torch.sum(q.conj() * sol, dim=(1, 2)))         # (nf,)
        expl += float((term * wgt_all[fa:fb]).sum())
        del P, q, sol
        if verbose:
            print(f"    f {fb}/{F}", flush=True)
        if device != 'cpu':
            torch.cuda.empty_cache()
    return max(0.0, trace_p0 - expl)


# ---------------------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------------------

def _explicit_theta(h, H, Wd):
    """Rows of the G-equivariant design: theta_{a,u}[ch, w] = h_a[ch, w - u]."""
    c, Cin = h.shape[0], h.shape[1]
    rows = []
    for a in range(c):
        for u1 in range(H):
            for u2 in range(Wd):
                rows.append(torch.roll(h[a], shifts=(u1, u2), dims=(-2, -1)).reshape(-1))
    return torch.stack(rows)                     # (c*H*W, Cin*H*W), row order (a, u1, u2)


def _bccb_basis(Cin, c, H, Wd, device, dtype):
    """E_p in R^{d x K} for p = (ch, a, v): E_p[(ch', u'), (a', u)] = [ch'=ch][a'=a][u'-u=v].

    The span of {E_p} is exactly the set of readouts
    D(y)[ch, u'] = sum_{a, v} W[ch, a, v] phi_a[u' + v]  -- i.e. BCCB blocks with free
    channel mixing -- so a brute-force least squares over p is the constrained optimum.
    """
    D = H * Wd
    d, K = Cin * D, c * D
    E = torch.zeros(Cin * c * D, d, K, device=device, dtype=dtype)
    p = 0
    for ch in range(Cin):
        for a in range(c):
            for v1 in range(H):
                for v2 in range(Wd):
                    for u1 in range(H):
                        for u2 in range(Wd):
                            r = ch * D + u1 * Wd + u2
                            cc = a * D + ((u1 - v1) % H) * Wd + ((u2 - v2) % Wd)
                            E[p, r, cc] = 1.0
                    p += 1
    return E


def circulant2d_rf_mmse_bruteforce(x0, h, sigma, lam=1e-6, device='cpu',
                                   dtype=torch.float64):
    """Same loss, computed by materialising the K x K covariance and solving the
    constrained least squares over an explicit real parameterization of the readout.

    Only usable at toy sizes (K = c*H*W and Cin*c*H*W parameters).
    """
    x0 = x0.to(device=device, dtype=dtype)
    h = h.to(device=device, dtype=dtype)
    N, Cin, H, Wd = x0.shape
    c = h.shape[0]
    D = H * Wd
    d, K = Cin * D, c * D

    Th = _explicit_theta(h, H, Wd).to(device=device, dtype=dtype)       # (K, d)
    Xf = x0.reshape(N, d)
    M = Xf @ Th.T                                                       # (N, K)
    nr = torch.linalg.norm(Th, dim=1)
    rho = (Th @ Th.T) / torch.outer(nr, nr)
    sa = sigma * nr                                                     # (K,)

    z = M / sa
    Phi = _ndtr(z)
    ph = _npdf(z)
    G = M * Phi + sa * ph
    C = [sa * Phi, sa * ph / 2.0, -M * ph / 6.0]
    de = ((M ** 2 + sa ** 2) * Phi + M * sa * ph - G ** 2).mean(0)
    fo = (C[0] ** 2).mean(0) + 2.0 * (C[1] ** 2).mean(0) + 6.0 * (C[2] ** 2).mean(0)

    Gc = G - G.mean(0)
    Sig = (Gc.T @ Gc) / N
    coef = (1.0, 2.0, 6.0)
    for i in range(3):
        Sig = Sig + coef[i] * (rho ** (i + 1)) * ((C[i].T @ C[i]) / N)
    Sig = Sig + torch.diag(de - fo) + lam * torch.eye(K, device=device, dtype=dtype)

    Xc = Xf - Xf.mean(0)
    trace_p0 = float((Xc ** 2).sum() / N)
    Sxp = (Xc.T @ Gc) / N                                               # (d, K)

    E = _bccb_basis(Cin, c, H, Wd, device, dtype)
    A = torch.einsum('pri,ij,qrj->pq', E, Sig, E)
    b = torch.einsum('pri,ri->p', E, Sxp)
    w = torch.linalg.solve(A, b)
    return float(trace_p0 - float(b @ w))


def selftest2d(seed=0, verbose=True, device=None):
    """Structured path vs the brute-force constrained least squares, on non-Gaussian,
    non-stationary, cross-channel-correlated toy data."""
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ok = True
    cases = ((4, 4, 2, 2, 2, 400, 1.1),
             (4, 4, 3, 2, 2, 500, 0.6),
             (6, 6, 2, 3, 2, 400, 1.7),
             (6, 6, 1, 2, 3, 500, 0.9),
             (8, 4, 3, 2, 2, 600, 0.4))
    for (H, Wd, Cin, c, t, N, sig) in cases:
        rng = np.random.default_rng(seed)
        d = Cin * H * Wd
        A = rng.standard_normal((d, d)) * 0.5
        z = rng.standard_normal((N, d))
        z = z + 0.4 * rng.standard_normal((N, 1))          # cross-coordinate correlation
        raw = (z ** 3) / 3.0 @ A.T                         # non-Gaussian marginals
        raw = raw + np.linspace(-1.0, 1.0, d)[None, :]     # non-stationary mean
        x0 = torch.tensor(raw.reshape(N, Cin, H, Wd), dtype=torch.float64, device=dev)

        hh = torch.zeros(c, Cin, H, Wd, dtype=torch.float64, device=dev)
        hh[:, :, :t, :t] = torch.tensor(
            rng.standard_normal((c, Cin, t, t)) / np.sqrt(Cin * t * t),
            dtype=torch.float64, device=dev)

        new = circulant2d_rf_mmse(x0, hh, sig, t, lam=1e-6, device=dev,
                                  sample_chunk=7, freq_chunk=3, super_chunk=13)
        ref = circulant2d_rf_mmse_bruteforce(x0, hh, sig, lam=1e-6, device=dev)
        rel = abs(new - ref) / max(abs(ref), 1e-12)
        ok &= rel < 1e-9
        if verbose:
            print(f"  {H}x{Wd} Cin={Cin} c={c} t={t} sigma={sig:<4}: "
                  f"ref={ref:.12f} struct={new:.12f} rel={rel:.2e} "
                  f"{'OK' if rel < 1e-9 else 'FAIL'}", flush=True)
    return ok


if __name__ == '__main__':
    print("selftest2d:", "PASS" if selftest2d() else "FAIL")
