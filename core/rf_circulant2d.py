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

HELD-OUT EVALUATION (`x0_test=`)
--------------------------------
Every loss above is an IN-SAMPLE one: the per-frequency readout w_f = P_f^{-1} q_f is solved
on the same moments it is scored against, and with c|G| features against N images that is
exactly the setting where a model can look good by memorising (it is what the dense RF turned
out to be doing on 2026-09-21).  Passing `x0_test` keeps w_f instead of discarding it and
scores it against moments built from a disjoint split:

    L_test = Tr(Sigma_test) + sum_f wgt_f [ -2 Re <q_f^test, w_f> + Tr(w_f^H P_f^test w_f) ]

with w_f = (P_f^train + lam I)^{-1} q_f^train.  The ridge appears only in the solve, never in
the scoring.

CENTRING IS PART OF THE MODEL, so the test moments are centred by the TRAIN means:
vhat^test uses mu_train and uhat^test uses gmean_train.  The quantity returned is therefore

    E_test || x0 - W phi(y) - beta_train ||^2,   beta_train = mu_train - W gmean_train,

i.e. the free per-position bias is held out too.  Re-centring on the test split would be
refitting 3072 of the model's parameters at evaluation time.

The test-side noise terms (the lag Grams S^n and the exact-diagonal correction) are rebuilt
on the test samples, since E_test[Cov(phi|x0)] is a test-set expectation like everything
else.  Only h, w_f and the two train means cross the split boundary.

TWO IN-SAMPLE NUMBERS ARE RETURNED and they are not the same thing:

  'train'       = Tr(Sigma) - sum_f wgt_f Tr(q_f^H (P_f + lam I)^{-1} q_f)
  'train_resid' = Tr(Sigma) + sum_f wgt_f [ -2 Re <q_f, w_f> + Tr(w_f^H P_f w_f) ]

They differ by exactly lam * ||W||^2, the ridge shift.  'train' is the quantity every table
in this project already holds, so it is what the c-sweep must be differenced against;
'train_resid' is the achieved in-sample residual of the SAME w_f that is scored on the test
split, so 'test' - 'train_resid' is the apples-to-apples generalisation gap.  At lam = 1e-6
the two agree to ~1e-4 on toy data, but do not assume that at CIFAR scale: the shift is
proportional to ||W||^2, which grows with c.
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
                        super_chunk=2048, verbose=False, x0_test=None):
    """L^circ on G = Z_H x Z_W with free Cin-channel mixing.

    x0      : (N, Cin, H, W) real images (NOT flattened).
    h       : (c, Cin, H, W) filters, spatially supported on [0,t) x [0,t); one feature PLANE
              per a, so the feature count is c*H*W and the readout has Cin*c*H*W parameters.
    x0_test : optional (Nt, Cin, H, W) held-out split.  If given, the per-frequency readout
              solved on the train moments is scored against the test moments and a dict
              {'train': ..., 'test': ...} is returned instead of a float.  See the module
              docstring for what is and is not held out.
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

    mu = x0.mean(0)                 # TRAIN mean: centres both splits (see docstring)

    Hc = torch.fft.rfft2(h, dim=(-2, -1)).reshape(c, Cin, F)
    HpT = Hc.conj().permute(2, 1, 0).contiguous()                      # (F, Cin, c)
    del Hc
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

    def prep(xs):
        """Per-split constants: the rfft2 of the raw images (for the features), the rfft2 of
        the TRAIN-centred images (for q), and Tr(Sigma) about the train mean."""
        xs = xs.to(device=device, dtype=dtype)
        ns = xs.shape[0]
        Xr = torch.fft.rfft2(xs, dim=(-2, -1)).reshape(ns, Cin, F)
        vv = torch.fft.rfft2(xs - mu, dim=(-2, -1)).reshape(ns, Cin, F) / sq
        tr = float(((xs - mu) ** 2).sum() / max(ns, 1))
        return {'Xr': Xr, 'vh': vv, 'N': ns, 'trace': tr}

    def feats(sp, n0, n1):
        """M[a, n, u] = sum_{ch,v} h_a[ch,v] x[n,ch,u+v], plus the pointwise Stein pieces."""
        pr = torch.matmul(sp['Xr'][n0:n1].permute(2, 0, 1), HpT)       # (F, nb, c)
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

    coef = (1.0, 2.0, 6.0)

    def pass1(sp):
        """Feature mean, exact-diagonal correction, and the lag tensor B for one split."""
        ns = sp['N']
        gmean = torch.zeros(c, D, dtype=dtype, device=device)
        R = [torch.zeros(c, c, nl, dtype=dtype, device=device) for _ in range(3)]
        diag_corr = torch.zeros(c, dtype=dtype, device=device)
        for n0 in range(0, ns, nb):
            n1 = min(n0 + nb, ns)
            M, Phi, ph, G, sa = feats(sp, n0, n1)
            gmean += G.sum(1)
            C = [sa * Phi, sa * ph / 2.0, -M * ph / 6.0]
            de = ((M ** 2 + sa ** 2) * Phi + M * sa * ph - G ** 2).sum(1)
            fo = (C[0] ** 2).sum(1) + 2.0 * (C[1] ** 2).sum(1) + 6.0 * (C[2] ** 2).sum(1)
            # a diagonal correction diag(delta_{a,u}) contributes (1/|G|) sum_u delta[a,u]
            # to EVERY P_f, so only its mean over u is ever needed
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
        gmean /= ns
        diag_corr /= ns
        # Real lag tensor with the Hermite coefficients folded in and the negative lags
        # materialised as transposes, contiguous in the lag axis so the whole per-frequency
        # noise assembly is two (c^2, 3nL) x (3nL, nf) matmuls.
        B = torch.zeros(c, c, 3 * nL, dtype=dtype, device=device)
        for i in range(3):
            R[i] /= ns
            for li, (m1, m2) in enumerate(reps):
                Am = (psi[:, :, li] ** (i + 1)) * R[i][:, :, li]
                B[:, :, i * nL + pos[(m1, m2)]] = coef[i] * Am
                if (m1, m2) != (0, 0):
                    B[:, :, i * nL + pos[(-m1, -m2)]] = coef[i] * Am.T
                del Am
            R[i] = None
        del R
        return gmean, B, torch.diag_embed(diag_corr.to(cdt))

    tr_sp = prep(x0)
    gmean, B_tr, dgc_tr = pass1(tr_sp)
    te_sp = B_te = dgc_te = None
    if x0_test is not None:
        te_sp = prep(x0_test)
        _, B_te, dgc_te = pass1(te_sp)          # gmean_test discarded: centring is train's

    lm1 = torch.tensor([m[0] for m in full] * 3, device=device, dtype=dtype)
    lm2 = torch.tensor([m[1] for m in full] * 3, device=device, dtype=dtype)
    eye = torch.eye(c, dtype=cdt, device=device)

    f1i = (torch.arange(F, device=device, dtype=dtype) // Wh)
    f2i = (torch.arange(F, device=device, dtype=dtype) % Wh)
    selfconj = (f2i == 0) | ((Wd % 2 == 0) & (f2i == Wd // 2))
    wgt_all = torch.where(selfconj, 1.0, 2.0).to(dtype)

    NS = max(nb, int(super_chunk))

    def moments(sp, B, dgc, fa, fb):
        """P_f and q_f for one split over the frequency block [fa, fb), noise included."""
        ns = sp['N']
        nf = fb - fa
        P = torch.zeros(nf, c, c, dtype=cdt, device=device)
        q = torch.zeros(nf, c, Cin, dtype=cdt, device=device)
        for ns0 in range(0, ns, NS):
            ns1 = min(ns0 + NS, ns)
            buf = torch.zeros(nf, ns1 - ns0, c, dtype=cdt, device=device)
            for n0 in range(ns0, ns1, nb):
                n1 = min(n0 + nb, ns1)
                M, Phi, ph, G, sa = feats(sp, n0, n1)
                Gc = (G - gmean.unsqueeze(1)).reshape(c, n1 - n0, H, Wd)
                U = torch.fft.rfft2(Gc, dim=(-2, -1)).reshape(c, n1 - n0, F)[:, :, fa:fb] / sq
                buf[:, n0 - ns0:n1 - ns0, :] = U.permute(2, 1, 0)
                del M, Phi, ph, G, Gc, U
            P += buf.transpose(1, 2) @ buf.conj()
            q += buf.transpose(1, 2) @ sp['vh'][ns0:ns1, :, fa:fb].conj().permute(2, 0, 1)
            del buf
        P /= ns
        q /= ns
        ang = (-2.0 * np.pi) * (torch.outer(f1i[fa:fb], lm1) / H
                                + torch.outer(f2i[fa:fb], lm2) / Wd)     # (nf, 3nL)
        Pv = torch.view_as_real(P)
        for part, trig in ((0, torch.cos), (1, torch.sin)):
            Pv[..., part] += torch.einsum('fm,abm->fab', trig(ang), B) / D
        del Pv, ang
        P += dgc.unsqueeze(0)
        return P, q

    expl = 0.0
    dev_train = 0.0
    dev_test = 0.0
    fc = max(1, int(freq_chunk))
    for fa in range(0, F, fc):
        fb = min(fa + fc, F)
        wg = wgt_all[fa:fb]
        P, q = moments(tr_sp, B_tr, dgc_tr, fa, fb)
        w = torch.linalg.solve(P + lam * eye.unsqueeze(0), q)            # (nf, c, Cin)
        cross = torch.real(torch.sum(q.conj() * w, dim=(1, 2)))
        expl += float((cross * wg).sum())
        if te_sp is not None:
            # achieved in-sample residual of this same w, against the UNRIDGED P
            quad = torch.real(torch.sum(w.conj() * torch.matmul(P, w), dim=(1, 2)))
            dev_train += float(((quad - 2.0 * cross) * wg).sum())
            del quad
            Pt, qt = moments(te_sp, B_te, dgc_te, fa, fb)
            quad = torch.real(torch.sum(w.conj() * torch.matmul(Pt, w), dim=(1, 2)))
            crt = torch.real(torch.sum(qt.conj() * w, dim=(1, 2)))
            dev_test += float(((quad - 2.0 * crt) * wg).sum())
            del Pt, qt, quad, crt
        del P, q, w, cross
        if verbose:
            print(f"    f {fb}/{F}", flush=True)
        if device != 'cpu':
            torch.cuda.empty_cache()
    train = max(0.0, tr_sp['trace'] - expl)
    if te_sp is None:
        return train
    return {'train': train,
            'train_resid': tr_sp['trace'] + dev_train,
            'test': te_sp['trace'] + dev_test,
            'trace_train': tr_sp['trace'], 'trace_test': te_sp['trace']}


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


def _kk_moments(xf, Th, sigma, mu, gm):
    """K x K feature second moment about `gm` and its cross moment with x0 about `mu`."""
    N = xf.shape[0]
    M = xf @ Th.T
    nr = torch.linalg.norm(Th, dim=1)
    rho = (Th @ Th.T) / torch.outer(nr, nr)
    sa = sigma * nr
    z = M / sa
    Phi = _ndtr(z)
    ph = _npdf(z)
    G = M * Phi + sa * ph
    C = [sa * Phi, sa * ph / 2.0, -M * ph / 6.0]
    de = ((M ** 2 + sa ** 2) * Phi + M * sa * ph - G ** 2).mean(0)
    fo = (C[0] ** 2).mean(0) + 2.0 * (C[1] ** 2).mean(0) + 6.0 * (C[2] ** 2).mean(0)
    if gm is None:
        gm = G.mean(0)
    Gc = G - gm
    Sig = (Gc.T @ Gc) / N
    coef = (1.0, 2.0, 6.0)
    for i in range(3):
        Sig = Sig + coef[i] * (rho ** (i + 1)) * ((C[i].T @ C[i]) / N)
    Sig = Sig + torch.diag(de - fo)
    Xc = xf - mu
    return Sig, (Xc.T @ Gc) / N, float((Xc ** 2).sum() / N), gm


def circulant2d_rf_mmse_bruteforce(x0, h, sigma, lam=1e-6, device='cpu',
                                   dtype=torch.float64, x0_test=None):
    """Same loss, computed by materialising the K x K covariance and solving the
    constrained least squares over an explicit real parameterization of the readout.

    Only usable at toy sizes (K = c*H*W and Cin*c*H*W parameters).
    """
    x0 = x0.to(device=device, dtype=dtype)
    h = h.to(device=device, dtype=dtype)
    N, Cin, H, Wd = x0.shape
    c = h.shape[0]
    d, K = Cin * H * Wd, c * H * Wd

    Th = _explicit_theta(h, H, Wd).to(device=device, dtype=dtype)       # (K, d)
    Xf = x0.reshape(N, d)
    mu = Xf.mean(0)
    Sig, Sxp, trace_p0, gm = _kk_moments(Xf, Th, sigma, mu, None)

    E = _bccb_basis(Cin, c, H, Wd, device, dtype)
    A = torch.einsum('pri,ij,qrj->pq', E, Sig, E)
    b = torch.einsum('pri,ri->p', E, Sxp)
    # <E_p, E_q>_F = D delta_pq (the E_p are disjointly supported indicators with D ones
    # each), so this is the tap-space image of the per-frequency lam*I the structured path
    # adds -- the |G|^{-1/2} in uhat is what makes the two coincide.
    reg = lam * torch.einsum('pri,qri->pq', E, E)
    w = torch.linalg.solve(A + reg, b)
    train = float(trace_p0 - float(b @ w))              # ridge-shifted, project convention
    if x0_test is None:
        return train
    xt = x0_test.to(device=device, dtype=dtype).reshape(-1, d)
    St, Sxt, tr_t, _ = _kk_moments(xt, Th, sigma, mu, gm)
    At = torch.einsum('pri,ij,qrj->pq', E, St, E)
    bt = torch.einsum('pri,ri->p', E, Sxt)
    return {'train': train,
            'train_resid': float(trace_p0 - 2.0 * float(b @ w) + float(w @ (A @ w))),
            'test': float(tr_t - 2.0 * float(bt @ w) + float(w @ (At @ w)))}


def selftest2d(seed=0, verbose=True, device=None):
    """Structured path vs the brute-force constrained least squares, on non-Gaussian,
    non-stationary, cross-channel-correlated toy data.  Both the in-sample loss and the
    held-out loss are checked, the latter on a test split drawn from a SHIFTED, RESCALED
    distribution so that train and test moments genuinely differ."""
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

        def draw(n, shift, scale):
            z = rng.standard_normal((n, d))
            z = z + 0.4 * rng.standard_normal((n, 1))      # cross-coordinate correlation
            raw = (z ** 3) / 3.0 @ A.T * scale             # non-Gaussian marginals
            raw = raw + np.linspace(-1.0, 1.0, d)[None, :] + shift   # non-stationary mean
            return torch.tensor(raw.reshape(n, Cin, H, Wd), dtype=torch.float64, device=dev)

        x0 = draw(N, 0.0, 1.0)
        xt = draw(N // 2 + 37, 0.25, 1.15)

        hh = torch.zeros(c, Cin, H, Wd, dtype=torch.float64, device=dev)
        hh[:, :, :t, :t] = torch.tensor(
            rng.standard_normal((c, Cin, t, t)) / np.sqrt(Cin * t * t),
            dtype=torch.float64, device=dev)

        new = circulant2d_rf_mmse(x0, hh, sig, t, lam=1e-6, device=dev, sample_chunk=7,
                                  freq_chunk=3, super_chunk=13, x0_test=xt)
        ref = circulant2d_rf_mmse_bruteforce(x0, hh, sig, lam=1e-6, device=dev, x0_test=xt)
        rel = {k: abs(new[k] - ref[k]) / max(abs(ref[k]), 1e-12)
               for k in ('train', 'train_resid', 'test')}
        good = max(rel.values()) < 1e-9
        ok &= good
        if verbose:
            print(f"  {H}x{Wd} Cin={Cin} c={c} t={t} sigma={sig:<4}: "
                  f"train {new['train']:.8f} ({rel['train']:.1e})  "
                  f"resid ({rel['train_resid']:.1e})  "
                  f"test {new['test']:.8f} ({rel['test']:.1e})  "
                  f"{'OK' if good else 'FAIL'}", flush=True)

    # identity check: with lam = 0 and test == train the held-out loss must collapse onto
    # the in-sample one, which pins the sign and normalisation of the -2Re<q,w> + w^H P w
    # assembly independently of the brute force.
    rng = np.random.default_rng(seed + 5)
    H = Wd = 6; Cin = 2; c = 3; t = 2; N = 400
    x0 = torch.tensor(rng.standard_normal((N, Cin, H, Wd)) * 1.3 + 0.2,
                      dtype=torch.float64, device=dev)
    hh = torch.zeros(c, Cin, H, Wd, dtype=torch.float64, device=dev)
    hh[:, :, :t, :t] = torch.tensor(rng.standard_normal((c, Cin, t, t)) / np.sqrt(Cin * t * t),
                                    dtype=torch.float64, device=dev)
    r = circulant2d_rf_mmse(x0, hh, 0.8, t, lam=0.0, device=dev, sample_chunk=11,
                            freq_chunk=4, super_chunk=23, x0_test=x0)
    rel = abs(r['test'] - r['train']) / max(abs(r['train']), 1e-12)
    ok &= rel < 1e-9
    if verbose:
        print(f"  test==train, lam=0: train {r['train']:.10f}  test {r['test']:.10f}  "
              f"rel={rel:.2e} {'OK' if rel < 1e-9 else 'FAIL'}", flush=True)
    return ok


if __name__ == '__main__':
    print("selftest2d:", "PASS" if selftest2d() else "FAIL")
