"""BAND-MODULATED NONLINEAR RF on G = Z_H x Z_W  (michimin, 2026-09-23 06:36).

    "run B=1 experiment for 2D circulant for c=512 for all the noises"

WHAT THIS IS, AND WHY IT DID NOT EXIST BEFORE
---------------------------------------------
Everything done on the band relaxation so far -- `scripts/rf_band_relaxation.py` (michimin's),
`scripts/rf_band_relaxation_atlas.py`, `scripts/rf_band_relaxation_heldout.py` -- is LINEAR:
it measures the best linear denoiser inside the band class, i.e. the class's *toll*.  This
file is the NONLINEAR member of that same class, and it is the decisive test of the one
load-bearing assumption in michimin's docs/rf_band_relaxation.md section 5:

    the nonlinear gain (equivariant-linear minus nonlinear RF, 1.144 in Z_3072 and
    1.930 in Z_32 x Z_32 at sigma=0.127) TRANSFERS to a larger linear class.

The default worry runs the other way: a larger linear class absorbs what relu bought.  The
2-D re-index already gave one data point against that worry (the gain GREW, 1.76x-4.74x);
this run asks the same question of the band, which is the relaxation that actually removes
most of the toll per parameter.

THE MODEL
---------
Take the plain 2-D model of `core.rf_circulant2d` and MODULATE THE FEATURES:

    phi_a[u] = relu( sum_{ch,v} h_a[ch,v] y[ch, u+v] ),       a = 1..c,  u in G
    psi_{a,r}[u] = m_r[u] phi_a[u],   m_r[u] = exp(2 pi i <s_r, u>),  s_r in box(B)
    D(y)[ch,u'] = sum_{a,r,v} W[ch,(a,r),v] psi_{a,r}[u'+v] + beta[ch,u']

with box(B) = {-B..B}^2, |G_band| = (2B+1)^2 = 9 at B=1.  The readout is BCCB over the
c|G_band| modulated planes, so it is exactly michimin's band class
A = sum_s diag(e_s) BCCB(a_s) with a nonlinear feature map underneath.

REALITY.  The output is real iff W[ch,(a,-s),v] = conj(W[ch,(a,s),v]).  Equivalently the
model is a plain BCCB readout over the c|G_band| REAL planes {phi_a, cos(.)phi_a, sin(.)phi_a}
-- which is what the brute force parameterizes.  The complex form is used here because the
per-frequency solve stays UNCONSTRAINED: box(B) is symmetric, so conjugating an optimum and
swapping r <-> -r gives a feasible optimum, and P > 0 makes it unique, hence the complex
solve is already the real one.  (Same argument that retired `_selfconj_block` in the band
relaxation.  The brute force is what actually checks it.)

TRAINED PARAMETERS = Cin * c * |G| * |G_band| = 3072 * c * (2B+1)^2.  At c=512, B=1 that is
14,155,776 -- 9x the plain 2-D model at the same c, and 3x the free Wiener's 4,720,128.
THAT IS THE WHOLE REASON THIS RUNS HELD OUT.  14.2M parameters against 10^4 images is
precisely the regime where the dense RF turned out to be memorising (2026-09-21), and an
in-sample number here would be uninterpretable.

THE ALGEBRA (the only genuinely new part)
-----------------------------------------
psi_hat_{a,r}[f] = phi_hat_a[f - s_r], so the per-frequency block couples SHIFTED source
frequencies:

    P_f[(a,r),(b,r')] = E[ phi_hat_a[g] phi_hat_b[g']^* ],   g = f - s_r,  g' = f - s_r',
    q_f[(a,r),ch]     = E[ phi_hat_a[f - s_r] v_hat_ch[f]^* ].

The SIGNAL part is a gather: the rfft2 of the centred features is formed once and read at the
|G_band| shifted frequencies (conjugated where the shift leaves the rfft2 half-plane).

The NOISE part needs a genuine generalisation of the Stein assembly.  With
Delta := g - g' = s_r' - s_r in box(2B),

    P^noise_f[(a,r),(b,r')] = (1/|G|) sum_n coef_n sum_m psi_ab(m)^n T^n_ab(m,Delta)
                                                          exp(-2 pi i <g', m>),
    T^n_ab(m,Delta) := (1/N) sum_samples sum_u C_n[a,u] C_n[b,u-m] exp(-2 pi i <Delta,u>).

At Delta = 0 this collapses to the plain lag Gram S^n_ab(m) of `core.rf_circulant2d`, which
is the regression the selftest runs at B = 0.  Two symmetries keep the cost at ~26x rather
than ~81x the plain model:

    T_ab(m,-Delta) = conj( T_ab(m,Delta) )            (C is real)  -> 13 of 25 Deltas at B=1
    T_ab(-m,Delta) = exp(2 pi i <Delta,m>) T_ba(m,Delta)           -> half the lags

THE EXACT-DIAGONAL CORRECTION IS ALSO Delta-DEPENDENT, and this is the easiest thing in the
whole file to get wrong.  The plain model only ever needs the MEAN over u of the per-position
correction delta[a,u], because a diagonal matrix contributes (1/|G|) sum_u delta[a,u] to every
P_f.  Here it contributes

    (1/|G|) sum_u delta[a,u] exp(-2 pi i <Delta,u>)

to the (r,r') block, i.e. the Delta-th FOURIER COEFFICIENT of delta[a,.], obtained by an fft2
of the accumulated (c,H,W) tensor.  Delta = 0 recovers the mean.

HERMITIAN FOLD survives verbatim.  psi_hat_{a,r}[-f] = conj( psi_hat_{a,rbar}[f] ) with
s_rbar = -s_r, so P_{-f} is conj(P_f) up to the r <-> rbar permutation and the two frequencies
contribute the same real part.  The rfft2 weights of `core.rf_circulant2d` (weight 1 where
f2 in {0, W/2}, else 2) apply to the OUTPUT frequency f unchanged.

RIDGE.  The structured path adds lam*I per frequency in the COMPLEX modulation basis.  In the
real tap basis that is lam * <E_p, E_p>_F, and <E_p,E_p>_F = sum_u m_j[u]^2 = |G| for the
constant plane but |G|/2 for every cos/sin plane.  The brute force uses that exact weighting,
so the two agree at lam > 0 and the selftest does not have to fall back to lam = 0.

HELD-OUT EVALUATION (`x0_test=`) is identical in contract to `core.rf_circulant2d`: w_f is
solved on train moments and scored against test moments centred by the TRAIN means, and
{'train', 'train_resid', 'test'} are returned.  See that module's docstring for what does and
does not cross the split.
"""

import numpy as np
import torch

from core.rf_gmm_estimators_torch import _ndtr, _npdf
from core.rf_circulant2d import lag_reps, lag_full, _explicit_theta, _bccb_basis, _kk_moments


def band_offsets(B):
    """box(B) = the modulation frequencies s_r, in a NEGATION-SYMMETRIC order.

    The list reads as a raster over {-B..B}^2, so index(-s) = nG - 1 - index(s).  That
    symmetry is what makes the unconstrained complex per-frequency solve equal the real
    optimum, so it is a property of the *set*, not a convenience of the ordering.
    """
    return [(a, b) for a in range(-B, B + 1) for b in range(-B, B + 1)]


def band_half(B):
    """One representative of each {+s,-s} pair in box(B) \\ {0}."""
    return [(a, b) for (a, b) in band_offsets(B) if (a > 0) or (a == 0 and b > 0)]


def delta_reps(B):
    """One representative of each {+Delta,-Delta} pair in box(2B), (0,0) included.

    Delta = s_r' - s_r ranges over box(2B) (25 values at B=1) and T_ab(m,-Delta) is the
    conjugate of T_ab(m,Delta), so only this half (13 values at B=1) has to be formed.
    """
    return [(a, b)
            for a in range(-2 * B, 2 * B + 1)
            for b in range(-2 * B, 2 * B + 1)
            if (a > 0) or (a == 0 and b >= 0)]


def circulant2d_band_rf_mmse(x0, h, sigma, t_band, B, lam=1e-6, device='cuda',
                             dtype=torch.float64, sample_chunk=None, pass1_chunk=None,
                             freq_chunk=8, super_chunk=2048, verbose=False, x0_test=None):
    """L for the band-modulated nonlinear RF on G = Z_H x Z_W with free Cin-channel mixing.

    x0      : (N, Cin, H, W) real images.
    h       : (c, Cin, H, W) filters, spatially supported on [0,t) x [0,t).
    B       : band half-width.  B = 0 reproduces `circulant2d_rf_mmse` exactly.
    x0_test : optional held-out split; returns a dict instead of a float.

    Trained parameters = Cin * c * H * W * (2B+1)^2.
    """
    x0 = x0.to(device=device, dtype=dtype)
    h = h.to(device=device, dtype=dtype)
    N, Cin, H, Wd = x0.shape
    c = h.shape[0]
    if h.shape[1:] != (Cin, H, Wd):
        raise ValueError(f"h must be (c, {Cin}, {H}, {Wd}), got {tuple(h.shape)}")
    t = int(t_band)
    B = int(B)
    if 2 * t - 1 > min(H, Wd):
        raise ValueError("lag set wraps: need 2t-1 <= min(H, W)")
    if 2 * B + 1 > min(H, Wd):
        raise ValueError("band aliases: need 2B+1 <= min(H, W), else two s_r coincide")
    D = H * Wd
    Wh = Wd // 2 + 1
    F = H * Wh
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    sq = D ** 0.5

    offs = band_offsets(B)
    nG = len(offs)
    K = nG * c                                  # per-frequency block size
    dreps = delta_reps(B)
    nD = len(dreps)
    # (r, r') -> (index into dreps, take the conjugate)   with Delta = s_r' - s_r
    dmap = {}
    dlook = {}
    for i, (d1, d2) in enumerate(dreps):
        dlook[(d1, d2)] = (i, False)
        if (d1, d2) != (0, 0):
            dlook[(-d1, -d2)] = (i, True)
    for r, (a1, a2) in enumerate(offs):
        for rp, (b1, b2) in enumerate(offs):
            dmap[(r, rp)] = dlook[(b1 - a1, b2 - a2)]

    if sample_chunk is None:
        sample_chunk = max(1, min(N, int(2.0e9 / (c * D * 8))))
    nb = int(sample_chunk)
    if pass1_chunk is None:
        # pass 1 holds ONE rolled real copy of C_i per lag representative at once, so its
        # working set is ~len(reps) x (c, nb1, D) fp64 -- keep that near 6 GB.
        pass1_chunk = max(1, min(N, int(6.0e9 / (len(lag_reps(t)) * c * D * 8))))
    nb1 = int(pass1_chunk)

    mu = x0.mean(0)

    Hc = torch.fft.rfft2(h, dim=(-2, -1)).reshape(c, Cin, F)
    HpT = Hc.conj().permute(2, 1, 0).contiguous()
    del Hc
    nrm = torch.linalg.norm(h.reshape(c, -1), dim=1)
    s = sigma * nrm

    reps = lag_reps(t)
    full = lag_full(t)
    pos = {m: i for i, m in enumerate(full)}
    nl, nL = len(reps), len(full)

    psi = torch.stack([torch.einsum('acij,bcij->ab', h,
                                    torch.roll(h, shifts=(-m1, -m2), dims=(2, 3)))
                       for (m1, m2) in reps], dim=-1)
    psi /= (nrm.view(-1, 1, 1) * nrm.view(1, -1, 1))

    # exp(-2 pi i <Delta, u>) on the position grid, one row per Delta representative
    u1 = torch.arange(H, device=device, dtype=dtype).view(-1, 1).expand(H, Wd).reshape(-1)
    u2 = torch.arange(Wd, device=device, dtype=dtype).view(1, -1).expand(H, Wd).reshape(-1)
    dang = torch.stack([(-2.0 * np.pi) * (d1 * u1 / H + d2 * u2 / Wd) for (d1, d2) in dreps])
    dcos, dsin = torch.cos(dang), torch.sin(dang)               # (nD, D)
    del dang

    def prep(xs):
        xs = xs.to(device=device, dtype=dtype)
        ns = xs.shape[0]
        Xr = torch.fft.rfft2(xs, dim=(-2, -1)).reshape(ns, Cin, F)
        vv = torch.fft.rfft2(xs - mu, dim=(-2, -1)).reshape(ns, Cin, F) / sq
        tr = float(((xs - mu) ** 2).sum() / max(ns, 1))
        return {'Xr': Xr, 'vh': vv, 'N': ns, 'trace': tr}

    def feats(sp, n0, n1):
        pr = torch.matmul(sp['Xr'][n0:n1].permute(2, 0, 1), HpT)
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
        """Feature mean, the Delta-resolved lag tensor, and the Delta-resolved diagonal."""
        ns = sp['N']
        gmean = torch.zeros(c, D, dtype=dtype, device=device)
        dsum = torch.zeros(c, D, dtype=dtype, device=device)
        Tre = torch.zeros(nD, 3, nl, c, c, dtype=dtype, device=device)
        Tim = torch.zeros(nD, 3, nl, c, c, dtype=dtype, device=device)
        for n0 in range(0, ns, nb1):
            n1 = min(n0 + nb1, ns)
            M, Phi, ph, G, sa = feats(sp, n0, n1)
            gmean += G.sum(1)
            C = [sa * Phi, sa * ph / 2.0, -M * ph / 6.0]
            de = ((M ** 2 + sa ** 2) * Phi + M * sa * ph - G ** 2).sum(1)
            fo = (C[0] ** 2).sum(1) + 2.0 * (C[1] ** 2).sum(1) + 6.0 * (C[2] ** 2).sum(1)
            dsum += de - fo                       # per POSITION now, not just its mean
            del M, Phi, ph, G, de, fo
            for i in range(3):
                Ci = C[i].reshape(c, n1 - n0, H, Wd)
                # T(m,Delta) = (C_i o e_Delta) @ roll(C_i, +m)^T ; the roll is shared by
                # every Delta, the modulation by every lag, so neither is rebuilt inside
                # the other's loop.
                rolled = [torch.roll(Ci, shifts=(m1, m2), dims=(2, 3)).reshape(c, -1)
                          for (m1, m2) in reps]
                for dI in range(nD):
                    if dreps[dI] == (0, 0):
                        Ar, Ai = C[i].reshape(c, -1), None
                    else:
                        Ar = (C[i] * dcos[dI]).reshape(c, -1)
                        Ai = (C[i] * dsin[dI]).reshape(c, -1)
                    for li in range(nl):
                        Tre[dI, i, li].addmm_(Ar, rolled[li].T)
                        if Ai is not None:
                            Tim[dI, i, li].addmm_(Ai, rolled[li].T)
                    del Ar, Ai
                del Ci, rolled
            del C
        gmean /= ns
        Bc = torch.zeros(nD, c, c, 3 * nL, dtype=cdt, device=device)
        for dI, (d1, d2) in enumerate(dreps):
            for i in range(3):
                for li, (m1, m2) in enumerate(reps):
                    Tm = torch.complex(Tre[dI, i, li], Tim[dI, i, li]) / ns
                    Am = (psi[:, :, li] ** (i + 1)) * Tm
                    Bc[dI, :, :, i * nL + pos[(m1, m2)]] = coef[i] * Am
                    if (m1, m2) != (0, 0):
                        # psi_ab(-m)^n T_ab(-m,D) = exp(2 pi i <D,m>) [psi^n T]_ba(m,D)
                        pf = np.exp(2j * np.pi * (d1 * m1 / H + d2 * m2 / Wd))
                        Bc[dI, :, :, i * nL + pos[(-m1, -m2)]] = \
                            coef[i] * torch.tensor(pf, dtype=cdt, device=device) * Am.T
                    del Tm, Am
        del Tre, Tim
        # Delta-th Fourier coefficient of the exact-diagonal correction (Delta=0 = its mean)
        dg = torch.fft.fft2((dsum / (ns * D)).reshape(c, H, Wd), dim=(-2, -1))
        dgc = torch.stack([dg[:, d1 % H, d2 % Wd] for (d1, d2) in dreps])   # (nD, c)
        return gmean, Bc, dgc.to(cdt)

    tr_sp = prep(x0)
    gmean, B_tr, dgc_tr = pass1(tr_sp)
    te_sp = B_te = dgc_te = None
    if x0_test is not None:
        te_sp = prep(x0_test)
        _, B_te, dgc_te = pass1(te_sp)

    lm1 = torch.tensor([m[0] for m in full] * 3, device=device, dtype=dtype)
    lm2 = torch.tensor([m[1] for m in full] * 3, device=device, dtype=dtype)
    eye = torch.eye(K, dtype=cdt, device=device)

    f1i = (torch.arange(F, device=device, dtype=dtype) // Wh)
    f2i = (torch.arange(F, device=device, dtype=dtype) % Wh)
    selfconj = (f2i == 0) | ((Wd % 2 == 0) & (f2i == Wd // 2))
    wgt_all = torch.where(selfconj, 1.0, 2.0).to(dtype)
    so1 = torch.tensor([o[0] for o in offs], device=device)
    so2 = torch.tensor([o[1] for o in offs], device=device)

    NS = max(nb, int(super_chunk))

    def moments(sp, Bc, dgc, fa, fb):
        """P_f and q_f over the OUTPUT frequency block [fa, fb), noise included."""
        ns = sp['N']
        nf = fb - fa
        # source frequencies g = f - s_r, folded back into the rfft2 half-plane
        g1 = (f1i[fa:fb].long().view(-1, 1) - so1.view(1, -1)) % H          # (nf, nG)
        g2 = (f2i[fa:fb].long().view(-1, 1) - so2.view(1, -1)) % Wd
        cj = g2 >= Wh
        r1 = torch.where(cj, (-g1) % H, g1)
        r2 = torch.where(cj, (-g2) % Wd, g2)
        idx = (r1 * Wh + r2).reshape(-1)
        cjf = cj.reshape(-1)
        P = torch.zeros(nf, K, K, dtype=cdt, device=device)
        q = torch.zeros(nf, K, Cin, dtype=cdt, device=device)
        for ns0 in range(0, ns, NS):
            ns1 = min(ns0 + NS, ns)
            buf = torch.zeros(nf, ns1 - ns0, K, dtype=cdt, device=device)
            for n0 in range(ns0, ns1, nb):
                n1 = min(n0 + nb, ns1)
                M, Phi, ph, G, sa = feats(sp, n0, n1)
                Gc = (G - gmean.unsqueeze(1)).reshape(c, n1 - n0, H, Wd)
                U = torch.fft.rfft2(Gc, dim=(-2, -1)).reshape(c, n1 - n0, F) / sq
                sel = U[:, :, idx]
                sel = torch.where(cjf, sel.conj(), sel)
                # (c, nn, nf, nG) -> (nf, nn, nG, c) -> block index (r, a) = r*c + a
                buf[:, n0 - ns0:n1 - ns0, :] = sel.reshape(c, n1 - n0, nf, nG) \
                    .permute(2, 1, 3, 0).reshape(nf, n1 - n0, K)
                del M, Phi, ph, G, Gc, U, sel
            P += buf.transpose(1, 2) @ buf.conj()
            q += buf.transpose(1, 2) @ sp['vh'][ns0:ns1, :, fa:fb].conj().permute(2, 0, 1)
            del buf
        P /= ns
        q /= ns
        # noise: block (r,r') carries Delta = s_r' - s_r and the phase of g' = f - s_r'
        phs = []
        for rp, (b1, b2) in enumerate(offs):
            ang = (-2.0 * np.pi) * (torch.outer(f1i[fa:fb] - b1, lm1) / H
                                    + torch.outer(f2i[fa:fb] - b2, lm2) / Wd)
            phs.append(torch.complex(torch.cos(ang), torch.sin(ang)))       # (nf, 3nL)
            del ang
        for r in range(nG):
            for rp in range(nG):
                dI, takeconj = dmap[(r, rp)]
                Bd = Bc[dI].conj() if takeconj else Bc[dI]
                dv = dgc[dI].conj() if takeconj else dgc[dI]
                blk = torch.einsum('fm,abm->fab', phs[rp], Bd) / D
                blk += torch.diag_embed(dv).unsqueeze(0)
                P[:, r * c:(r + 1) * c, rp * c:(rp + 1) * c] += blk
                del blk
        del phs
        return P, q

    expl = 0.0
    dev_train = 0.0
    dev_test = 0.0
    fc = max(1, int(freq_chunk))
    for fa in range(0, F, fc):
        fb = min(fa + fc, F)
        wg = wgt_all[fa:fb]
        P, q = moments(tr_sp, B_tr, dgc_tr, fa, fb)
        w = torch.linalg.solve(P + lam * eye.unsqueeze(0), q)
        cross = torch.real(torch.sum(q.conj() * w, dim=(1, 2)))
        expl += float((cross * wg).sum())
        if te_sp is not None:
            quad = torch.real(torch.sum(w.conj() * torch.matmul(P, w), dim=(1, 2)))
            dev_train += float(((quad - 2.0 * cross) * wg).sum())
            del quad, P, q
            Pt, qt = moments(te_sp, B_te, dgc_te, fa, fb)
            quad = torch.real(torch.sum(w.conj() * torch.matmul(Pt, w), dim=(1, 2)))
            crt = torch.real(torch.sum(qt.conj() * w, dim=(1, 2)))
            dev_test += float(((quad - 2.0 * crt) * wg).sum())
            del Pt, qt, quad, crt
            del w, cross
        else:
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
# brute-force reference: the SAME model written as a plain BCCB readout over c*(2B+1)^2
# REAL modulated planes, solved as an explicit constrained least squares.  This is what
# checks that the unconstrained complex per-frequency solve is the real optimum.
# ---------------------------------------------------------------------------------------

def real_modulations(B, H, Wd, device='cpu', dtype=torch.float64):
    """The (2B+1)^2 REAL modulation functions: 1, then cos and sin for each +/- pair.

    span{1, cos<s,.>, sin<s,.> : s in half of box(B)} = span{exp(i<s,.>) : s in box(B)}
    intersected with the reals, which is exactly the set of readouts the complex
    parameterization reaches under W[-s] = conj(W[s]).
    """
    u1 = torch.arange(H, device=device, dtype=dtype).view(-1, 1).expand(H, Wd).reshape(-1)
    u2 = torch.arange(Wd, device=device, dtype=dtype).view(1, -1).expand(H, Wd).reshape(-1)
    mods = [torch.ones(H * Wd, device=device, dtype=dtype)]
    for (s1, s2) in band_half(B):
        ang = 2.0 * np.pi * (s1 * u1 / H + s2 * u2 / Wd)
        mods += [torch.cos(ang), torch.sin(ang)]
    return torch.stack(mods)                                    # (nR, D), nR = (2B+1)^2


def circulant2d_band_rf_mmse_bruteforce(x0, h, sigma, B, lam=1e-6, device='cpu',
                                        dtype=torch.float64, x0_test=None):
    """Same loss by materialising the K x K covariance of the MODULATED features.

    The modulation is a per-feature scalar multiplier, so the K x K moments of the plain
    features are formed once and conjugated by diag(mult) -- the nonlinearity is untouched.
    Only usable at toy sizes.

    ⚠ THE MOMENTS ARE BUILT ON THE *UNREPLICATED* (c*D) FEATURE SET AND THEN GATHERED, NOT
    on a theta with each row repeated nR times.  Those are NOT the same computation.
    `_kk_moments` substitutes the exact closed form  E[relu^2] - E[relu]^2  only on the
    LITERAL INDEX DIAGONAL, because that is where rho = 1 and the truncated Hermite series
    is at its worst.  Replicating a row puts rho = 1 pairs OFF the diagonal, where they
    silently keep the truncated value -- a ~1e-6 relative error in the loss, constant in
    lam, which is exactly the discrepancy this reference showed before the rewrite.
    Indexing the (a,u) moment matrix and scaling by m_j[u] instead keeps every rho = 1 pair
    on the diagonal of `Sig0`, so the correction is applied once and reused for all (j,j').
    """
    x0 = x0.to(device=device, dtype=dtype)
    h = h.to(device=device, dtype=dtype)
    N, Cin, H, Wd = x0.shape
    c = h.shape[0]
    D = H * Wd
    d = Cin * D
    mods = real_modulations(B, H, Wd, device, dtype)
    nR = mods.shape[0]
    ceff = c * nR
    K = ceff * D

    Th = _explicit_theta(h, H, Wd).to(device=device, dtype=dtype)           # (c*D, d)
    # full index k = (a*nR + j)*D + u  -- the ordering `_bccb_basis` expects over ceff
    # planes -- maps to the unreplicated row src[k] = a*D + u and the multiplier m_j[u].
    kk = torch.arange(K, device=device)
    src = (kk // (nR * D)) * D + (kk % D)
    mult = mods[(kk // D) % nR, kk % D].contiguous()

    Xf = x0.reshape(N, d)
    mu = Xf.mean(0)
    Sig0, Sxp0, trace_p0, gm0 = _kk_moments(Xf, Th, sigma, mu, None)
    Sig = mult.view(-1, 1) * Sig0[src][:, src] * mult.view(1, -1)
    Sxp = Sxp0[:, src] * mult.view(1, -1)

    E = _bccb_basis(Cin, ceff, H, Wd, device, dtype)
    A = torch.einsum('pri,ij,qrj->pq', E, Sig, E)
    b = torch.einsum('pri,ri->p', E, Sxp)
    # <E_p,E_p>_F = sum_u m_j[u]^2 = D for the constant plane, D/2 for every cos/sin plane.
    # That weighting -- NOT a uniform lam*D*I -- is the tap-space image of the per-frequency
    # lam*I, because sum over the complex pair {s,-s} of |W_s|^2 is (alpha^2+beta^2)/2.
    modnorm = (mods ** 2).sum(1)                                            # (nR,)
    pj = (torch.arange(Cin * ceff * D, device=device) // D) % ceff % nR
    reg = torch.diag(lam * modnorm[pj])
    w = torch.linalg.solve(A + reg, b)
    train = float(trace_p0 - float(b @ w))
    if x0_test is None:
        return train
    xt = x0_test.to(device=device, dtype=dtype).reshape(-1, d)
    St0, Sxt0, tr_t, _ = _kk_moments(xt, Th, sigma, mu, gm0)
    St = mult.view(-1, 1) * St0[src][:, src] * mult.view(1, -1)
    Sxt = Sxt0[:, src] * mult.view(1, -1)
    At = torch.einsum('pri,ij,qrj->pq', E, St, E)
    bt = torch.einsum('pri,ri->p', E, Sxt)
    return {'train': train,
            'train_resid': float(trace_p0 - 2.0 * float(b @ w) + float(w @ (A @ w))),
            'test': float(tr_t - 2.0 * float(bt @ w) + float(w @ (At @ w)))}


def _toy(rng, n, Cin, H, Wd, A, shift, scale, dev):
    """Non-Gaussian marginals, cross-coordinate correlation, non-stationary mean."""
    d = Cin * H * Wd
    z = rng.standard_normal((n, d))
    z = z + 0.4 * rng.standard_normal((n, 1))
    raw = (z ** 3) / 3.0 @ A.T * scale + np.linspace(-1.0, 1.0, d)[None, :] + shift
    return torch.tensor(raw.reshape(n, Cin, H, Wd), dtype=torch.float64, device=dev)


def selftest_band(seed=0, verbose=True, device=None):
    """Three independent checks, all of which must pass:

    (a) BRUTE FORCE.  The structured complex per-frequency solve against an explicit real
        constrained least squares over the c(2B+1)^2 modulated planes, on non-Gaussian,
        non-stationary, cross-channel-correlated data, in-sample AND held out (the test
        split drawn from a SHIFTED, RESCALED distribution so the moments genuinely differ).
        This validates the Delta-resolved Stein assembly, the Delta-dependent diagonal
        correction, the shifted-frequency gather, the Hermitian fold, the ridge convention
        AND the claim that the unconstrained complex solve is already the real optimum --
        if any of those were wrong the two would not agree.
    (b) B = 0 REGRESSION against `core.rf_circulant2d.circulant2d_rf_mmse`, which exercises
        the whole new code path against a separately validated implementation at CIFAR-like
        settings (odd t, Cin=3) rather than toy ones.
    (c) lam = 0 with test == train, where the held-out loss must collapse exactly onto the
        in-sample one.  Pins the sign and normalisation of the -2Re<q,w> + w^H P w assembly
        without either reference.
    """
    from core.rf_circulant2d import circulant2d_rf_mmse
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ok = True

    # *** THE GRID MUST BE AT LEAST 2B+1 IN EACH DIMENSION. ***  box(B) offsets are taken mod
    # (H, Wd), so on a grid smaller than 2B+1 two distinct s_r alias onto the SAME modulation,
    # the design goes rank-deficient and the ridge -- not the algebra -- decides the answer.
    # That would make a brute-force agreement meaningless rather than failing loudly, so the
    # B=3 cases below run at 7x7 and 8x8, not at the 4x4/5x5 used for B=1 and B=2.
    cases = ((4, 4, 2, 2, 2, 400, 1.1, 1),
             (4, 4, 3, 2, 2, 400, 0.6, 1),
             (6, 4, 2, 2, 2, 400, 1.5, 1),
             (6, 6, 1, 2, 2, 400, 0.9, 1),
             (5, 5, 1, 2, 2, 300, 0.9, 2),
    # ⚠ AND THE BRUTE FORCE COSTS Cin*c*nR*D x c*nR*D x d DOUBLES FOR ITS BASIS TENSOR `E`,
    # i.e. it grows like B^4 c^2 Cin D^2.  At B=3 that is 9 GB for (7,7,Cin=1,c=2) and 80 GB
    # for (8,8,Cin=2,c=2) -- the reference OOMs long before the estimator would.  Keep the
    # Cin>1 B=3 case at c=1 so it costs the same as the Cin=1 c=2 one.
             (7, 7, 1, 2, 2, 300, 0.9, 3),
             (7, 7, 2, 1, 2, 300, 1.2, 3),
    # *** AND THE GRID MUST ALSO BE AT LEAST 2t-1 IN EACH DIMENSION, FOR THE SAME REASON ON
    # THE OTHER INDEX. ***  The lag set is m in [-(t-1), t-1]^2 taken mod (H, Wd), so on a
    # grid smaller than 2t-1 two distinct lags alias and the Delta-resolved Stein Gram
    # double-counts.  `circulant2d_band_rf_mmse` raises on that (see the 2t-1 <= min(H,W)
    # check), which is why the two cases below run at 13x13 and 7x7 rather than reusing the
    # small grids above.
    #
    # ⚠ EVERY CASE ABOVE RUNS AT t=2.  Until 2026-09-23 the brute force had NEVER exercised a
    # tap size larger than that, so the lag machinery was validated only at its smallest
    # non-trivial setting -- exactly the hole B=3 was in before the two 7x7 cases were added.
    # These two close it before the T2=7 CIFAR sweep (michimin, 2026-09-23 22:30: "is a
    # 27-dimensional window enough for a useful nonlinear feature").  t=7 on 13x13 is the
    # tightest possible case: the lag span is exactly the grid, so any off-by-one in the
    # wraparound shows up as a hard disagreement rather than as a small bias.
    # E costs 3.1 GB at (13,13,Cin=1,c=1,B=1) and 1.2 GB at (7,7,Cin=2,c=2,B=1); the c and Cin
    # are kept small for that reason, NOT because large t is expensive (t does not enter E).
             (13, 13, 1, 1, 7, 300, 0.9, 1),
             (7, 7, 2, 2, 4, 300, 1.2, 1))
    for (H, Wd, Cin, c, t, N, sig, B) in cases:
        rng = np.random.default_rng(seed)
        d = Cin * H * Wd
        A = rng.standard_normal((d, d)) * 0.5
        x0 = _toy(rng, N, Cin, H, Wd, A, 0.0, 1.0, dev)
        xt = _toy(rng, N // 2 + 37, Cin, H, Wd, A, 0.25, 1.15, dev)
        hh = torch.zeros(c, Cin, H, Wd, dtype=torch.float64, device=dev)
        hh[:, :, :t, :t] = torch.tensor(
            rng.standard_normal((c, Cin, t, t)) / np.sqrt(Cin * t * t),
            dtype=torch.float64, device=dev)

        new = circulant2d_band_rf_mmse(x0, hh, sig, t, B, lam=1e-6, device=dev,
                                       sample_chunk=7, pass1_chunk=9, freq_chunk=3,
                                       super_chunk=13, x0_test=xt)
        ref = circulant2d_band_rf_mmse_bruteforce(x0, hh, sig, B, lam=1e-6, device=dev,
                                                  x0_test=xt)
        rel = {k: abs(new[k] - ref[k]) / max(abs(ref[k]), 1e-12)
               for k in ('train', 'train_resid', 'test')}
        good = max(rel.values()) < 1e-9
        ok &= good
        if verbose:
            print(f"  brute  {H}x{Wd} Cin={Cin} c={c} t={t} B={B} sigma={sig:<4}: "
                  f"train {new['train']:.8f} ({rel['train']:.1e})  "
                  f"resid ({rel['train_resid']:.1e})  "
                  f"test {new['test']:.8f} ({rel['test']:.1e})  "
                  f"{'OK' if good else 'FAIL'}", flush=True)

    # (b) B = 0 must reproduce the plain 2-D estimator bit for bit
    rng = np.random.default_rng(seed + 3)
    H = Wd = 8; Cin = 3; c = 3; t = 3; N = 500
    d = Cin * H * Wd
    A = rng.standard_normal((d, d)) * 0.5
    x0 = _toy(rng, N, Cin, H, Wd, A, 0.0, 1.0, dev)
    xt = _toy(rng, 211, Cin, H, Wd, A, 0.3, 1.2, dev)
    hh = torch.zeros(c, Cin, H, Wd, dtype=torch.float64, device=dev)
    hh[:, :, :t, :t] = torch.tensor(rng.standard_normal((c, Cin, t, t)) / np.sqrt(Cin * t * t),
                                    dtype=torch.float64, device=dev)
    for sig in (0.5, 1.7):
        a = circulant2d_band_rf_mmse(x0, hh, sig, t, 0, lam=1e-6, device=dev,
                                     sample_chunk=17, pass1_chunk=23, freq_chunk=5,
                                     super_chunk=41, x0_test=xt)
        b = circulant2d_rf_mmse(x0, hh, sig, t, lam=1e-6, device=dev, sample_chunk=13,
                                freq_chunk=7, super_chunk=29, x0_test=xt)
        rel = max(abs(a[k] - b[k]) / max(abs(b[k]), 1e-12)
                  for k in ('train', 'train_resid', 'test'))
        ok &= rel < 1e-11
        if verbose:
            print(f"  B=0 vs circulant2d_rf_mmse  sigma={sig}: train {a['train']:.8f} vs "
                  f"{b['train']:.8f}   rel={rel:.1e} {'OK' if rel < 1e-11 else 'FAIL'}",
                  flush=True)

    # (c) lam = 0, test == train
    rng = np.random.default_rng(seed + 5)
    H = Wd = 6; Cin = 2; c = 2; t = 2; N = 400
    x0 = torch.tensor(rng.standard_normal((N, Cin, H, Wd)) * 1.3 + 0.2,
                      dtype=torch.float64, device=dev)
    hh = torch.zeros(c, Cin, H, Wd, dtype=torch.float64, device=dev)
    hh[:, :, :t, :t] = torch.tensor(rng.standard_normal((c, Cin, t, t)) / np.sqrt(Cin * t * t),
                                    dtype=torch.float64, device=dev)
    r = circulant2d_band_rf_mmse(x0, hh, 0.8, t, 1, lam=0.0, device=dev, sample_chunk=11,
                                 pass1_chunk=13, freq_chunk=4, super_chunk=23, x0_test=x0)
    rel = abs(r['test'] - r['train']) / max(abs(r['train']), 1e-12)
    ok &= rel < 1e-9
    if verbose:
        print(f"  test==train, lam=0, B=1: train {r['train']:.10f}  test {r['test']:.10f}  "
              f"rel={rel:.2e} {'OK' if rel < 1e-9 else 'FAIL'}", flush=True)
    return ok


if __name__ == '__main__':
    print("selftest_band:", "PASS" if selftest_band() else "FAIL")
