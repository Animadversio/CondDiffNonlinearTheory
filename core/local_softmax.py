"""LOCAL EMPIRICAL-BAYES (SOFTMAX) DENOISERS: Kamb & Ganguli LS / ELS and Lukoianov et al.

Ported from https://github.com/analytic-diffusion/analytic-diffusion-studio
(src/local_diffusion/models/local_score_machines.py and models/pca_locality.py, CC BY-NC 4.0),
re-derived in the EDM / VE convention used by scripts/edm_pixel_heldout.py:

    x in [-1, 1],   y = x0 + sigma_edm * z,   sigma_edm = 2 * sigma_pixel.

The reference code is VP (x_t = sqrt(abar) x0 + sqrt(1-abar) z) and every logit there is
-||x_t - sqrt(abar) x_n||^2 / (2 (1-abar)); substituting y = x_t / sqrt(abar) and
sigma^2 = (1-abar)/abar gives exactly the -||y - x_n||^2 / (2 sigma^2) used below, so the
x0-hat is the same function -- minus the reference's snapping of sigma to one of 1000 DDPM
timesteps.  All three estimators are the global empirical-Bayes oracle of
scripts/bayes_oracle_heldout.py with the distance restricted to a WINDOW around each output
coordinate:

    x_hat(y)[m] = sum_cand softmax( -||M_m (y - cand)||^2 / 2 sigma^2 ) * cand[m]

  LS    (Kamb & Ganguli, position-specific)   cand = train image n at the SAME location;
        M_m = k x k square window around pixel(m), all channels, ZERO-padded (window truncated
        at the border), exactly as the reference pads the squared-difference map.
  ELS   (Kamb & Ganguli, equivariant)         cand = every k x k patch of every train image at
        EVERY location (N * H * W candidates per output pixel), CIRCULAR boundary as in the
        reference ELSMachine; the value is the candidate patch's centre pixel.
  LUK   (Lukoianov et al. 2025, "pca_locality") cand = train image n at the same coordinate;
        M_m = binarised row m of the Wiener filter W = Sigma (Sigma + sigma^2 I)^{-1},
        row-normalised by W_mm and thresholded at tau * max|W / diag W| (the reference uses
        the GLOBAL max; tau_mode='row' gives the per-row reading of the paper's text).
        The mask spans channels.  Sigma is estimated on the POOL (train[:N]), not on 50k.

TWO DELIBERATE DEPARTURES FROM THE REFERENCE, BOTH IN THE DIRECTION OF THE PAPERS' EQUATIONS:
  * every softmax is over the WHOLE pool (online log-sum-exp across chunks).  The reference
    pca_locality normalises inside each 512-image dataloader batch and averages the batches
    (WeightedStreamingSoftmax.add), which is not Eq. 8 and blurs heavily at low sigma.
  * sigma is continuous, not snapped to a DDPM timestep.

PRECISION.  The LS / LUK distances are a GEMM of NON-NEGATIVE squared differences against a
0/1 mask, so there is no cancellation and float64 on the H100 tensor cores is cheap.  ELS
builds the per-pixel squared difference directly (no ||a||^2 - 2ab + ||b||^2 expansion) and
box-sums it with an integral image in `els_dtype` (float64 by default).

Every function is validated in scripts/local_softmax_heldout.py::_validate against a
brute-force loop, against the global oracle (window covering everything), and against the
sigma -> infinity limit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------------------------
def ls_window_mask(H: int, W: int, k: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """(H*W, H*W) 0/1 mask: row p selects the k x k window around pixel p, truncated at the
    border (zero padding -- the reference LocalScoreMachine).  k >= 2*max(H,W)-1 is global."""
    r = k // 2
    ii = torch.arange(H, device=device)
    jj = torch.arange(W, device=device)
    mi = (ii[:, None] - ii[None, :]).abs() <= r            # (H, H)
    mj = (jj[:, None] - jj[None, :]).abs() <= r            # (W, W)
    return (mi[:, None, :, None] & mj[None, :, None, :]).reshape(H * W, H * W).to(dtype)


def wiener_filter(evals: torch.Tensor, U: torch.Tensor, sigma: float) -> torch.Tensor:
    """W = U diag(lam / (lam + sigma^2)) U^T  (d x d)."""
    shr = evals / (evals + sigma ** 2)
    return (U * shr) @ U.T


def wiener_locality_mask(evals: torch.Tensor, U: torch.Tensor, sigma: float, tau: float,
                         tau_mode: str = 'global') -> torch.Tensor:
    """Lukoianov et al. mask, (d, d) 0/1: |W_mj / W_mm| >= tau * max.  `global` = reference
    code (max over the whole normalised matrix), `row` = max within row m."""
    Wf = wiener_filter(evals, U, sigma)
    Wn = (Wf / torch.diagonal(Wf).unsqueeze(1)).abs()
    if tau_mode == 'global':
        thr = tau * Wn.max()
    elif tau_mode == 'row':
        thr = tau * Wn.max(dim=1, keepdim=True).values
    else:
        raise ValueError(tau_mode)
    return (Wn >= thr).to(evals.dtype)


# ----------------------------------------------------------------------------------------
# position-specific masked softmax  (LS and LUK)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def masked_softmax_denoise(y: torch.Tensor, X: torch.Tensor, sigma: float, mask: torch.Tensor,
                           pixel_mask: bool, C: int = 3, pool_chunk: int = 1024,
                           query_chunk: int = 16, return_neff: bool = False):
    """Position-specific windowed posterior mean, exact softmax over the whole pool.

    y (B, d), X (N, d), flattened channel-major (C, H, W) so coordinate m = c*HW + p.
    pixel_mask=True  : mask is (HW, HW) on PIXELS; the distance sums channels first and every
                       channel of pixel p shares row p's weights (LS).
    pixel_mask=False : mask is (d, d) on COORDINATES; each output coordinate has its own
                       weights (LUK).
    Returns x_hat (B, d) [and the median posterior N_eff = exp(H(w)) over rows if asked].
    """
    B, d = y.shape
    N = X.shape[0]
    HW = d // C
    R = mask.shape[0]
    MT = mask.T.contiguous().to(y.dtype)
    inv = 1.0 / (2.0 * sigma ** 2)
    out = torch.empty_like(y)
    neff = []
    for b0 in range(0, B, query_chunk):
        yb = y[b0:b0 + query_chunk]
        nb = yb.shape[0]
        m = torch.full((nb, R), -float('inf'), device=y.device, dtype=y.dtype)
        s = torch.zeros((nb, R), device=y.device, dtype=y.dtype)
        num = torch.zeros((nb, d), device=y.device, dtype=y.dtype)
        ent = torch.zeros((nb, R), device=y.device, dtype=y.dtype) if return_neff else None
        for n0 in range(0, N, pool_chunk):
            Xc = X[n0:n0 + pool_chunk]
            sq = (yb[:, None, :] - Xc[None]) ** 2                    # (nb, nc, d)
            if pixel_mask:
                sq = sq.view(nb, Xc.shape[0], C, HW).sum(2)          # (nb, nc, HW)
            logit = -(sq @ MT) * inv                                 # (nb, nc, R)
            del sq
            mnew = torch.maximum(m, logit.max(1).values)
            sc = torch.exp(m - mnew)
            p = torch.exp(logit - mnew[:, None, :])                  # (nb, nc, R)
            if return_neff:     # running sum of p * logit, rescaled with the same factor
                ent = ent * sc + (p * logit).sum(1)
            s = s * sc + p.sum(1)
            if pixel_mask:
                sc_d = sc.repeat(1, C)                               # coordinate c*HW+p -> row p
                pw = p.repeat(1, 1, C)                               # (nb, nc, d)
            else:
                sc_d, pw = sc, p
            num = num * sc_d + (pw * Xc[None]).sum(1)
            m = mnew
            del logit, p, pw
        s_d = s.repeat(1, C) if pixel_mask else s
        out[b0:b0 + nb] = num / s_d
        if return_neff:
            # H(w) = log s + m - E_w[logit]
            H = torch.log(s) + m - ent / s
            neff.append(torch.exp(H).median(dim=1).values)
    if return_neff:
        return out, torch.cat(neff)
    return out


# ----------------------------------------------------------------------------------------
# equivariant local score machine (ELS), circular boundary
# ----------------------------------------------------------------------------------------
def _shift_index(H: int, W: int, device) -> torch.Tensor:
    """idx[s, u] = flat index of (u + s) on the H x W torus, s and u both flat."""
    i = torch.arange(H, device=device)
    j = torch.arange(W, device=device)
    ui, uj = torch.meshgrid(i, j, indexing='ij')
    si, sj = ui.reshape(-1), uj.reshape(-1)                      # s
    vi = (si[:, None] + ui.reshape(1, -1)) % H
    vj = (sj[:, None] + uj.reshape(1, -1)) % W
    return vi * W + vj                                           # (HW_s, HW_u)


def _box_sums(Q: torch.Tensor, ks, dtype) -> dict:
    """Circular k x k window sums over the FIRST two dims of Q (H, W, L), for every k in ks,
    from ONE integral image.  Requires k <= min(H, W) so the window never overlaps itself.

    The candidate planes are the LAST (contiguous) dim on purpose: a cumsum along a short
    trailing spatial axis ran ~15x slower on the H100 (38 ms vs 2.3 ms per 32x1024 planes)."""
    H, W = Q.shape[:2]
    rmax = max(ks) // 2
    P = Q.to(dtype)
    P = torch.cat([P[H - rmax:], P, P[:rmax]], 0)
    P = torch.cat([P[:, W - rmax:], P, P[:, :rmax]], 1)          # (H+2r, W+2r, L)
    I = F.pad(P.cumsum(0).cumsum(1), (0, 0, 1, 0, 1, 0))         # (H+2r+1, W+2r+1, L)
    del P
    out = {}
    for k in ks:
        r = k // 2
        a, b = rmax - r, rmax + r + 1                            # rows a..b-1 for centre 0
        out[k] = (I[b:b + H, b:b + W] - I[a:a + H, b:b + W]
                  - I[b:b + H, a:a + W] + I[a:a + H, a:a + W])
    return out


@torch.no_grad()
def els_denoise(y: torch.Tensor, X: torch.Tensor, sigma: float, ks, pool_chunk: int = 32,
                els_dtype=torch.float64) -> dict:
    """Equivariant local score machine, circular boundary, exact softmax over N*H*W patches.

    y (B, C, H, W), X (N, C, H, W).  Returns {k: x_hat (B, C, H, W)} for every odd k in ks
    (all k share the squared-difference map and its integral image).

    For query pixel u and candidate (n, s) -- patch of train image n centred at u+s --
        d2(u; n, s) = sum_{|delta| <= r} || y(u+delta) - x_n(u+s+delta) ||^2
                    = box_r[ Q_{n,s} ](u),   Q_{n,s}(w) = || y(w) - x_n(w+s) ||^2,
    so one shift-gather of the pool and one integral image serve every window size.
    Layout: query pixel u leads, candidates (n, s) trail, so the per-u softmax reduces along
    the contiguous dim and the numerator is one batched (1 x nS) @ (nS x C) matmul per u.
    """
    ks = sorted(set(int(k) for k in ks))
    B, C, H, W = y.shape
    assert all(k % 2 == 1 and k <= min(H, W) for k in ks), ks
    HW = H * W
    idx = _shift_index(H, W, y.device)                           # (S, U): flat u+s
    inv = 1.0 / (2.0 * sigma ** 2)
    yT = y.reshape(B, C, HW).transpose(1, 2).to(els_dtype)       # (B, U, C)
    st = {k: dict(m=torch.full((B, HW), -float('inf'), device=y.device, dtype=els_dtype),
                  s=torch.zeros((B, HW), device=y.device, dtype=els_dtype),
                  num=torch.zeros((B, HW, C), device=y.device, dtype=els_dtype)) for k in ks}
    for n0 in range(0, X.shape[0], pool_chunk):
        Xc = X[n0:n0 + pool_chunk].reshape(-1, C, HW).to(els_dtype)
        nc = Xc.shape[0]
        # Xs[u, n, s, c] = x_n[c](u + s)
        Xs = Xc[:, :, idx.T].permute(2, 0, 3, 1).contiguous()    # (U, nc, S, C)
        Xv = Xs.view(HW, nc * HW, C)
        for b in range(B):
            Q = ((yT[b][:, None, None, :] - Xs) ** 2).sum(-1)    # (U, nc, S)
            D = _box_sums(Q.view(H, W, nc * HW), ks, els_dtype)
            del Q
            for k in ks:
                lg = D[k].reshape(HW, nc * HW) * (-inv)          # (U, nc*S)
                t = st[k]
                mnew = torch.maximum(t['m'][b], lg.max(1).values)
                sc = torch.exp(t['m'][b] - mnew)
                p = torch.exp(lg - mnew[:, None])
                t['s'][b] = t['s'][b] * sc + p.sum(1)
                t['num'][b] = t['num'][b] * sc[:, None] + torch.bmm(p[:, None, :], Xv)[:, 0]
                t['m'][b] = mnew
                del lg, p
            del D
        del Xs, Xv
    return {k: (st[k]['num'] / st[k]['s'][:, :, None]).transpose(1, 2)
            .reshape(B, C, H, W).to(y.dtype) for k in ks}


# ----------------------------------------------------------------------------------------
# linear reference in the same convention
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def wiener_denoise(y: torch.Tensor, mu: torch.Tensor, evals: torch.Tensor, U: torch.Tensor,
                   sigma: float) -> torch.Tensor:
    """x_hat = mu + W (y - mu), W from the pool covariance; y (B, d)."""
    shr = evals / (evals + sigma ** 2)
    return mu + (((y - mu) @ U) * shr) @ U.T
