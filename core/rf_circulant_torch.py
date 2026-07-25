"""
core/rf_circulant_torch.py

GPU/torch backend for the circulant-constrained RF denoiser (core.rf_circulant).
Reuses stein_covariances_t for the (k,k) noise-marginalised covariances, then does
the per-frequency c x c solves as a single batched torch.linalg.solve. Same math as
core.rf_circulant.circulant_rf_mmse.
"""

import numpy as np
import torch

from .rf_gmm_estimators_torch import stein_covariances_t, _c0_t, _to


def _dft_matrix_t(d, device, dtype):
    j = torch.arange(d, device=device, dtype=dtype)
    ang = -2.0 * np.pi * torch.outer(j, j) / d
    return torch.complex(torch.cos(ang), torch.sin(ang)) / (d ** 0.5)   # (d,d)


def circulant_rf_mmse_t(x0, U, Theta, Gamma, sigma, lam, conditional=True,
                        device='cuda', dtype=torch.float64, equivariant_bias=False):
    """GPU port of core.rf_circulant.circulant_rf_mmse. Theta must be block-circulant.

    equivariant_bias : False (default) = free readout bias in R^d; the matching floor is
        then core.equivariant_floor's 'floor_free', NOT MMSE(pbar0). True = b = beta*1,
        genuinely shift-equivariant, for which MMSE(pbar0) IS the floor. See the numpy
        version for the math.
    """
    Cov, Sig, trace_p0 = stein_covariances_t(x0, U, Theta, Gamma, sigma, lam,
                                             conditional, device, dtype)
    d = Cov.shape[0]
    k = Sig.shape[0]
    c = k // d
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    F = _dft_matrix_t(d, device, dtype).to(cdtype)                      # (d,d)
    Fc = F.conj()

    Sig4 = Sig.reshape(c, d, c, d).to(cdtype)                           # [a,p,b,q]
    # P^{(a,b)}_f = f_f^H Sigma^{(a,b)} f_f  -> (f, a, b)
    Pblocks = torch.einsum('fp,apbq,fq->fab', F, Sig4, Fc)              # (d, c, c)

    CovT4 = Cov.T.reshape(c, d, d).to(cdtype)                           # [a,p,q]
    Qblocks = torch.einsum('fp,apq,fq->fa', F, CovT4, Fc)              # (d, c)

    const = 0.0
    if equivariant_bias:
        mu_phi, mu_x = _stein_means_t(x0, U, Theta, Gamma, sigma, conditional, device, dtype)
        mphi_f = (F @ mu_phi.reshape(c, d).T.to(cdtype))                # (d, c)
        mx_f = F @ mu_x.to(cdtype)                                      # (d,)
        # skip DC (f=0): a constant bias is free there
        Pblocks[1:] = Pblocks[1:] + mphi_f[1:].unsqueeze(-1) * mphi_f[1:].conj().unsqueeze(-2)
        Qblocks[1:] = Qblocks[1:] + mphi_f[1:] * mx_f[1:].conj().unsqueeze(-1)
        const = float((mx_f[1:].abs() ** 2).sum())

    # Batched solve over the d frequencies: sol_f = P_f^{-1} q_f
    sol = torch.linalg.solve(Pblocks, Qblocks.unsqueeze(-1)).squeeze(-1)   # (d, c)
    expl = torch.real(torch.sum(Qblocks.conj() * sol))                     # sum_f q_f^H P_f^{-1} q_f
    return max(0.0, trace_p0 + const - float(expl))


def _stein_means_t(x0, U, Theta, Gamma, sigma, conditional, device, dtype):
    """(mu_phi, mu_x): noise-marginalised feature mean E[phi] (k,) and data mean (d,)."""
    x0 = _to(x0, device, dtype); Theta = _to(Theta, device, dtype)
    Gamma = _to(Gamma, device, dtype); U = _to(U, device, dtype)
    s = sigma * torch.linalg.norm(Theta, dim=1)                          # (k,)
    M = x0 @ Theta.T + (U @ Gamma.T if conditional else 0.0)             # (N, k)
    return _c0_t(M, s[None, :]).mean(0), x0.mean(0)


def circulant_rf_mmse_pop_t(gmm, Theta, Gamma, sigma, lam=1e-4, n_max=6,
                            conditional=False, device='cuda', dtype=torch.float64,
                            equivariant_bias=False):
    """L^circ from EXACT POPULATION moments of the GMM (no sampling error).

    Same per-frequency solve as circulant_rf_mmse_t, but Cov(x0,phi) and Sigma_phi come
    from the exact per-component Stein/Hermite population theory
    (core.rf_gmm_estimators_torch.mmse_theory_gmm_pop_t) instead of an empirical sample.

    Why this matters: with an empirical sample of size N the measured L^circ is the loss
    on p0_emp, whose Tr(Sigma) differs from the population by O(1/sqrt(N)). At large sigma
    the loss is ~Tr(Sigma) - small, so that deficit propagates almost 1:1 and L^circ can
    appear to fall below a POPULATION-computed floor purely as a finite-sample artefact
    (seen at sigma=5: Tr deficit 0.066 at N=16000). Using population moments on both sides
    makes the floor comparison exact.
    """
    from .rf_gmm_estimators_torch import mmse_theory_gmm_pop_t
    Cov, Sig, trace_p0, mu_phi, mu_x = mmse_theory_gmm_pop_t(
        gmm, Theta, Gamma, sigma, lam=lam, n_max=n_max, conditional=conditional,
        device=device, dtype=dtype, return_covs=True)
    d = Cov.shape[0]; k = Sig.shape[0]; c = k // d
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    F = _dft_matrix_t(d, device, dtype).to(cdtype); Fc = F.conj()

    Sig4 = Sig.reshape(c, d, c, d).to(cdtype)
    Pblocks = torch.einsum('fp,apbq,fq->fab', F, Sig4, Fc)
    CovT4 = Cov.T.reshape(c, d, d).to(cdtype)
    Qblocks = torch.einsum('fp,apq,fq->fa', F, CovT4, Fc)

    const = 0.0
    if equivariant_bias:
        mphi_f = (F @ mu_phi.reshape(c, d).T.to(cdtype))
        mx_f = F @ mu_x.to(cdtype)
        Pblocks[1:] = Pblocks[1:] + mphi_f[1:].unsqueeze(-1) * mphi_f[1:].conj().unsqueeze(-2)
        Qblocks[1:] = Qblocks[1:] + mphi_f[1:] * mx_f[1:].conj().unsqueeze(-1)
        const = float((mx_f[1:].abs() ** 2).sum())

    sol = torch.linalg.solve(Pblocks, Qblocks.unsqueeze(-1)).squeeze(-1)
    expl = torch.real(torch.sum(Qblocks.conj() * sol))
    return max(0.0, trace_p0 + const - float(expl))
