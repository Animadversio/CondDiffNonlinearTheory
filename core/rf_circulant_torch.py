"""
core/rf_circulant_torch.py

GPU/torch backend for the circulant-constrained RF denoiser (core.rf_circulant).
Reuses stein_covariances_t for the (k,k) noise-marginalised covariances, then does
the per-frequency c x c solves as a single batched torch.linalg.solve. Same math as
core.rf_circulant.circulant_rf_mmse.
"""

import numpy as np
import torch

from .rf_gmm_estimators_torch import stein_covariances_t


def _dft_matrix_t(d, device, dtype):
    j = torch.arange(d, device=device, dtype=dtype)
    ang = -2.0 * np.pi * torch.outer(j, j) / d
    return torch.complex(torch.cos(ang), torch.sin(ang)) / (d ** 0.5)   # (d,d)


def circulant_rf_mmse_t(x0, U, Theta, Gamma, sigma, lam, conditional=True,
                        device='cuda', dtype=torch.float64):
    """GPU port of core.rf_circulant.circulant_rf_mmse. Theta must be block-circulant."""
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

    # Batched solve over the d frequencies: sol_f = P_f^{-1} q_f
    sol = torch.linalg.solve(Pblocks, Qblocks.unsqueeze(-1)).squeeze(-1)   # (d, c)
    expl = torch.real(torch.sum(Qblocks.conj() * sol))                     # sum_f q_f^H P_f^{-1} q_f
    return max(0.0, trace_p0 - float(expl))
