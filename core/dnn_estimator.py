"""
DNN feature-based MMSE estimator.

Given a fixed (pretrained) encoder phi(y) -> R^k, estimates the Bayes-optimal
linear-readout denoiser loss:

    L_sigma = Tr(Sigma_p0) - Tr(Cov(x0, phi) Sigma_phi^{-1} Cov(phi, x0))

and the conditional version with U via three feature-concatenation strategies:

    A) phi^U(y, U) = [phi(y);  U]                   -- raw concat
    B) phi^U(y, U) = [phi(y);  omega(Gamma @ U + b)] -- random nonlinear U features
    C) phi^U(y, U) = [phi(y);  Gamma @ U;  phi(y) * (W @ U)]  -- + linear interaction

Also provides the analytic linear (Wiener filter) baseline.

All covariance estimation uses the population formula (expectations over x0 ~ p0 and Z ~ N(0,I)).
With finite samples the covariances are estimated from data; regularized with ridge lambda.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, Literal


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    x0_batch: torch.Tensor,
    sigma: float,
    n_noise: int = 1,
    device: str = 'cuda',
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract DNN features from noisy observations y = x0 + sigma * Z.

    Parameters
    ----------
    encoder    : fixed network phi, maps images -> (N, k) features
    x0_batch   : (N, C, H, W) clean images, values in [0,1] or normalized
    sigma      : noise level
    n_noise    : number of noise draws per image (for variance reduction)
    device     : 'cuda' or 'cpu'

    Returns
    -------
    Phi : (N * n_noise, k)  feature matrix
    X0  : (N * n_noise, d)  flattened clean images (repeated n_noise times)
    """
    if rng is None:
        rng = np.random.default_rng()

    encoder = encoder.to(device).eval()
    N = x0_batch.shape[0]
    d = x0_batch[0].numel()

    x0_rep = x0_batch.repeat_interleave(n_noise, dim=0)  # (N*n_noise, C, H, W)
    Z = torch.from_numpy(
        rng.standard_normal(x0_rep.shape).astype(np.float32)
    ).to(device)
    y = x0_rep.to(device) + sigma * Z

    # Extract features in mini-batches to avoid OOM
    batch_size = 256
    phi_list = []
    for i in range(0, len(y), batch_size):
        phi_list.append(encoder(y[i:i+batch_size]).cpu().numpy())
    Phi = np.concatenate(phi_list, axis=0)           # (N*n_noise, k)
    X0  = x0_rep.reshape(N * n_noise, d).numpy()     # (N*n_noise, d)
    return Phi, X0


# ---------------------------------------------------------------------------
# Conditioning: Options A / B / C
# ---------------------------------------------------------------------------

def build_conditional_features(
    Phi: np.ndarray,
    U: np.ndarray,
    mode: Literal['A', 'B', 'C'] = 'A',
    k_u: int = 64,
    omega: Callable = np.tanh,
    rng: Optional[np.random.Generator] = None,
    _cache: dict = {},
) -> np.ndarray:
    """
    Build conditional feature vector phi^U(y, U) from DNN features and U.

    Parameters
    ----------
    Phi  : (N, k) DNN features
    U    : (N, d_u) conditioning signal (e.g. class label, embedding)
    mode : 'A', 'B', or 'C'
        A: [phi(y); U]
        B: [phi(y); omega(Gamma @ U + b)]   (Gamma random, fixed)
        C: [phi(y); Gamma @ U; phi(y) * (W @ U)]   (+ interaction term)
    k_u  : number of U features for modes B and C
    omega: nonlinearity for mode B
    _cache: internal cache for random projections (avoids re-sampling across calls)

    Returns
    -------
    Phi_U : (N, k + k_u)  or  (N, k + k_u + k) for mode C
    """
    if rng is None:
        rng = np.random.default_rng(0)  # fixed seed for reproducibility of random projections

    N, k    = Phi.shape
    d_u     = U.shape[1]
    cache_key = (mode, k, d_u, k_u)

    if mode == 'A':
        # Simple concatenation
        return np.concatenate([Phi, U], axis=1)  # (N, k + d_u)

    elif mode == 'B':
        # Random nonlinear U features
        if cache_key not in _cache:
            Gamma = rng.standard_normal((k_u, d_u)) / math.sqrt(d_u)
            b     = rng.standard_normal(k_u) * 0.1
            _cache[cache_key] = (Gamma, b)
        Gamma, b = _cache[cache_key]
        U_feat = omega(U @ Gamma.T + b)  # (N, k_u)
        return np.concatenate([Phi, U_feat], axis=1)  # (N, k + k_u)

    elif mode == 'C':
        # Random affine U features + multiplicative interaction with phi
        if cache_key not in _cache:
            Gamma = rng.standard_normal((k_u, d_u)) / math.sqrt(d_u)
            b_G   = rng.standard_normal(k_u) * 0.1
            W     = rng.standard_normal((k, d_u))   / math.sqrt(d_u)
            b_W   = rng.standard_normal(k) * 0.1
            _cache[cache_key] = (Gamma, b_G, W, b_W)
        Gamma, b_G, W, b_W = _cache[cache_key]
        U_affine = U @ Gamma.T + b_G            # (N, k_u)
        scale    = U @ W.T + b_W                # (N, k)  -- FiLM-style modulation
        interact = Phi * scale                  # (N, k)
        return np.concatenate([Phi, U_affine, interact], axis=1)  # (N, k + k_u + k)

    else:
        raise ValueError(f"mode must be 'A', 'B', or 'C', got '{mode}'")


# ---------------------------------------------------------------------------
# MMSE from features (covariance inversion formula)
# ---------------------------------------------------------------------------

def mmse_from_features(
    Phi: np.ndarray,
    X0: np.ndarray,
    lam: float = 1e-4,
) -> dict:
    """
    Compute the optimal linear-readout denoiser loss from features.

    L = Tr(Sigma_p0) - Tr(Cov(x0, phi) (Sigma_phi + lam*I)^{-1} Cov(phi, x0))

    Uses the *dual* form (N x N) which is efficient when k < N, and the
    *primal* form (k x k) otherwise. Picks automatically.

    Parameters
    ----------
    Phi : (N, k)
    X0  : (N, d)
    lam : ridge regularization on Sigma_phi

    Returns
    -------
    dict with: loss, explained_var, r2, Sigma_p0_trace, Cov_x0_phi, Sigma_phi
    """
    N, k = Phi.shape
    N2, d = X0.shape
    assert N == N2

    mu_phi = Phi.mean(0)
    mu_x0  = X0.mean(0)
    Phi_c  = Phi - mu_phi     # (N, k)
    X0_c   = X0  - mu_x0     # (N, d)

    Sigma_p0_trace = float(np.sum(X0_c ** 2) / (N - 1))

    if k <= N:
        # Primal form: (k x k) inversion
        Sigma_phi  = (Phi_c.T @ Phi_c) / (N - 1)         # (k, k)
        Cov_x0_phi = (X0_c.T  @ Phi_c) / (N - 1)         # (d, k)
        A = Cov_x0_phi @ np.linalg.solve(
            Sigma_phi + lam * np.eye(k), Cov_x0_phi.T
        )  # (d, d) -- but we only need trace
        explained = float(np.trace(A))
    else:
        # Dual form: (N x N) kernel trick via Woodbury
        # Tr(Cov Sigma_phi^{-1} Cov^T) = (1/N^2) Tr(X0_c^T Phi_c (Phi_c^T Phi_c/(N-1) + lam I)^{-1} Phi_c^T X0_c)
        # = (1/(N-1)^2) Tr(X0_c^T K_lambda^{-1} X0_c)  where K = Phi_c Phi_c^T / (N-1) + lam I
        K = (Phi_c @ Phi_c.T) / (N - 1) + lam * np.eye(N)  # (N, N)
        M = np.linalg.solve(K, X0_c)                         # (N, d)
        explained = float(np.trace(X0_c.T @ M)) / (N - 1)
        Sigma_phi  = None
        Cov_x0_phi = None

    loss = Sigma_p0_trace - explained
    r2   = explained / Sigma_p0_trace if Sigma_p0_trace > 0 else 0.0

    return dict(
        loss=loss,
        explained_var=explained,
        r2=r2,
        Sigma_p0_trace=Sigma_p0_trace,
        Cov_x0_phi=Cov_x0_phi,
        Sigma_phi=Sigma_phi,
    )


# ---------------------------------------------------------------------------
# Linear (Wiener filter) baseline
# ---------------------------------------------------------------------------

def wiener_filter_loss(
    X0: np.ndarray,
    sigma: float,
    lam: float = 1e-6,
) -> dict:
    """
    Analytic LMMSE (Wiener filter) loss:
        L_linear = Tr(Sigma_p0) - Tr(Sigma_p0 (Sigma_p0 + sigma^2 I + lam I)^{-1} Sigma_p0)
                 = Tr(sigma^2 (Sigma_p0 + sigma^2 I)^{-1} Sigma_p0)

    This is the best achievable MSE with a *linear* denoiser (pixel-space regression).
    Equivalent to using phi(y) = y as features.

    Parameters
    ----------
    X0    : (N, d) clean samples
    sigma : noise level

    Returns
    -------
    dict with: loss, explained_var, r2, eigvals (of Sigma_p0)
    """
    N, d = X0.shape
    X0_c = X0 - X0.mean(0)
    Sigma_p0 = (X0_c.T @ X0_c) / (N - 1)  # (d, d)

    # Eigendecomposition for numerically stable computation
    eigvals = np.linalg.eigvalsh(Sigma_p0)  # (d,) ascending
    # L = sum_i sigma^2 * lambda_i / (lambda_i + sigma^2)
    explained = float(np.sum(eigvals ** 2 / (eigvals + sigma ** 2 + lam)))
    total     = float(np.sum(eigvals))
    loss      = total - explained
    r2        = explained / total if total > 0 else 0.0

    return dict(loss=loss, explained_var=explained, r2=r2, eigvals=eigvals)


def wiener_filter_cond_loss(
    X0: np.ndarray,
    U: np.ndarray,
    sigma: float,
    lam: float = 1e-6,
) -> dict:
    """
    Conditional Wiener filter: optimal linear estimator of x0 from (y, U)
    where y = x0 + sigma*Z.

    Uses phi^U(y, U) = [y ; U] as features, then calls mmse_from_features.
    This gives the conditional LMMSE — the best linear denoiser that has
    access to U at test time.

    Returns dict same as mmse_from_features.
    """
    N, d = X0.shape
    # Simulate noisy observations
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(X0.shape).astype(np.float32)
    Y = X0 + sigma * Z
    Phi_cond = np.concatenate([Y, U], axis=1)   # (N, d + d_u)
    return mmse_from_features(Phi_cond, X0, lam=lam)
