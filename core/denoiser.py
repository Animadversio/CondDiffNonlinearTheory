"""
Random-feature denoiser model and theoretical loss computation.

Implements the random-feature model:
    phi(y) = omega(Theta @ y + epsilon)          [unconditional]
    phi^U(y, U) = omega(Theta @ y + Gamma @ U + epsilon)  [conditional]

And the theoretical/empirical Bayes-optimal denoiser losses:
    L_sigma   = Tr(Sigma_p0 - Cov(x0, phi) @ inv(Sigma_phi) @ Cov(phi, x0))
    L_{sigma,U} = Tr(Sigma_p0 - Cov(x0,phi^U) @ inv(Sigma_phi^U) @ Cov(phi^U,x0))

Reference: Section 1 & 2 of the theory notes.
"""

import numpy as np
from typing import Callable, Optional

from .hermite import (
    hermite_coeffs_batch,
    hermite_series_covariance,
    smoothed_activation_mc,
)


# ---------------------------------------------------------------------------
# Random feature map
# ---------------------------------------------------------------------------

class RandomFeatureMap:
    """
    Conditional random feature map:
        phi^U(y, U) = omega(Theta @ y + Gamma @ U + epsilon)

    For unconditional use, set Gamma=None or U=0.

    Parameters
    ----------
    Theta   : (k, d) random projection matrix (fixed, not learned)
    omega   : nonlinearity, e.g. np.tanh, sigmoid
    Gamma   : (k, d_u) conditioning projection matrix (optional)
    epsilon : (k,) random bias
    """

    def __init__(
        self,
        Theta: np.ndarray,
        omega: Callable,
        Gamma: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
    ):
        self.Theta = Theta        # (k, d)
        self.omega = omega
        self.Gamma = Gamma        # (k, d_u) or None
        self.epsilon = epsilon if epsilon is not None else np.zeros(Theta.shape[0])
        self.k, self.d = Theta.shape

    def __call__(self, y: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        """
        y : (..., d) or (d,)
        U : (..., d_u) or (d_u,), or None
        Returns phi^U(y, U) of shape (..., k)
        """
        pre = y @ self.Theta.T + self.epsilon  # (..., k)
        if self.Gamma is not None and U is not None:
            pre = pre + U @ self.Gamma.T
        return self.omega(pre)

    def pre_activation(self, y: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Return pre-activation Theta @ y + Gamma @ U + epsilon."""
        pre = y @ self.Theta.T + self.epsilon
        if self.Gamma is not None and U is not None:
            pre = pre + U @ self.Gamma.T
        return pre

    def feature_params(self, sigma: float):
        """
        Per-feature scale s_j = sigma * ||theta_j|| for each row j.
        Returns s : (k,)
        """
        return sigma * np.linalg.norm(self.Theta, axis=1)

    def correlation_matrix(self) -> np.ndarray:
        """
        Correlation rho_{ij} = theta_i^T theta_j / (||theta_i|| ||theta_j||)
        Returns (k, k) matrix.
        """
        norms = np.linalg.norm(self.Theta, axis=1, keepdims=True)  # (k, 1)
        Theta_norm = self.Theta / norms
        return Theta_norm @ Theta_norm.T  # (k, k)


# ---------------------------------------------------------------------------
# Empirical loss (train optimal linear head, measure MSE)
# ---------------------------------------------------------------------------

def fit_optimal_denoiser(
    phi_samples: np.ndarray,
    x0_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit optimal linear denoiser: D(phi) = W @ phi + b
    minimizing E[||W phi + b - x0||^2].

    Closed-form solution:
        W* = Cov(x0, phi) @ inv(Sigma_phi)
        b* = mu_p0 - W* @ mu_phi

    Parameters
    ----------
    phi_samples : (N, k) feature samples
    x0_samples  : (N, d) clean data samples

    Returns
    -------
    W : (d, k)
    b : (d,)
    """
    mu_phi = phi_samples.mean(axis=0)       # (k,)
    mu_x0  = x0_samples.mean(axis=0)       # (d,)

    phi_c = phi_samples - mu_phi            # (N, k)
    x0_c  = x0_samples - mu_x0             # (N, d)

    Sigma_phi   = (phi_c.T @ phi_c) / (len(phi_samples) - 1)  # (k, k)
    Cov_x0_phi  = (x0_c.T @ phi_c)  / (len(phi_samples) - 1)  # (d, k)

    W = Cov_x0_phi @ np.linalg.pinv(Sigma_phi)  # (d, k)
    b = mu_x0 - W @ mu_phi                       # (d,)
    return W, b


def empirical_loss(
    phi_map: RandomFeatureMap,
    x0_samples: np.ndarray,
    sigma: float,
    U_samples: Optional[np.ndarray] = None,
    n_noise: int = 1,
    rng: np.random.Generator = None,
) -> float:
    """
    Estimate the optimal denoiser MSE loss empirically.

    For each x0 (and U if conditional), generate noisy observation y = x0 + sigma*Z,
    compute phi^U(y, U), fit optimal linear head, return average MSE.

    Parameters
    ----------
    phi_map    : RandomFeatureMap
    x0_samples : (N, d)
    sigma      : noise level
    U_samples  : (N, d_u) or None for unconditional
    n_noise    : number of noise realizations per x0 (for variance reduction)
    rng        : numpy Generator

    Returns
    -------
    loss : scalar MSE
    """
    if rng is None:
        rng = np.random.default_rng()
    N, d = x0_samples.shape

    # generate noisy observations
    Z = rng.standard_normal((N * n_noise, d))
    x0_rep = np.repeat(x0_samples, n_noise, axis=0)          # (N*n_noise, d)
    U_rep  = np.repeat(U_samples,  n_noise, axis=0) if U_samples is not None else None
    y = x0_rep + sigma * Z

    phi_samples = phi_map(y, U_rep)  # (N*n_noise, k)

    W, b = fit_optimal_denoiser(phi_samples, x0_rep)

    # compute MSE
    preds = phi_samples @ W.T + b    # (N*n_noise, d)
    return float(np.mean((preds - x0_rep) ** 2))


# ---------------------------------------------------------------------------
# Theoretical loss via covariance formula
# ---------------------------------------------------------------------------

def theoretical_loss_from_cov(
    Sigma_p0: np.ndarray,
    Cov_x0_phi: np.ndarray,
    Sigma_phi: np.ndarray,
) -> float:
    """
    L = Tr(Sigma_p0) - Tr(Cov(x0,phi) @ inv(Sigma_phi) @ Cov(phi,x0))

    Parameters
    ----------
    Sigma_p0    : (d, d) data covariance
    Cov_x0_phi  : (d, k) cross-covariance
    Sigma_phi   : (k, k) feature covariance

    Returns
    -------
    loss : scalar
    """
    Sigma_phi_inv = np.linalg.pinv(Sigma_phi)
    explained = Cov_x0_phi @ Sigma_phi_inv @ Cov_x0_phi.T  # (d, d)
    return float(np.trace(Sigma_p0) - np.trace(explained))


def empirical_covariances(
    phi_map: RandomFeatureMap,
    x0_samples: np.ndarray,
    sigma: float,
    U_samples: Optional[np.ndarray] = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    Estimate Sigma_p0, Sigma_phi, Cov(x0, phi) from samples.

    Returns dict with keys: Sigma_p0, Sigma_phi, Cov_x0_phi
    """
    if rng is None:
        rng = np.random.default_rng()
    N, d = x0_samples.shape
    Z = rng.standard_normal((N, d))
    y = x0_samples + sigma * Z

    phi_samples = phi_map(y, U_samples)  # (N, k)

    mu_x0  = x0_samples.mean(0)
    mu_phi = phi_samples.mean(0)
    x0_c   = x0_samples - mu_x0
    phi_c  = phi_samples - mu_phi

    Sigma_p0   = (x0_c.T  @ x0_c)  / (N - 1)
    Sigma_phi  = (phi_c.T @ phi_c) / (N - 1)
    Cov_x0_phi = (x0_c.T  @ phi_c) / (N - 1)

    return dict(Sigma_p0=Sigma_p0, Sigma_phi=Sigma_phi, Cov_x0_phi=Cov_x0_phi)


# ---------------------------------------------------------------------------
# Theoretical covariances via Hermite expansion (general case)
# ---------------------------------------------------------------------------

def theoretical_Sigma_phi(
    phi_map: RandomFeatureMap,
    x0_samples: np.ndarray,
    sigma: float,
    U_samples: Optional[np.ndarray] = None,
    n_terms: int = 10,
    n_mc: int = 100_000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Compute Sigma_phi^U using eq. 2.1 (general case):
        Sigma_phi_{ij} = Cov_{x0,U}(g_i^U, g_j^U)
                       + E_{x0,U}[ sum_{n>=1} n! rho_ij^n c_n(M_i^U, s_i) c_n(M_j^U, s_j) ]

    Evaluated by MC over x0 (and U) samples.

    Parameters
    ----------
    x0_samples : (N, d)
    U_samples  : (N, d_u) or None

    Returns
    -------
    Sigma_phi : (k, k)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(x0_samples)
    k = phi_map.k
    s = phi_map.feature_params(sigma)  # (k,)
    rho = phi_map.correlation_matrix()  # (k, k)

    # For each x0 (and U), compute g_j^U(x0) = c_0(M_j^U(x0), s_j) and c_n coefficients
    # M_j^U(x0) = pre-activation at zero noise = Theta @ x0 + Gamma @ U + epsilon
    M = phi_map.pre_activation(x0_samples, U_samples)  # (N, k)

    # g values: g_j(x0) = omega convolved with N(0, s_j^2), evaluated at M_j
    # Approximated as c_0(M_j, s_j) via MC
    g = np.zeros((N, k))
    for j in range(k):
        xi = rng.standard_normal(n_mc)
        g[:, j] = np.mean(phi_map.omega(M[:, [j]] + s[j] * xi), axis=1)  # (N,)

    # Term 1: Cov_{x0,U}(g_i, g_j)
    g_c = g - g.mean(0)
    Sigma_term1 = (g_c.T @ g_c) / (N - 1)  # (k, k)

    # Term 2: E[ sum_{n>=1} n! rho_ij^n c_n(M_i, s_i) c_n(M_j, s_j) ]
    # Compute c_n(M_j(x0), s_j) for each x0 and feature j
    # Shape: (N, k, n_terms)
    # For memory efficiency, compute term2 incrementally
    Sigma_term2 = np.zeros((k, k))
    xi_shared = rng.standard_normal(n_mc)
    He_all = None  # will be computed lazily

    import math
    # Precompute He_n(xi) once
    from .hermite import hermite_poly_all
    He_all = hermite_poly_all(n_terms - 1, xi_shared)  # (n_terms, n_mc)

    # Compute c_n(M_j(x_0), s_j) for all x_0 and j
    # c_n matrix: (N, k, n_terms)
    C = np.zeros((N, k, n_terms))
    for j in range(k):
        f_vals = phi_map.omega(M[:, [j]] + s[j] * xi_shared)  # (N, n_mc)
        for n in range(1, n_terms):
            C[:, j, n] = np.mean(f_vals * He_all[n], axis=1) / math.factorial(n)

    # Term2_{ij} = E_{x0}[ sum_{n>=1} n! rho_ij^n c_n_i c_n_j ]
    for n in range(1, n_terms):
        factor = math.factorial(n) * (rho ** n)  # (k, k)
        # outer product over x0: mean_x0[ c_n_i(x0) * c_n_j(x0) ]
        cn_i = C[:, :, n]  # (N, k)
        cn_j = C[:, :, n]  # (N, k)
        Sigma_term2 += factor * (cn_i.T @ cn_j) / N

    return Sigma_term1 + Sigma_term2


def theoretical_Cov_x0_phi(
    phi_map: RandomFeatureMap,
    x0_samples: np.ndarray,
    sigma: float,
    mu_p0: np.ndarray,
    U_samples: Optional[np.ndarray] = None,
    n_mc: int = 100_000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Compute Cov(x0, phi^U) via:
        Cov(x0, phi^U)_{ij} = E_{x0,U}[(x0_i - mu_{p0,i}) * g_j^U(x0)]

    Parameters
    ----------
    mu_p0 : (d,) data mean

    Returns
    -------
    Cov_x0_phi : (d, k)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, d = x0_samples.shape
    k = phi_map.k
    s = phi_map.feature_params(sigma)  # (k,)

    M = phi_map.pre_activation(x0_samples, U_samples)  # (N, k)

    # g_j^U(x0) for each x0
    g = np.zeros((N, k))
    for j in range(k):
        xi = rng.standard_normal(n_mc)
        g[:, j] = np.mean(phi_map.omega(M[:, [j]] + s[j] * xi), axis=1)

    x0_c = x0_samples - mu_p0  # (N, d)
    # Cov_{ij} = E[(x0_i - mu_i) * g_j] = mean over x0
    Cov_x0_phi = (x0_c.T @ g) / N  # (d, k)
    return Cov_x0_phi
