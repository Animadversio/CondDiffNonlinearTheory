"""
Jointly Gaussian (x0, U) case — simplified theoretical formulas.

When (x0, U) are jointly Gaussian, Stein's lemma collapses the
conditional covariance expressions to closed-form quantities.

Key results (Section 3 of the theory notes):
  - alpha_j = c_1(m_tilde_j, S_tilde_j) / S_tilde_j
  - Cov(x0, phi^U)_{:,j} = (Sigma_p0 @ theta_j + C_xU @ gamma_j) * alpha_j
  - A_U = (Sigma_p0 @ Theta^T + C_xU @ Gamma^T) @ diag(alpha)
  - Sigma_phi^U_{ij} = sum_{n>=1} n! * r_tilde_ij^n * c_n(m_tilde_i,S_i) * c_n(m_tilde_j,S_j)
  - L_{sigma,U} = Tr(Sigma_p0) - Tr(A_U @ inv(Sigma_phi^U) @ A_U^T)
"""

import math
import numpy as np
from typing import Optional, Callable

from .hermite import hermite_coeffs_mc, hermite_series_covariance


# ---------------------------------------------------------------------------
# Jointly Gaussian distribution
# ---------------------------------------------------------------------------

class JointGaussian:
    """
    Joint Gaussian distribution over (x0, U):

        [x0]   ~ N([mu_x], [Sigma_p0  C_xU])
        [U ]       [mu_U]   [C_Ux      Sigma_U]

    Parameters
    ----------
    mu_x      : (d,)
    mu_U      : (d_u,)
    Sigma_p0  : (d, d)   data covariance
    Sigma_U   : (d_u, d_u)
    C_xU      : (d, d_u) cross-covariance Cov(x0, U)
    """

    def __init__(
        self,
        mu_x: np.ndarray,
        mu_U: np.ndarray,
        Sigma_p0: np.ndarray,
        Sigma_U: np.ndarray,
        C_xU: np.ndarray,
    ):
        self.mu_x     = np.asarray(mu_x, dtype=float)
        self.mu_U     = np.asarray(mu_U, dtype=float)
        self.Sigma_p0 = np.asarray(Sigma_p0, dtype=float)
        self.Sigma_U  = np.asarray(Sigma_U,  dtype=float)
        self.C_xU     = np.asarray(C_xU,     dtype=float)
        self.d  = len(mu_x)
        self.d_u = len(mu_U)

    def sample(self, N: int, rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw N joint samples (x0, U).

        Returns
        -------
        x0_samples : (N, d)
        U_samples  : (N, d_u)
        """
        if rng is None:
            rng = np.random.default_rng()
        d, d_u = self.d, self.d_u
        mu_joint = np.concatenate([self.mu_x, self.mu_U])
        Sigma_joint = np.block([
            [self.Sigma_p0, self.C_xU],
            [self.C_xU.T,  self.Sigma_U],
        ])
        joint = rng.multivariate_normal(mu_joint, Sigma_joint, size=N)
        return joint[:, :d], joint[:, d:]


# ---------------------------------------------------------------------------
# Per-feature Gaussian parameters (conditional on x0 and U)
# ---------------------------------------------------------------------------

def feature_gaussian_params(
    Theta: np.ndarray,
    Gamma: Optional[np.ndarray],
    epsilon: np.ndarray,
    sigma: float,
    mu_x: np.ndarray,
    mu_U: np.ndarray,
    Sigma_p0: np.ndarray,
    Sigma_U: np.ndarray,
    C_xU: np.ndarray,
) -> dict:
    """
    For feature j, the pre-activation is:
        l_j = theta_j^T x0 + gamma_j^T U + epsilon_j + s_j * xi

    where xi ~ N(0,1) is the noise direction, s_j = sigma * ||theta_j||.

    With (x0, U) jointly Gaussian, l_j is Gaussian with:
        m_tilde_j = theta_j^T mu_x + gamma_j^T mu_U + epsilon_j
        S_tilde_j^2 = [theta_j, gamma_j]^T Sigma_joint [theta_j, gamma_j] + sigma^2 ||theta_j||^2

    Returns dict with arrays of shape (k,):
        m_tilde, S_tilde, s (noise scale)
    and correlation matrix r_tilde of shape (k, k).
    """
    k = Theta.shape[0]
    s = sigma * np.linalg.norm(Theta, axis=1)  # (k,)

    m_tilde = Theta @ mu_x + epsilon  # (k,)
    if Gamma is not None:
        m_tilde = m_tilde + Gamma @ mu_U

    # S_tilde_j^2 = theta_j^T Sigma_p0 theta_j + 2 theta_j^T C_xU gamma_j
    #              + gamma_j^T Sigma_U gamma_j + sigma^2 ||theta_j||^2
    ThetaSigma = Theta @ Sigma_p0  # (k, d)
    S2 = np.einsum('ki,ki->k', ThetaSigma, Theta)  # theta_j^T Sigma_p0 theta_j
    if Gamma is not None:
        S2 += 2 * np.einsum('ki,ki->k', Theta @ C_xU, Gamma)
        GammaSigmaU = Gamma @ Sigma_U
        S2 += np.einsum('ki,ki->k', GammaSigmaU, Gamma)
    S2 += s ** 2
    S_tilde = np.sqrt(np.maximum(S2, 1e-12))  # (k,)

    # Correlation r_tilde_{ij} = Corr(t_i, t_j) where t = (l - m_tilde) / S_tilde
    # Numerator: Cov(l_i, l_j) = theta_i^T Sigma_p0 theta_j + theta_i^T C_xU gamma_j
    #           + gamma_i^T C_xU^T theta_j + gamma_i^T Sigma_U gamma_j + sigma^2 theta_i^T theta_j
    Cov_l = ThetaSigma @ Theta.T  # (k, k)
    if Gamma is not None:
        Cov_l += Theta @ C_xU @ Gamma.T
        Cov_l += Gamma @ C_xU.T @ Theta.T
        Cov_l += Gamma @ Sigma_U @ Gamma.T
    Cov_l += (s[:, None] * s[None, :]) * (Theta @ Theta.T) / (
        np.linalg.norm(Theta, axis=1)[:, None] * np.linalg.norm(Theta, axis=1)[None, :]
        + 1e-12
    )
    r_tilde = Cov_l / (S_tilde[:, None] * S_tilde[None, :])
    r_tilde = np.clip(r_tilde, -1 + 1e-6, 1 - 1e-6)

    return dict(m_tilde=m_tilde, S_tilde=S_tilde, s=s, r_tilde=r_tilde)


# ---------------------------------------------------------------------------
# Theoretical loss under jointly Gaussian assumption (Section 3)
# ---------------------------------------------------------------------------

def gaussian_theoretical_loss(
    omega: Callable,
    Theta: np.ndarray,
    Gamma: Optional[np.ndarray],
    epsilon: np.ndarray,
    sigma: float,
    dist: JointGaussian,
    n_terms: int = 12,
    n_mc: int = 200_000,
    rng: np.random.Generator = None,
) -> dict:
    """
    Compute the closed-form conditional loss L_{sigma,U} and unconditional
    loss L_sigma under the jointly Gaussian assumption.

    Returns dict with keys:
        L_sigma      : unconditional loss
        L_sigma_U    : conditional loss
        gain         : L_sigma - L_sigma_U  (mutual info approximation integrand)
        A_U, A       : slope matrices (d, k)
        Sigma_phi_U  : (k, k) conditional feature covariance
        Sigma_phi    : (k, k) unconditional feature covariance
    """
    if rng is None:
        rng = np.random.default_rng()

    k = Theta.shape[0]
    Sigma_p0 = dist.Sigma_p0
    d = dist.d

    # ---- Conditional (with U) parameters ----
    params_U = feature_gaussian_params(
        Theta, Gamma, epsilon, sigma,
        dist.mu_x, dist.mu_U, Sigma_p0, dist.Sigma_U, dist.C_xU,
    )
    m_tilde = params_U['m_tilde']   # (k,)
    S_tilde = params_U['S_tilde']   # (k,)
    s       = params_U['s']         # (k,)
    r_tilde = params_U['r_tilde']   # (k, k)

    # Hermite coefficients c_n(m_tilde_j, S_tilde_j)
    C_U = hermite_coeffs_batch(omega, m_tilde, S_tilde, n_terms=n_terms,
                                n_mc=n_mc, rng=rng)  # (k, n_terms)

    # alpha_j = c_1(m_tilde_j, S_tilde_j) / S_tilde_j
    alpha_U = C_U[:, 1] / S_tilde  # (k,)

    # Sigma_phi^U: sum_{n>=1} n! r_tilde^n c_n_i c_n_j
    Sigma_phi_U = np.zeros((k, k))
    for n in range(1, n_terms):
        fn = math.factorial(n)
        cn = C_U[:, n]  # (k,)
        Sigma_phi_U += fn * (r_tilde ** n) * np.outer(cn, cn)

    # A_U = (Sigma_p0 Theta^T + C_xU Gamma^T) diag(alpha_U)
    A_U = Sigma_p0 @ Theta.T  # (d, k)
    if Gamma is not None:
        A_U = A_U + dist.C_xU @ Gamma.T
    A_U = A_U * alpha_U[None, :]  # broadcast diag(alpha_U)

    # L_{sigma,U} = Tr(Sigma_p0) - Tr(A_U inv(Sigma_phi_U) A_U^T)
    Sigma_phi_U_inv = np.linalg.pinv(Sigma_phi_U)
    L_U = float(np.trace(Sigma_p0) - np.trace(A_U @ Sigma_phi_U_inv @ A_U.T))

    # ---- Unconditional parameters ----
    # Set Gamma=None, C_xU=0
    zero_C = np.zeros((dist.d, dist.d_u))
    params_0 = feature_gaussian_params(
        Theta, None, epsilon, sigma,
        dist.mu_x, dist.mu_U * 0, Sigma_p0,
        np.eye(dist.d_u), zero_C,
    )
    m0     = params_0['m_tilde']
    S0     = params_0['S_tilde']
    r0     = params_0['r_tilde']

    C_0 = hermite_coeffs_batch(omega, m0, S0, n_terms=n_terms,
                                n_mc=n_mc, rng=rng)  # (k, n_terms)
    alpha_0 = C_0[:, 1] / S0

    Sigma_phi_0 = np.zeros((k, k))
    for n in range(1, n_terms):
        fn = math.factorial(n)
        cn = C_0[:, n]
        Sigma_phi_0 += fn * (r0 ** n) * np.outer(cn, cn)

    A_0 = Sigma_p0 @ Theta.T * alpha_0[None, :]

    Sigma_phi_0_inv = np.linalg.pinv(Sigma_phi_0)
    L_0 = float(np.trace(Sigma_p0) - np.trace(A_0 @ Sigma_phi_0_inv @ A_0.T))

    return dict(
        L_sigma=L_0,
        L_sigma_U=L_U,
        gain=L_0 - L_U,
        A_U=A_U,
        A=A_0,
        Sigma_phi_U=Sigma_phi_U,
        Sigma_phi=Sigma_phi_0,
        alpha_U=alpha_U,
        alpha_0=alpha_0,
    )
