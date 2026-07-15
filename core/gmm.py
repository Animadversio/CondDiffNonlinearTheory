"""
Gaussian Mixture Model (GMM) target distribution for RF denoiser theory.

Supports:
- Sampling from GMM with optional label noise
- Exact MMSE via posterior-weighted per-component Wiener filters (Monte Carlo)
- Conditional Wiener MMSE (per-class analytic, exact for Gaussian components)
- Unconditional linear Wiener MMSE (from mixture covariance)
- Closed-form Stein formula for Cov(x0, phi^U) with ReLU nonlinearity

Reference: docs/newfile2_article.pdf §1 — per-component Stein's lemma,
           weighted by mixture coefficients.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from typing import Optional


# ---------------------------------------------------------------------------
# ReLU closed-form helpers
# ---------------------------------------------------------------------------

def _c0(M: np.ndarray, s: np.ndarray) -> np.ndarray:
    """c0(M, s) = M Phi(z) + s phi(z)  where z=M/s.  Gaussian-smoothed ReLU."""
    s_safe = np.maximum(s, 1e-12)
    z = M / s_safe
    return M * norm.cdf(z) + s_safe * norm.pdf(z)


def _c1(M: np.ndarray, s: np.ndarray) -> np.ndarray:
    """c1(M, s) = s Phi(z)."""
    z = M / np.maximum(s, 1e-12)
    return s * norm.cdf(z)


def _c2(M: np.ndarray, s: np.ndarray) -> np.ndarray:
    """c2(M, s) = s phi(z) / 2."""
    z = M / np.maximum(s, 1e-12)
    return s * norm.pdf(z) / 2.0


def _c3(M: np.ndarray, s: np.ndarray) -> np.ndarray:
    """c3(M, s) = -M phi(z) / 6.  Derivation: c_n = (1/n!) E[relu(M+sz) He_n(z)];
    for n=3, integrating by parts gives E[relu(M+sz)(z^3-3z)] = -M phi(M/s),
    so c3 = -M phi(M/s) / 6.  Sign matters for odd cross-terms in Hermite expansions."""
    z = M / np.maximum(s, 1e-12)
    return -M * norm.pdf(z) / 6.0


# ---------------------------------------------------------------------------
# GaussianMixture class
# ---------------------------------------------------------------------------

class GaussianMixture:
    """
    GMM: p0(x0) = sum_c w_c N(x0; mu_c, Sigma_c),  c in {0, ..., C-1}.

    Parameters
    ----------
    weights : (C,) mixture weights, must sum to 1
    means   : (C, d) component means
    covs    : (C, d, d) component covariances (positive definite)
    """

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
    ):
        self.weights = np.asarray(weights, float)
        self.means = np.asarray(means, float)   # (C, d)
        self.covs = np.asarray(covs, float)     # (C, d, d)
        self.C = len(self.weights)
        self.d = self.means.shape[1]

        # Global mean
        self.mu = (self.weights[:, None] * self.means).sum(0)  # (d,)

        # Global covariance (mixture covariance formula)
        self.Sigma = sum(
            w * (S + np.outer(m - self.mu, m - self.mu))
            for w, m, S in zip(self.weights, self.means, self.covs)
        )  # (d, d)

        # Per-class eigenvalues (for analytic conditional Wiener)
        self._eigvals = [np.linalg.eigvalsh(S) for S in self.covs]  # each (d,)

        # Population moments for jointly-Gaussian (x0, U) theory — Section 3
        # mu_U = E[U] = class weights
        self.mu_U = self.weights.copy()                # (C,)
        # Sigma_U = Cov(U, U) = diag(w) - w w^T  (multinomial covariance)
        self.Sigma_U = np.diag(self.weights) - np.outer(self.weights, self.weights)  # (C, C)
        # C_xU = Cov(x0, U) = sum_c w_c (mu_c - mu) (e_c - w)^T
        self.C_xU = sum(
            w * np.outer(m - self.mu, np.eye(self.C)[c] - self.weights)
            for c, (w, m) in enumerate(zip(self.weights, self.means))
        )  # (d, C)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        N: int,
        label_noise: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """
        Sample (x0, labels, U) from the GMM.

        U_c = (1 - label_noise) * 1(label==c) + label_noise / C
        (pure one-hot when label_noise=0).

        Returns
        -------
        x0     : (N, d)
        labels : (N,) int
        U      : (N, C) soft one-hot class conditioning vector
        """
        if rng is None:
            rng = np.random.default_rng()
        labels = rng.choice(self.C, size=N, p=self.weights)
        x0 = np.zeros((N, self.d))
        for c in range(self.C):
            mask = labels == c
            nc = int(mask.sum())
            if nc > 0:
                x0[mask] = rng.multivariate_normal(self.means[c], self.covs[c], size=nc)
        U = np.eye(self.C)[labels]   # hard one-hot by default
        if label_noise > 0.0:
            U = (1.0 - label_noise) * U + label_noise / self.C
        return x0, labels, U

    # ------------------------------------------------------------------
    # Baseline MMSE curves (closed-form / Monte Carlo)
    # ------------------------------------------------------------------

    def mmse_uncond_wiener(self, sigma: float) -> float:
        """Unconditional linear Wiener MMSE using mixture covariance eigenvalues."""
        eigvals = np.linalg.eigvalsh(self.Sigma)
        sig2 = sigma ** 2
        return float((sig2 * eigvals / (eigvals + sig2)).sum())

    def mmse_cond_wiener(self, sigma: float) -> float:
        """
        Per-class Wiener MMSE averaged over mixture weights.
        Exact for Gaussian components:
          MMSE_c = sum_i sigma^2 lambda_{c,i} / (lambda_{c,i} + sigma^2)
        """
        sig2 = sigma ** 2
        return float(sum(
            w * float((sig2 * ev / (ev + sig2)).sum())
            for w, ev in zip(self.weights, self._eigvals)
        ))

    # mmse_cond_exact == mmse_cond_wiener for Gaussian components
    mmse_cond_exact = mmse_cond_wiener

    def mmse_uncond_exact(
        self,
        sigma: float,
        N_mc: int = 200_000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Exact unconditional MMSE via Monte Carlo.

        Posterior denoiser: D*(y) = sum_c pi_c(y) D_c(y)
        where pi_c(y) = P(c|y) (posterior) and
              D_c(y)  = mu_c + Sigma_c (Sigma_c + sigma^2 I)^{-1} (y - mu_c)
        """
        if rng is None:
            rng = np.random.default_rng()
        x0, labels, _ = self.sample(N_mc, rng=rng)
        y = x0 + sigma * rng.standard_normal(x0.shape)

        I = np.eye(self.d)
        # Precompute per-class Wiener gains W_c = Sigma_c (Sigma_c + sigma^2 I)^{-1}
        gains = [
            self.covs[c] @ np.linalg.solve(self.covs[c] + sigma**2 * I, I)
            for c in range(self.C)
        ]

        # Compute log p(y|c) for all c
        log_lik = np.stack([
            multivariate_normal.logpdf(y, mean=self.means[c],
                                        cov=self.covs[c] + sigma**2 * I)
            for c in range(self.C)
        ], axis=1)  # (N_mc, C)

        log_post = np.log(self.weights + 1e-300)[None, :] + log_lik
        log_post -= log_post.max(1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(1, keepdims=True)  # (N_mc, C)

        D_star = np.zeros_like(x0)
        for c, W_c in enumerate(gains):
            D_c = self.means[c] + (y - self.means[c]) @ W_c.T  # (N_mc, d)
            D_star += post[:, c:c+1] * D_c

        return float(np.mean(np.sum((D_star - x0) ** 2, axis=1)))

    # ------------------------------------------------------------------
    # Closed-form Stein formula for Cov(x0, phi^U) — ReLU nonlinearity
    # ------------------------------------------------------------------

    def cov_x0_phi_stein_relu(
        self,
        Theta: np.ndarray,
        Gamma: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Closed-form Cov(x0, phi^U)_{ij} via per-component Stein's lemma.

        For component c, x0 | c ~ N(mu_c, Sigma_c), U = e_c:
          pre-activation mean:  mbar_{j,c} = theta_j^T mu_c + Gamma[j,c]
          x0 spread:            v_{j,c} = sqrt(theta_j^T Sigma_c theta_j)
          noise scale:          s_j = sigma * ||theta_j||
          total spread:         s_tilde_{j,c} = sqrt(s_j^2 + v_{j,c}^2)
          std-pre-activation:   z_tilde_{j,c} = mbar_{j,c} / s_tilde_{j,c}

        Stein's lemma (per component, ReLU derivative c0'(M,s) = Phi(M/s)):
          E_{x0|c}[(x0_i - mu_{c,i}) g_j^c(x0)] = (Sigma_c theta_j)_i Phi(z_tilde_{j,c})

        Full formula (weighted mixture, using global mean mu):
          Cov(x0_i, phi^U_j) = sum_c w_c [
            (Sigma_c @ theta_j)_i * Phi(z_tilde_{j,c})
            + (mu_{c,i} - mu_i) * c0(mbar_{j,c}, s_tilde_{j,c})
          ]

        Parameters
        ----------
        Theta : (k, d) random projection matrix
        Gamma : (k, C) conditioning projection matrix
        sigma : noise level

        Returns
        -------
        Cov : (d, k)
        """
        s2 = (sigma * np.linalg.norm(Theta, axis=1)) ** 2  # (k,)

        Cov = np.zeros((self.d, Theta.shape[0]))
        for c in range(self.C):
            wc = self.weights[c]
            mbar = Theta @ self.means[c] + Gamma[:, c]            # (k,)
            Sig_c_Theta = self.covs[c] @ Theta.T                  # (d, k)
            v2 = np.einsum('ji,ji->i', Theta.T, Sig_c_Theta)      # (k,) = theta_j^T Sigma_c theta_j
            s_tilde = np.sqrt(s2 + v2)                            # (k,)
            z_tilde = mbar / np.maximum(s_tilde, 1e-12)           # (k,)
            Phi_z   = norm.cdf(z_tilde)                           # (k,)
            g_tilde = _c0(mbar, s_tilde)                          # (k,) = E[g_j | c]
            delta_mu = (self.means[c] - self.mu)[:, None]         # (d, 1)
            Cov += wc * (Sig_c_Theta * Phi_z[None, :] + delta_mu * g_tilde[None, :])

        return Cov  # (d, k)

    def sigma_phi_noise_relu(
        self,
        Theta: np.ndarray,
        Gamma: np.ndarray,
        sigma: float,
        x0: np.ndarray,
        U: np.ndarray,
        n_terms: int = 3,
    ) -> np.ndarray:
        """
        Hermite noise contribution to Sigma_phi using GMM data samples.

        E_{(x0,U)}[sum_{n=1}^{n_terms} n! rho^n c_n_i c_n_j]

        Uses the provided (x0, U) samples to estimate E[c_n_i c_n_j].

        Parameters
        ----------
        Theta  : (k, d)
        Gamma  : (k, C)
        sigma  : noise level
        x0     : (N, d) samples from GMM
        U      : (N, C) one-hot class labels
        n_terms: max Hermite order (1, 2, or 3)

        Returns
        -------
        Sig_noise : (k, k)
        """
        k = Theta.shape[0]
        N = x0.shape[0]
        s = sigma * np.linalg.norm(Theta, axis=1)  # (k,)

        M = x0 @ Theta.T + U @ Gamma.T             # (N, k)
        z = M / np.maximum(s[None, :], 1e-12)

        # Hermite coefficients per sample
        C1 = s[None, :] * norm.cdf(z)              # (N, k)
        C2 = s[None, :] * norm.pdf(z) / 2.0        # (N, k)

        # Noise-only correlation rho_ij = theta_i^T theta_j / (||theta_i|| ||theta_j||)
        norms = np.linalg.norm(Theta, axis=1)       # (k,)
        Theta_n = Theta / norms[:, None]            # (k, d) normalized
        rho = Theta_n @ Theta_n.T                   # (k, k)
        rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)

        Sig_noise = rho * (C1.T @ C1 / N) + 2.0 * rho**2 * (C2.T @ C2 / N)

        if n_terms >= 3:
            C3 = -M * norm.pdf(z) / 6.0   # negative sign: c3 = -m phi(m/s) / 6
            Sig_noise += 6.0 * rho**3 * (C3.T @ C3 / N)

        return Sig_noise  # (k, k)


# ---------------------------------------------------------------------------
# Jointly Gaussian (x0, U) theory — Section 3 of docs/newfile2_article.pdf
# ---------------------------------------------------------------------------

def mmse_theory_joint_gaussian(
    Sigma_p0: np.ndarray,
    mu_x0: np.ndarray,
    Theta: np.ndarray,
    Gamma: np.ndarray,
    sigma: float,
    C_xU: Optional[np.ndarray] = None,
    Sigma_U: Optional[np.ndarray] = None,
    mu_U: Optional[np.ndarray] = None,
    lam: float = 1e-4,
    n_terms: int = 3,
) -> float:
    """
    RF theory MMSE using the jointly Gaussian (x0, U) formulas from Section 3
    of docs/newfile2_article.pdf.  Only requires second-order statistics;
    no knowledge of individual GMM components is needed.

    For each feature j (theta_j, gamma_j):
      m̃_j  = theta_j^T mu_x0 + gamma_j^T mu_U
      S̃_j² = theta_j^T Σ_p0 theta_j + 2 theta_j^T C_xU gamma_j
              + gamma_j^T Σ_U gamma_j + σ² ||theta_j||²
      α_j   = c1(m̃_j, S̃_j) / S̃_j

    Cov(x0, φ^U)_j = (Σ_p0 theta_j + C_xU gamma_j) · α_j       — (d, k)

    r̃_ij = [theta_i^T Σ_p0 theta_j + theta_i^T C_xU gamma_j
             + gamma_i^T C_Ux theta_j + gamma_i^T Σ_U gamma_j
             + σ² theta_i^T theta_j] / (S̃_i S̃_j)

    Σ^U_{φ,ij} = Σ_{n=1}^{n_terms} n! r̃_ij^n c_n(m̃_i, S̃_i) c_n(m̃_j, S̃_j)

    MMSE = Tr(Σ_p0) - Tr(Cov(x0,φ^U) [Σ^U_φ + λI]^{-1} Cov(x0,φ^U)^T)

    For the unconditional case, pass Gamma=zeros (or C_xU=None).

    Parameters
    ----------
    Sigma_p0 : (d, d)  data covariance
    mu_x0    : (d,)    data mean
    Theta    : (k, d)  random projections
    Gamma    : (k, C)  label projections (pass zeros for unconditional)
    sigma    : float   noise level
    C_xU     : (d, C)  cross-covariance E[(x0-mu)(U-mu_U)^T], or None
    Sigma_U  : (C, C)  covariance of U, or None
    mu_U     : (C,)    mean of U, or None
    lam      : float   ridge regularisation
    n_terms  : int     Hermite truncation (1, 2, or 3)

    Returns
    -------
    float  MMSE value
    """
    k = Theta.shape[0]
    trace_p0 = float(np.trace(Sigma_p0))
    conditional = (C_xU is not None) and (Gamma is not None) and (mu_U is not None)

    # --- Pre-activation means m̃_j ---
    M_tilde = Theta @ mu_x0                           # (k,)
    if conditional:
        M_tilde = M_tilde + Gamma @ mu_U              # (k,)

    # --- Pre-activation total variances S̃_j² ---
    theta_norms_sq = np.sum(Theta ** 2, axis=1)       # (k,)
    ThSigTh = np.einsum('ki,ij,kj->k', Theta, Sigma_p0, Theta)  # (k,)
    S_sq = ThSigTh + sigma ** 2 * theta_norms_sq      # (k,)
    if conditional:
        ThCG = np.einsum('ki,ij,kj->k', Theta, C_xU, Gamma)    # (k,) diag(Θ C_xU Γ^T)
        GaSuG = np.einsum('ki,ij,kj->k', Gamma, Sigma_U, Gamma)  # (k,)
        S_sq = S_sq + 2.0 * ThCG + GaSuG
    S_tilde = np.sqrt(np.maximum(S_sq, 1e-24))        # (k,)

    # --- α_j = c1(m̃_j, S̃_j) / S̃_j ---
    alpha = _c1(M_tilde, S_tilde) / np.maximum(S_tilde, 1e-12)  # (k,)

    # --- Cov(x0, φ^U) = (Σ_p0 Θ^T + C_xU Γ^T) diag(α)  — (d, k) ---
    Cov = Sigma_p0 @ Theta.T                          # (d, k)
    if conditional:
        Cov = Cov + C_xU @ Gamma.T                    # (d, k)
    Cov = Cov * alpha[None, :]                        # broadcast α over rows

    # --- Joint correlation r̃_ij ---
    Num = Theta @ Sigma_p0 @ Theta.T + sigma ** 2 * (Theta @ Theta.T)  # (k, k)
    if conditional:
        ThCGam = Theta @ C_xU @ Gamma.T               # (k, k)
        Num = Num + ThCGam + ThCGam.T + Gamma @ Sigma_U @ Gamma.T
    r_tilde = Num / np.outer(
        np.maximum(S_tilde, 1e-12),
        np.maximum(S_tilde, 1e-12),
    )
    r_tilde = np.clip(r_tilde, -1 + 1e-6, 1 - 1e-6)

    # --- Σ^U_{φ,ij} = Σ_{n≥1} n! r̃^n ⊙ outer(c_n_i, c_n_j) ---
    C1 = _c1(M_tilde, S_tilde)   # (k,)
    C2 = _c2(M_tilde, S_tilde)   # (k,)
    Sigma_phi = (1.0 * r_tilde        * np.outer(C1, C1)
               + 2.0 * r_tilde ** 2   * np.outer(C2, C2))
    if n_terms >= 3:
        C3 = _c3(M_tilde, S_tilde)   # (k,)
        Sigma_phi = Sigma_phi + 6.0 * r_tilde ** 3 * np.outer(C3, C3)
    Sigma_phi = Sigma_phi + lam * np.eye(k)

    # --- MMSE = Tr(Σ_p0) - Tr(Cov Σ_φ^{-1} Cov^T) ---
    try:
        A = np.linalg.solve(Sigma_phi, Cov.T)         # (k, d)
        explained = float(np.trace(Cov @ A))
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(Sigma_phi, Cov.T, rcond=None)[0]
        explained = float(np.trace(Cov @ A))

    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Correct per-component GMM population theory — replaces mmse_theory_joint_gaussian
# ---------------------------------------------------------------------------

def precompute_gmm_pop(gmm: 'GaussianMixture', Theta: np.ndarray) -> dict:
    """
    Precompute the sigma-independent (k,k) matrices for mmse_theory_gmm_pop.
    Call once per (k, Theta) and reuse across all sigma values to avoid
    recomputing the expensive Theta @ covs[c] @ Theta.T at large k.

    Returns a dict to pass as _precomp=... to mmse_theory_gmm_pop.
    """
    ThTh = Theta @ Theta.T                                         # (k, k)
    ThSigTh_c = [(Theta @ gmm.covs[c]) @ Theta.T for c in range(gmm.C)]
    return {'ThTh': ThTh, 'ThSigTh_c': ThSigTh_c}


def mmse_theory_gmm_pop(
    gmm: 'GaussianMixture',
    Theta: np.ndarray,
    Gamma: np.ndarray,
    sigma: float,
    lam: float = 1e-4,
    n_terms: int = 3,
    conditional: bool = True,
    _precomp: dict = None,
) -> float:
    """
    Population RF-linear MMSE for a GMM via per-component Stein/Hermite.

    CORRECT for GMM where U = e_c is CONSTANT within each component (one-hot,
    NOT jointly Gaussian).  Replaces mmse_theory_joint_gaussian for population
    (N→∞) theory.

    Within component c, the joint pre-activation (a_i, a_j) at any two features
    i, j is JOINTLY GAUSSIAN (x0 ~ N(mu_c, Sigma_c), z ~ N(0, sigma^2 I)):

      mbar_{cj} = theta_j^T mu_c + Gamma[j,c]   (label offset fixed per class)
      S_{cj}^2  = theta_j^T Sigma_c theta_j + sigma^2 ||theta_j||^2
      alpha_{cj} = Phi(mbar_{cj} / S_{cj})        (Stein coefficient = c1/S)
      c0_{cj}   = c0(mbar_{cj}, S_{cj})            (E[phi_j | component c])

    Cov(x0, phi_j) = sum_c w_c [Sigma_c theta_j * alpha_{cj}
                                 + (mu_c - mu) * c0_{cj}]           (d,)

    Sigma_phi = sum_c w_c [outer(c0_c - mu_phi, c0_c - mu_phi)   ← between-component
                           + Var_{x0~c,z}[phi | c]]              ← within-component

    Within-component covariance (fully analytic):
      r_{ij,c} = (theta_i^T Sigma_c theta_j + sigma^2 theta_i^T theta_j) / (S_{ci} S_{cj})
      Cov[phi_i,phi_j|c] = sum_{n=1}^{n_terms} n! r_{ij,c}^n c_n(mbar_ci,S_ci) c_n(mbar_cj,S_cj)
      Diagonal (exact, not truncated):
        Var[phi_j|c] = E[relu^2] - c0^2
                     = (mbar_{cj}^2 + S_{cj}^2) Phi_{cj} + mbar_{cj} S_{cj} phi_{cj} - c0_{cj}^2

    Parameters
    ----------
    gmm        : GaussianMixture instance (weights, means, covs)
    Theta      : (k, d) random projections
    Gamma      : (k, C) label projections
    sigma      : float noise level
    lam        : float ridge regularization
    n_terms    : int   Hermite truncation order for off-diagonal (1, 2, or 3)
    conditional: bool  if False, ignore Gamma (unconditional denoiser)
    """
    k, d = Theta.shape
    C = gmm.C
    s  = sigma * np.linalg.norm(Theta, axis=1)   # (k,)
    s2 = s ** 2

    mu = gmm.mu                    # (d,) global mean
    trace_p0 = float(np.trace(gmm.Sigma))

    # ---- Per-component pre-computations ----
    mbar_c  = np.zeros((C, k))   # pre-act mean per (component, feature)
    S_c     = np.zeros((C, k))   # pre-act total std
    Phi_c   = np.zeros((C, k))
    phi_c_  = np.zeros((C, k))
    c0_c    = np.zeros((C, k))   # E[phi | component c]

    for c in range(C):
        gamma_c   = Gamma[:, c] if conditional else np.zeros(k)
        mbar_c[c] = Theta @ gmm.means[c] + gamma_c
        v2_c      = np.einsum('ki,ij,kj->k', Theta, gmm.covs[c], Theta)
        S_c[c]    = np.sqrt(np.maximum(v2_c + s2, 1e-24))
        z_c       = mbar_c[c] / np.maximum(S_c[c], 1e-12)
        Phi_c[c]  = norm.cdf(z_c)
        phi_c_[c] = norm.pdf(z_c)
        c0_c[c]   = mbar_c[c] * Phi_c[c] + S_c[c] * phi_c_[c]

    mu_phi = (gmm.weights[:, None] * c0_c).sum(0)   # (k,) global expected activation

    # ---- Cov(x0, phi) ---- (d, k)
    Cov = np.zeros((d, k))
    for c in range(C):
        Sig_c_Th = gmm.covs[c] @ Theta.T                          # (d, k)
        delta_mu = (gmm.means[c] - mu)[:, None]                   # (d, 1)
        Cov += gmm.weights[c] * (Sig_c_Th * Phi_c[c][None, :]
                                 + delta_mu * c0_c[c][None, :])

    # ---- Sigma_phi ---- (k, k)
    # Between-component term first (O(C*k) only)
    dg = c0_c - mu_phi[None, :]                                    # (C, k)
    Sigma_phi = np.einsum('c,ci,cj->ij', gmm.weights, dg, dg)     # (k, k)

    # ThTh and ThSigTh_c are sigma-independent; accept precomputed versions
    # (the driver caches them across sigma values to avoid recomputing at large k).
    ThTh = _precomp.get('ThTh') if _precomp else None
    ThSigTh_c = _precomp.get('ThSigTh_c') if _precomp else None
    if ThTh is None:
        ThTh = Theta @ Theta.T                                     # (k, k)
    if ThSigTh_c is None:
        ThSigTh_c = [(Theta @ gmm.covs[c]) @ Theta.T for c in range(C)]  # list of (k,k)

    # Within-component: loop over C to avoid materialising (C, k, k) arrays.
    # Each iteration holds at most ~4 × (k, k) matrices in memory (r, r^2, r^3, outer).
    C1_c = S_c * Phi_c                              # (C, k)
    C2_c = S_c * phi_c_ / 2.0                       # (C, k)
    C3_c = -mbar_c * phi_c_ / 6.0                   # (C, k)

    diag_within  = np.zeros(k)
    between_diag = np.einsum('c,ck->k', gmm.weights, dg**2)       # (k,) already in Sigma_phi diag
    for c in range(C):
        wc = gmm.weights[c]
        Num_cc  = ThSigTh_c[c] + sigma**2 * ThTh                      # (k, k)
        S_out   = np.maximum(np.outer(S_c[c], S_c[c]), 1e-24)
        r_cc    = np.clip(Num_cc / S_out, -1 + 1e-7, 1 - 1e-7)        # (k, k)
        r2 = r_cc ** 2; r3 = r_cc ** 3
        Sigma_phi += wc * (r_cc    * np.outer(C1_c[c], C1_c[c])
                         + 2.0*r2  * np.outer(C2_c[c], C2_c[c])
                         + 6.0*r3  * np.outer(C3_c[c], C3_c[c]))
        # Exact diagonal contribution from component c
        E_phi_sq_c = ((mbar_c[c]**2 + S_c[c]**2) * Phi_c[c]
                      + mbar_c[c] * S_c[c] * phi_c_[c])              # (k,)
        diag_within += wc * np.maximum(E_phi_sq_c - c0_c[c]**2, 0.0)

    np.fill_diagonal(Sigma_phi, between_diag + diag_within)

    Sigma_phi = Sigma_phi + lam * np.eye(k)

    # ---- MMSE = Tr(Sigma_p0) - Tr(Cov Sigma_phi^{-1} Cov^T) ----
    try:
        A = np.linalg.solve(Sigma_phi, Cov.T)
        explained = float(np.trace(Cov @ A))
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(Sigma_phi, Cov.T, rcond=None)[0]
        explained = float(np.trace(Cov @ A))

    return max(0.0, trace_p0 - explained)
