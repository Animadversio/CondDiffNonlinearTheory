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
    """c3(M, s) = M phi(z) / 6.  Derivation: E[relu(M+s*xi)He_3(xi)] = M phi(z)."""
    z = M / np.maximum(s, 1e-12)
    return M * norm.pdf(z) / 6.0


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

        return float(np.mean((D_star - x0) ** 2))

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
            C3 = M * norm.pdf(z) / 6.0
            Sig_noise += 6.0 * rho**3 * (C3.T @ C3 / N)

        return Sig_noise  # (k, k)
