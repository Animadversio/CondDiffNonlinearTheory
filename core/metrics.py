"""
Mutual information estimation and conditioning gain metrics.

The main quantity of interest (Section 2 of the theory notes):

    I(X; U) ≈ (1/2) ∫_0^inf (d sigma / sigma^3) * [L_sigma - L_{sigma,U}]

where the gain integrand at each sigma is:
    Delta_L(sigma) = L_sigma - L_{sigma,U}
                   = Tr(A_U Sigma_phi_U^{-1} A_U^T) - Tr(A Sigma_phi^{-1} A^T)

This module also provides utilities for sweeping over sigma values.
"""

import numpy as np
from typing import Callable, Optional
from scipy import integrate


# ---------------------------------------------------------------------------
# MI approximation via sigma sweep
# ---------------------------------------------------------------------------

def mi_integrand(
    sigma: float,
    Sigma_p0: np.ndarray,
    Cov_x0_phi: np.ndarray,
    Sigma_phi: np.ndarray,
    Cov_x0_phiU: np.ndarray,
    Sigma_phiU: np.ndarray,
) -> float:
    """
    Compute (1/sigma^3) * [L_sigma - L_{sigma,U}] at a single sigma.

    Parameters
    ----------
    Cov_x0_phi  : (d, k) unconditional cross-covariance
    Sigma_phi   : (k, k) unconditional feature covariance
    Cov_x0_phiU : (d, k) conditional cross-covariance
    Sigma_phiU  : (k, k) conditional feature covariance

    Returns
    -------
    integrand value (scalar)
    """
    def explained_var(Cov, Sigma):
        return np.trace(Cov @ np.linalg.pinv(Sigma) @ Cov.T)

    delta_L = explained_var(Cov_x0_phiU, Sigma_phiU) - explained_var(Cov_x0_phi, Sigma_phi)
    return float(delta_L / sigma ** 3)


def mi_sigma_sweep(
    sigma_grid: np.ndarray,
    get_covs: Callable[[float], dict],
) -> dict:
    """
    Sweep over sigma values and compute the MI integrand at each point.

    Parameters
    ----------
    sigma_grid : 1D array of sigma values
    get_covs   : callable(sigma) -> dict with keys
                 {Sigma_p0, Cov_x0_phi, Sigma_phi, Cov_x0_phiU, Sigma_phiU,
                  L_sigma, L_sigma_U}

    Returns
    -------
    dict with arrays:
        sigma_grid, L_sigma, L_sigma_U, gain, integrand, mi_approx
    """
    L_list, LU_list, gain_list, integrand_list = [], [], [], []

    for sigma in sigma_grid:
        covs = get_covs(sigma)
        L_list.append(covs['L_sigma'])
        LU_list.append(covs['L_sigma_U'])
        gain = covs['L_sigma'] - covs['L_sigma_U']
        gain_list.append(gain)
        integrand_list.append(gain / sigma ** 3)

    L_arr       = np.array(L_list)
    LU_arr      = np.array(LU_list)
    gain_arr    = np.array(gain_list)
    integ_arr   = np.array(integrand_list)

    # Trapezoidal MI approximation: (1/2) * integral
    mi_approx = 0.5 * np.trapz(integ_arr, sigma_grid)

    return dict(
        sigma_grid=sigma_grid,
        L_sigma=L_arr,
        L_sigma_U=LU_arr,
        gain=gain_arr,
        integrand=integ_arr,
        mi_approx=mi_approx,
    )


# ---------------------------------------------------------------------------
# Explained variance (denoiser R^2)
# ---------------------------------------------------------------------------

def explained_variance(
    Cov_x0_phi: np.ndarray,
    Sigma_phi: np.ndarray,
    Sigma_p0: np.ndarray,
) -> float:
    """
    Fraction of variance explained by the best linear regression of x0 onto phi:
        R^2 = Tr(Cov(x0,phi) Sigma_phi^{-1} Cov(phi,x0)) / Tr(Sigma_p0)
    """
    num = np.trace(Cov_x0_phi @ np.linalg.pinv(Sigma_phi) @ Cov_x0_phi.T)
    denom = np.trace(Sigma_p0)
    return float(num / denom)


def conditioning_gain_r2(
    Cov_x0_phi: np.ndarray,
    Sigma_phi: np.ndarray,
    Cov_x0_phiU: np.ndarray,
    Sigma_phiU: np.ndarray,
    Sigma_p0: np.ndarray,
) -> float:
    """
    Gain in R^2 from conditioning:
        Delta R^2 = R^2(phi^U) - R^2(phi)
    """
    r2_cond   = explained_variance(Cov_x0_phiU, Sigma_phiU, Sigma_p0)
    r2_uncond = explained_variance(Cov_x0_phi,  Sigma_phi,  Sigma_p0)
    return r2_cond - r2_uncond


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_results(results: dict) -> str:
    """Print a readable summary of sigma-sweep results."""
    lines = [
        f"MI approximation: {results['mi_approx']:.4f}",
        f"Max gain (L_sigma - L_sigma_U): {results['gain'].max():.4f}  "
        f"at sigma={results['sigma_grid'][results['gain'].argmax()]:.3f}",
        f"L_sigma range:   [{results['L_sigma'].min():.4f}, {results['L_sigma'].max():.4f}]",
        f"L_sigma_U range: [{results['L_sigma_U'].min():.4f}, {results['L_sigma_U'].max():.4f}]",
    ]
    return "\n".join(lines)
