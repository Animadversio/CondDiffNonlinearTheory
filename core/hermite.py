"""
Hermite polynomial utilities for non-linear denoiser theory.

Implements probabilist's Hermite polynomials He_n(x) and the Hermite
expansion coefficients c_n(m, s) of the random-feature activation omega.

Key identity (Mehler product formula): for correlated standard Gaussians
(xi_i, xi_j) with correlation rho,
    E[He_n(xi_i) He_m(xi_j)] = n! * rho^n * delta_{nm}
"""

import math
import numpy as np
from functools import lru_cache


# ---------------------------------------------------------------------------
# Hermite polynomials
# ---------------------------------------------------------------------------

def hermite_poly(n: int, x: np.ndarray) -> np.ndarray:
    """
    Probabilist's Hermite polynomial He_n(x) via 3-term recurrence.
    He_0 = 1, He_1 = x, He_{n+1} = x*He_n - n*He_{n-1}
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    He_prev2 = np.ones_like(x)
    He_prev1 = x.copy()
    for k in range(2, n + 1):
        He_curr = x * He_prev1 - (k - 1) * He_prev2
        He_prev2 = He_prev1
        He_prev1 = He_curr
    return He_prev1


def hermite_poly_all(n_max: int, x: np.ndarray) -> np.ndarray:
    """
    Return all He_0, ..., He_{n_max} evaluated at x.
    Shape: (n_max+1, *x.shape)
    """
    x = np.asarray(x, dtype=float)
    result = np.empty((n_max + 1,) + x.shape)
    result[0] = np.ones_like(x)
    if n_max == 0:
        return result
    result[1] = x
    for k in range(2, n_max + 1):
        result[k] = x * result[k - 1] - (k - 1) * result[k - 2]
    return result


# ---------------------------------------------------------------------------
# Hermite expansion coefficients
# ---------------------------------------------------------------------------

def hermite_coeffs_mc(
    omega,
    m: float,
    s: float,
    n_terms: int = 12,
    n_samples: int = 200_000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Monte-Carlo estimate of c_n(m, s) for n = 0, ..., n_terms-1.

    c_n(m, s) = (1/n!) * E_{xi~N(0,1)}[omega(m + s*xi) * He_n(xi)]

    Parameters
    ----------
    omega : callable, scalar -> scalar (vectorized)
    m     : pre-activation mean (M_j(x_0) or M_j^U(x_0))
    s     : noise scale s_j = sigma * ||theta_j||
    n_terms : number of coefficients to compute
    n_samples : MC sample size
    rng   : numpy random Generator (optional)
    """
    if rng is None:
        rng = np.random.default_rng()
    xi = rng.standard_normal(n_samples)
    f_vals = omega(m + s * xi)
    He_all = hermite_poly_all(n_terms - 1, xi)  # (n_terms, n_samples)
    # c_n = E[f * He_n] / n!
    coeffs = np.array([
        np.mean(f_vals * He_all[n]) / math.factorial(n)
        for n in range(n_terms)
    ])
    return coeffs


def hermite_coeffs_batch(
    omega,
    m_arr: np.ndarray,
    s_arr: np.ndarray,
    n_terms: int = 12,
    n_samples: int = 200_000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Batch version: compute c_n(m_j, s_j) for each j.

    Parameters
    ----------
    m_arr : (k,) array of means
    s_arr : (k,) array of noise scales
    Returns
    -------
    coeffs : (k, n_terms) array
    """
    if rng is None:
        rng = np.random.default_rng()
    k = len(m_arr)
    xi = rng.standard_normal(n_samples)  # shared noise
    He_all = hermite_poly_all(n_terms - 1, xi)  # (n_terms, n_samples)
    coeffs = np.zeros((k, n_terms))
    for j in range(k):
        f_vals = omega(m_arr[j] + s_arr[j] * xi)
        for n in range(n_terms):
            coeffs[j, n] = np.mean(f_vals * He_all[n]) / math.factorial(n)
    return coeffs


# ---------------------------------------------------------------------------
# Mehler kernel: E[He_n(xi_i) He_m(xi_j)] = n! rho^n delta_{nm}
# ---------------------------------------------------------------------------

def mehler_inner_product(n: int, rho: float) -> float:
    """
    E[He_n(xi_i) He_n(xi_j)] = n! * rho^n
    for (xi_i, xi_j) jointly N(0,I) with correlation rho.
    """
    return math.factorial(n) * (rho ** n)


def hermite_series_covariance(
    c_i: np.ndarray,
    c_j: np.ndarray,
    rho: float,
    n_start: int = 1,
) -> float:
    """
    Compute sum_{n=n_start}^{N} n! * rho^n * c_n(m_i, s_i) * c_n(m_j, s_j)

    This is one term in Sigma_phi_{ij} (eq. 2.1 in the paper).

    Parameters
    ----------
    c_i, c_j : (n_terms,) Hermite coefficients for features i and j
    rho      : correlation Corr(xi_i, xi_j) = theta_i^T theta_j / (||theta_i|| ||theta_j||)
    n_start  : start summation from n_start (use 1 to exclude n=0 term)
    """
    n_terms = min(len(c_i), len(c_j))
    total = 0.0
    for n in range(n_start, n_terms):
        total += math.factorial(n) * (rho ** n) * c_i[n] * c_j[n]
    return total


# ---------------------------------------------------------------------------
# Gaussian-smoothed activation  g(m) = (omega * N_{s^2})(m)
# ---------------------------------------------------------------------------

def smoothed_activation_mc(
    omega,
    m: float,
    s: float,
    n_samples: int = 200_000,
    rng: np.random.Generator = None,
) -> float:
    """
    Estimate g(m) = E_{u~N(0,s^2)}[omega(m + u)] = E_{xi~N(0,1)}[omega(m + s*xi)].
    Equivalent to c_0(m, s) * 0! = c_0(m, s).
    """
    if rng is None:
        rng = np.random.default_rng()
    xi = rng.standard_normal(n_samples)
    return float(np.mean(omega(m + s * xi)))
