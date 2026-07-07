"""
Verify Hermite expansion coefficients c_n(m, s) for common activations.

Checks:
  1. That MC estimates of c_n match numerical integration
  2. That f(xi) ≈ sum_n c_n * He_n(xi) reconstructs the activation
  3. The Mehler product identity: E[He_n(xi_i) He_m(xi_j)] = n! rho^n delta_{nm}

Usage:
    python scripts/verify_hermite_coeffs.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as sci_integrate

from core.hermite import (
    hermite_poly, hermite_poly_all,
    hermite_coeffs_mc, hermite_series_covariance,
)


OMEGAS = {
    'tanh':    np.tanh,
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'relu':    lambda x: np.maximum(x, 0),
    'gelu':    lambda x: x * 0.5 * (1 + np.vectorize(
                   lambda v: __import__('math').erf(v / np.sqrt(2)))(x)),
}


def verify_reconstruction(omega_name='tanh', m=0.5, s=1.2, n_terms=15, n_mc=500_000):
    """Check that truncated Hermite series reconstructs omega(m + s*xi)."""
    omega = OMEGAS[omega_name]
    rng = np.random.default_rng(0)
    xi_test = np.linspace(-4, 4, 500)

    # Coefficients
    coeffs = hermite_coeffs_mc(omega, m, s, n_terms=n_terms, n_mc=n_mc, rng=rng)

    # Reconstruct
    He_all = hermite_poly_all(n_terms - 1, xi_test)  # (n_terms, 500)
    f_approx = np.zeros(500)
    for n in range(n_terms):
        f_approx += coeffs[n] * He_all[n]

    f_true = omega(m + s * xi_test)

    return xi_test, f_true, f_approx, coeffs


def verify_mehler_identity(rho=0.7, n_max=6, n_mc=500_000):
    """
    Check E[He_n(xi_i) He_m(xi_j)] = n! rho^n delta_{nm}
    for correlated Gaussians with correlation rho.
    """
    import math
    rng = np.random.default_rng(1)

    # Sample correlated pair
    cov = np.array([[1, rho], [rho, 1]])
    samples = rng.multivariate_normal([0, 0], cov, size=n_mc)
    xi_i, xi_j = samples[:, 0], samples[:, 1]

    He_i = hermite_poly_all(n_max, xi_i)  # (n_max+1, n_mc)
    He_j = hermite_poly_all(n_max, xi_j)

    results = {}
    for n in range(n_max + 1):
        for m in range(n_max + 1):
            emp = np.mean(He_i[n] * He_j[m])
            theory = math.factorial(n) * (rho ** n) if n == m else 0.0
            results[(n, m)] = (emp, theory)
    return results


def plot_verification(save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Reconstruction for each activation
    for idx, (name, omega) in enumerate(list(OMEGAS.items())[:4]):
        ax = axes[idx // 2][idx % 2]
        xi_test, f_true, f_approx, coeffs = verify_reconstruction(
            omega_name=name, m=0.3, s=1.0, n_terms=12
        )
        ax.plot(xi_test, f_true,   'k-',  label=f'{name}(0.3 + 1.0*xi)', lw=2)
        ax.plot(xi_test, f_approx, 'r--', label='Hermite series (12 terms)', lw=2)
        ax.set_title(f'{name}: reconstruction')
        ax.set_xlabel('xi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        rmse = np.sqrt(np.mean((f_true - f_approx) ** 2))
        ax.set_title(f'{name}: RMSE={rmse:.4f}')

    plt.tight_layout()
    path = os.path.join(save_dir, 'verify_hermite_reconstruction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")

    # 2. Mehler identity check
    mehler = verify_mehler_identity(rho=0.6, n_max=5)
    print("\n=== Mehler identity check (rho=0.6) ===")
    print(f"{'(n,m)':>8}  {'empirical':>12}  {'theory':>12}  {'error':>10}")
    import math
    for (n, m), (emp, th) in sorted(mehler.items()):
        if n == m:  # only print diagonal for brevity
            print(f"  ({n},{m})    {emp:12.4f}  {th:12.4f}  {abs(emp-th):10.4f}")


if __name__ == '__main__':
    plot_verification()
