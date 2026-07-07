"""
Verify the jointly Gaussian closed-form loss formulas (Section 3).

Compares:
  (A) Empirical loss (train linear head on random-feature denoiser)
  (B) Theoretical loss via Hermite expansion + Stein's lemma

For both unconditional L_sigma and conditional L_{sigma,U}.

Usage:
    python scripts/verify_jointly_gaussian.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from core import (
    JointGaussian,
    RandomFeatureMap,
    empirical_loss,
    empirical_covariances,
    theoretical_loss_from_cov,
    gaussian_theoretical_loss,
)


def run_verification(
    d=8, d_u=4, k=64, N=5000,
    sigma_grid=None,
    omega=np.tanh,
    seed=42,
    n_terms=10,
    n_mc=100_000,
):
    rng = np.random.default_rng(seed)

    if sigma_grid is None:
        sigma_grid = np.logspace(-1, 1, 12)

    # ---- Define joint Gaussian distribution ----
    # Random positive-definite Sigma_p0
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    Sigma_p0 = A @ A.T + 0.5 * np.eye(d)

    A_U = rng.standard_normal((d_u, d_u)) / np.sqrt(d_u)
    Sigma_U = A_U @ A_U.T + 0.5 * np.eye(d_u)

    # Cross-covariance: moderate correlation
    C_xU = 0.3 * rng.standard_normal((d, d_u))

    mu_x = rng.standard_normal(d) * 0.5
    mu_U = rng.standard_normal(d_u) * 0.5

    dist = JointGaussian(mu_x, mu_U, Sigma_p0, Sigma_U, C_xU)

    # ---- Random feature map ----
    Theta   = rng.standard_normal((k, d))  / np.sqrt(d)
    Gamma   = rng.standard_normal((k, d_u)) / np.sqrt(d_u)
    epsilon = rng.standard_normal(k) * 0.1

    phi_map = RandomFeatureMap(Theta, omega, Gamma, epsilon)

    # ---- Sample data ----
    x0, U = dist.sample(N, rng=rng)

    # ---- Sweep over sigma ----
    results_empirical_uncond = []
    results_empirical_cond   = []
    results_theory_uncond    = []
    results_theory_cond      = []

    print(f"Running sigma sweep ({len(sigma_grid)} values) ...")
    for sigma in tqdm(sigma_grid):
        # Empirical
        L_emp_uncond = empirical_loss(phi_map, x0, sigma, U_samples=None,  rng=rng)
        L_emp_cond   = empirical_loss(phi_map, x0, sigma, U_samples=U,     rng=rng)

        # Theory (Stein / Gaussian)
        th = gaussian_theoretical_loss(
            omega, Theta, Gamma, epsilon, sigma, dist,
            n_terms=n_terms, n_mc=n_mc, rng=rng,
        )

        results_empirical_uncond.append(L_emp_uncond)
        results_empirical_cond.append(L_emp_cond)
        results_theory_uncond.append(th['L_sigma'])
        results_theory_cond.append(th['L_sigma_U'])

    res = dict(
        sigma_grid=sigma_grid,
        L_emp_uncond=np.array(results_empirical_uncond),
        L_emp_cond=np.array(results_empirical_cond),
        L_th_uncond=np.array(results_theory_uncond),
        L_th_cond=np.array(results_theory_cond),
    )
    return res


def plot_results(res, save_path='figures/verify_jointly_gaussian.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sigma = res['sigma_grid']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(sigma, res['L_emp_uncond'],  'o-', label='Empirical L_sigma', color='C0')
    ax.plot(sigma, res['L_th_uncond'],   's--', label='Theory L_sigma',   color='C0', alpha=0.7)
    ax.plot(sigma, res['L_emp_cond'],    'o-', label='Empirical L_{sigma,U}', color='C1')
    ax.plot(sigma, res['L_th_cond'],     's--', label='Theory L_{sigma,U}',   color='C1', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs sigma: Empirical vs Theory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    gain_emp = res['L_emp_uncond'] - res['L_emp_cond']
    gain_th  = res['L_th_uncond']  - res['L_th_cond']
    ax.plot(sigma, gain_emp, 'o-',  label='Empirical gain', color='C2')
    ax.plot(sigma, gain_th,  's--', label='Theory gain',    color='C2', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('L_sigma - L_{sigma,U}')
    ax.set_title('Conditioning gain vs sigma')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    return fig


if __name__ == '__main__':
    res = run_verification()
    plot_results(res)

    # Print summary
    print("\n=== Summary ===")
    for i, sigma in enumerate(res['sigma_grid']):
        print(f"sigma={sigma:.3f}: "
              f"L_emp={res['L_emp_uncond'][i]:.4f} vs L_th={res['L_th_uncond'][i]:.4f} | "
              f"LU_emp={res['L_emp_cond'][i]:.4f} vs LU_th={res['L_th_cond'][i]:.4f}")
