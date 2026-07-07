"""
Sweep over noise level sigma and compute MI approximation.

Plots:
  - L_sigma, L_{sigma,U} vs sigma
  - conditioning gain Delta_L vs sigma
  - MI integrand (1/sigma^3) * Delta_L vs sigma
  - Cumulative MI approximation

Usage:
    python scripts/sweep_sigma.py [--d D] [--k K] [--N N] [--seed SEED]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from core import (
    JointGaussian,
    RandomFeatureMap,
    gaussian_theoretical_loss,
    mi_sigma_sweep,
    summarize_results,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--d',    type=int, default=8,   help='data dim')
    p.add_argument('--d_u',  type=int, default=4,   help='conditioning dim')
    p.add_argument('--k',    type=int, default=128,  help='number of random features')
    p.add_argument('--sigma_min', type=float, default=0.05)
    p.add_argument('--sigma_max', type=float, default=20.0)
    p.add_argument('--n_sigma',   type=int,   default=30)
    p.add_argument('--n_terms',   type=int,   default=10)
    p.add_argument('--n_mc',      type=int,   default=100_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--omega', choices=['tanh', 'sigmoid', 'relu'], default='tanh')
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    d, d_u, k = args.d, args.d_u, args.k
    omega_map = {
        'tanh':    np.tanh,
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'relu':    lambda x: np.maximum(x, 0),
    }
    omega = omega_map[args.omega]

    # Distribution
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    Sigma_p0 = A @ A.T + 0.5 * np.eye(d)
    A_U = rng.standard_normal((d_u, d_u)) / np.sqrt(d_u)
    Sigma_U = A_U @ A_U.T + 0.5 * np.eye(d_u)
    C_xU = 0.4 * rng.standard_normal((d, d_u))
    dist = JointGaussian(
        np.zeros(d), np.zeros(d_u),
        Sigma_p0, Sigma_U, C_xU,
    )

    # Random feature map
    Theta   = rng.standard_normal((k, d))   / np.sqrt(d)
    Gamma   = rng.standard_normal((k, d_u)) / np.sqrt(d_u)
    epsilon = rng.standard_normal(k) * 0.1

    sigma_grid = np.logspace(
        np.log10(args.sigma_min),
        np.log10(args.sigma_max),
        args.n_sigma,
    )

    def get_covs(sigma):
        return gaussian_theoretical_loss(
            omega, Theta, Gamma, epsilon, sigma, dist,
            n_terms=args.n_terms, n_mc=args.n_mc, rng=rng,
        )

    print(f"Sweeping sigma in [{args.sigma_min}, {args.sigma_max}] with {args.n_sigma} points ...")
    from tqdm import tqdm
    results_list = [get_covs(s) for s in tqdm(sigma_grid)]

    results = dict(
        sigma_grid=sigma_grid,
        L_sigma    = np.array([r['L_sigma']   for r in results_list]),
        L_sigma_U  = np.array([r['L_sigma_U'] for r in results_list]),
        gain       = np.array([r['gain']       for r in results_list]),
    )
    results['integrand'] = results['gain'] / sigma_grid ** 3
    results['mi_approx'] = 0.5 * np.trapz(results['integrand'], sigma_grid)

    print("\n" + summarize_results(results))

    # ---- Plot ----
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'd={d}, d_u={d_u}, k={k}, omega={args.omega}', fontsize=12)

    ax = axes[0]
    ax.plot(sigma_grid, results['L_sigma'],   label='L_sigma (uncond)')
    ax.plot(sigma_grid, results['L_sigma_U'], label='L_sigma_U (cond)')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss vs sigma')

    ax = axes[1]
    ax.plot(sigma_grid, results['gain'], color='C2')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('L_sigma - L_{sigma,U}')
    ax.grid(True, alpha=0.3)
    ax.set_title('Conditioning gain')

    ax = axes[2]
    ax.plot(sigma_grid, results['integrand'], color='C3')
    ax.fill_between(sigma_grid, 0, results['integrand'], alpha=0.3, color='C3')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('(1/sigma^3) * gain')
    ax.set_title(f'MI integrand  (I(X;U) ≈ {results["mi_approx"]:.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'figures/sweep_sigma_{args.omega}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
