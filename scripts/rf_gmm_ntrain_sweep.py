"""
RF random-feature MMSE sweep over k for varying N_train on GMM target.

Goal: explore the data-scarce regime (N_train << d=8 to N_train >> d=8)
while k >> N_train (overparameterized feature regime).

Sweeps N_train ∈ {8, 64, 128, 256, 1024, 50000} × k ∈ powers-of-2 up to K_MAX.
Same GMM as rf_gmm_sweep.py (d=8, C=3, weights=[0.5,0.3,0.2]).

Outputs:
  figures/rf_gmm_ntrain_sweep.png
  tables/rf_gmm_ntrain_sweep.npz

Usage:
  python scripts/rf_gmm_ntrain_sweep.py            # k <= 4096
  K_MAX=32768 python scripts/rf_gmm_ntrain_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

from core.gmm import GaussianMixture, _c0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

D         = 8
N_CLASSES = 3
WEIGHTS   = [0.5, 0.3, 0.2]

SEED    = 42
N_NOISE = int(os.environ.get('N_NOISE', '5'))
LAM     = float(os.environ.get('LAM',   '1e-4'))
K_MAX   = int(os.environ.get('K_MAX',  '4096'))

# N_train values to sweep (small → data-scarce)
N_TRAIN_VALUES = [int(x) for x in
    os.environ.get('N_TRAIN_VALUES', '8,64,128,256,1024,50000').split(',')]

# k grid: powers of 2 from 8 up to K_MAX
K_GRID = [2**i for i in range(3, 16) if 2**i <= K_MAX]

# Fixed sigma values for k-sweep panels
SIGMA_VALUES = [float(s) for s in
    os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]

N_MC_EXACT = int(os.environ.get('N_MC_EXACT', '200000'))


# ---------------------------------------------------------------------------
# Same GMM as rf_gmm_sweep.py
# ---------------------------------------------------------------------------

def make_gmm(seed: int = SEED) -> GaussianMixture:
    rng = np.random.default_rng(seed)
    d = D
    means = np.zeros((N_CLASSES, d))
    means[0, 0] =  2.0
    means[1, 0] = -1.0; means[1, 1] =  1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2
    S0 = np.diag([1.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    S1 = np.diag([0.4, 1.0, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4])
    A  = rng.standard_normal((d, d)) * 0.3
    S2 = A @ A.T + 0.5 * np.eye(d)
    return GaussianMixture(
        weights=np.array(WEIGHTS),
        means=means,
        covs=np.stack([S0, S1, S2]),
    )


# ---------------------------------------------------------------------------
# Theory MMSE (Stein + Hermite n<=3) — same as rf_gmm_sweep.py
# ---------------------------------------------------------------------------

def mmse_theory(gmm, Theta, Gamma, x0, U, trace_p0, sigma, lam, conditional=True):
    N, d = x0.shape
    k = Theta.shape[0]
    if conditional:
        Cov_x0_phi = gmm.cov_x0_phi_stein_relu(Theta, Gamma, sigma)
    else:
        Gamma_zero = np.zeros_like(Gamma)
        Cov_x0_phi = gmm.cov_x0_phi_stein_relu(Theta, Gamma_zero, sigma)

    if conditional:
        M = x0 @ Theta.T + U @ Gamma.T
    else:
        M = x0 @ Theta.T
    s = sigma * np.linalg.norm(Theta, axis=1)
    z = M / np.maximum(s[None, :], 1e-12)
    Phi_z = scipy_norm.cdf(z)
    phi_z = scipy_norm.pdf(z)
    G = M * Phi_z + s[None, :] * phi_z
    G_c = G - G.mean(0)
    Sig_data = G_c.T @ G_c / N

    C1 = s[None, :] * Phi_z
    C2 = s[None, :] * phi_z / 2.0
    C3 = M * phi_z / 6.0
    norms_T = np.linalg.norm(Theta, axis=1)
    Theta_n = Theta / norms_T[:, None]
    rho = Theta_n @ Theta_n.T
    rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)
    Sig_noise = (rho * (C1.T @ C1 / N)
                 + 2.0 * rho**2 * (C2.T @ C2 / N)
                 + 6.0 * rho**3 * (C3.T @ C3 / N))

    Sigma_phi = Sig_data + Sig_noise + lam * np.eye(k)
    try:
        A = np.linalg.solve(Sigma_phi, Cov_x0_phi.T)
        explained = float(np.trace(Cov_x0_phi @ A))
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(Sigma_phi, Cov_x0_phi.T, rcond=None)[0]
        explained = float(np.trace(Cov_x0_phi @ A))
    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Empirical MMSE
# ---------------------------------------------------------------------------

class CovAccum:
    def __init__(self, d, k):
        self.sum_phi     = np.zeros(k, dtype=np.float64)
        self.sum_x0_phiT = np.zeros((d, k), dtype=np.float64)
        self.sum_phiphiT = np.zeros((k, k), dtype=np.float64)
        self.n = 0

    def add(self, phi, x0_c):
        phi  = phi.astype(np.float64)
        x0_c = x0_c.astype(np.float64)
        self.sum_phi     += phi.sum(0)
        self.sum_x0_phiT += x0_c.T @ phi
        self.sum_phiphiT += phi.T @ phi
        self.n           += phi.shape[0]

    def covariances(self, lam):
        n = self.n
        mu_phi = self.sum_phi / n
        Cov = (self.sum_x0_phiT / n).astype(np.float32)
        Sig = (self.sum_phiphiT / n
               - np.outer(mu_phi, mu_phi)
               + lam * np.eye(len(mu_phi))).astype(np.float32)
        return Cov, Sig


def mmse_cf(Cov_x0_phi, Sigma_phi, trace_p0):
    try:
        A = np.linalg.solve(Sigma_phi.astype(np.float64),
                            Cov_x0_phi.T.astype(np.float64))
        explained = float(np.trace(Cov_x0_phi.astype(np.float64) @ A))
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(Sigma_phi.astype(np.float64),
                            Cov_x0_phi.T.astype(np.float64), rcond=None)[0]
        explained = float(np.trace(Cov_x0_phi.astype(np.float64) @ A))
    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng_master = np.random.default_rng(SEED)
    print(f"Config: d={D}, C={N_CLASSES}, K_MAX={K_MAX}, N_NOISE={N_NOISE}")
    print(f"N_TRAIN_VALUES: {N_TRAIN_VALUES}")
    print(f"SIGMA_VALUES: {SIGMA_VALUES}")

    gmm = make_gmm(SEED)
    print(f"GMM Tr(Sigma_p0): {np.trace(gmm.Sigma):.4f}")

    # ---- Baselines (independent of N_train) ----
    print("Computing baselines ...")
    baselines = {}
    for sigma in SIGMA_VALUES:
        baselines[sigma] = {
            'linear_wiener':  gmm.mmse_uncond_wiener(sigma),
            'cond_wiener':    gmm.mmse_cond_wiener(sigma),
            'exact_mmse':     gmm.mmse_uncond_exact(sigma, N_mc=N_MC_EXACT,
                                  rng=np.random.default_rng(SEED+1)),
        }
        print(f"  sigma={sigma:.2f}: wiener={baselines[sigma]['linear_wiener']:.3f}, "
              f"exact={baselines[sigma]['exact_mmse']:.3f}, "
              f"cond_wiener={baselines[sigma]['cond_wiener']:.3f}")

    # ---- Pre-draw random projections (shared across N_train for fair comparison) ----
    # Use a different rng per k so Theta/Gamma don't depend on N_train
    Theta_cache = {}
    Gamma_cache = {}
    rng_proj = np.random.default_rng(SEED + 100)
    for k in K_GRID:
        Theta_cache[k] = rng_proj.standard_normal((k, D)) / np.sqrt(D)
        Gamma_cache[k] = rng_proj.standard_normal((k, N_CLASSES)) / np.sqrt(N_CLASSES)

    # ---- Results: dict[N_train][sigma][method] = list over k ----
    # keys: rf_uncond_cf, rf_cond_cf, rf_uncond_theory, rf_cond_theory
    all_results = {}

    for N_train in N_TRAIN_VALUES:
        print(f"\n=== N_train={N_train} ===")
        rng = np.random.default_rng(SEED + N_train)

        # Sample training data
        x0_train, labels_train, U_train = gmm.sample(N_train, rng=rng)
        mu_x0 = x0_train.mean(0)
        X0_c  = x0_train - mu_x0
        # Use N-1 denominator only when N>1
        denom = max(N_train - 1, 1)
        trace_p0 = float(np.trace(X0_c.T @ X0_c / denom))
        print(f"  Empirical Tr(Sigma_p0): {trace_p0:.4f}  (true: {np.trace(gmm.Sigma):.4f})")

        results = {sg: {key: [] for key in [
            'rf_uncond_cf', 'rf_cond_cf',
            'rf_uncond_theory', 'rf_cond_theory',
        ]} for sg in SIGMA_VALUES}

        for k in tqdm(K_GRID, desc=f'k (N={N_train})'):
            Theta = Theta_cache[k]
            Gamma = Gamma_cache[k]

            for sigma in SIGMA_VALUES:
                # Theory uses empirical x0/U for Sigma_phi data part,
                # but Stein Cov uses the GMM analytically
                rf_u_th = mmse_theory(gmm, Theta, Gamma, x0_train, U_train,
                                      trace_p0, sigma, LAM, conditional=False)
                rf_c_th = mmse_theory(gmm, Theta, Gamma, x0_train, U_train,
                                      trace_p0, sigma, LAM, conditional=True)
                results[sigma]['rf_uncond_theory'].append(rf_u_th)
                results[sigma]['rf_cond_theory'].append(rf_c_th)

                # Empirical CF (average over N_NOISE noise draws)
                accum_u = CovAccum(D, k)
                accum_c = CovAccum(D, k)
                for _ in range(N_NOISE):
                    Z = rng.standard_normal(x0_train.shape) * sigma
                    Y = x0_train + Z
                    phi_u = np.maximum(Y @ Theta.T, 0)
                    phi_c = np.maximum(Y @ Theta.T + U_train @ Gamma.T, 0)
                    accum_u.add(phi_u, X0_c)
                    accum_c.add(phi_c, X0_c)
                Cov_u, Sig_u = accum_u.covariances(LAM)
                Cov_c, Sig_c = accum_c.covariances(LAM)
                results[sigma]['rf_uncond_cf'].append(mmse_cf(Cov_u, Sig_u, trace_p0))
                results[sigma]['rf_cond_cf'].append(mmse_cf(Cov_c, Sig_c, trace_p0))

        all_results[N_train] = {'results': results, 'trace_p0': trace_p0}

    # ---- Save ----
    os.makedirs('tables', exist_ok=True)
    save_dict = {'k_grid': np.array(K_GRID),
                 'sigma_values': np.array(SIGMA_VALUES),
                 'n_train_values': np.array(N_TRAIN_VALUES)}
    for N_train, d in all_results.items():
        save_dict[f'trace_p0_N{N_train}'] = d['trace_p0']
        for sg in SIGMA_VALUES:
            for key, vals in d['results'][sg].items():
                save_dict[f'{key}_N{N_train}_s{sg}'] = np.array(vals)
    for sg in SIGMA_VALUES:
        for bkey, bval in baselines[sg].items():
            save_dict[f'baseline_{bkey}_s{sg}'] = bval
    np.savez('tables/rf_gmm_ntrain_sweep.npz', **save_dict)
    print("Saved tables/rf_gmm_ntrain_sweep.npz")

    plot(all_results, baselines, K_GRID, SIGMA_VALUES)


def plot(all_results, baselines, K_GRID, SIGMA_VALUES):
    N_TRAIN_VALUES = sorted(all_results.keys())
    k_over_d = np.array(K_GRID) / D

    # Color by N_train (light = small, dark = large)
    cmap = plt.cm.viridis
    n_colors = len(N_TRAIN_VALUES)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    n_sigma = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, n_sigma, figsize=(5 * n_sigma, 10))
    fig.suptitle(
        f'RF Denoiser on GMM (d={D}, C={N_CLASSES}) — N_train titration\n'
        f'Baselines are infinite-N limits  |  k up to {K_GRID[-1]}',
        fontsize=11,
    )

    from matplotlib.lines import Line2D

    for col, sigma in enumerate(SIGMA_VALUES):
        bl = baselines[sigma]

        # ---- Row 0: Unconditional ----
        ax = axes[0, col]
        ax.axhline(bl['linear_wiener'], color='k', ls=':', lw=1.5, label='Linear Wiener')
        ax.axhline(bl['exact_mmse'],    color='k', ls='--', lw=1.5, label='Exact MMSE')

        for N_train, color in zip(N_TRAIN_VALUES, colors):
            r = all_results[N_train]['results'][sigma]
            label = f'N={N_train}'
            ax.plot(k_over_d, r['rf_uncond_cf'],     color=color, lw=2,  ls='-',
                    label=label + ' CF')
            ax.plot(k_over_d, r['rf_uncond_theory'], color=color, lw=1.5, ls='-.',
                    label=label + ' Theory')

        ax.set_xscale('log')
        ax.set_xlabel('k / d')
        ax.set_ylabel('MSE')
        ax.set_title(f'Uncond  σ={sigma}')
        ax.grid(True, alpha=0.3)

        if col == n_sigma - 1:
            legend_style = [
                Line2D([0],[0], color='k', ls=':',  lw=1.5, label='Linear Wiener'),
                Line2D([0],[0], color='k', ls='--', lw=1.5, label='Exact MMSE'),
                Line2D([0],[0], color='k', ls='-',  lw=2,   label='RF CF'),
                Line2D([0],[0], color='k', ls='-.', lw=1.5, label='RF Theory'),
            ]
            legend_color = [Line2D([0],[0], color=c, lw=2, label=f'N={n}')
                            for n, c in zip(N_TRAIN_VALUES, colors)]
            ax.legend(handles=legend_style + legend_color, fontsize=7, ncol=2,
                      loc='upper right')

        # ---- Row 1: Conditional ----
        ax = axes[1, col]
        ax.axhline(bl['cond_wiener'], color='k', ls=':', lw=1.5, label='Cond Wiener')
        ax.axhline(bl['exact_mmse'], color='k', ls='--', lw=1.5, label='Exact MMSE')

        for N_train, color in zip(N_TRAIN_VALUES, colors):
            r = all_results[N_train]['results'][sigma]
            ax.plot(k_over_d, r['rf_cond_cf'],     color=color, lw=2,  ls='-')
            ax.plot(k_over_d, r['rf_cond_theory'], color=color, lw=1.5, ls='-.')

        ax.set_xscale('log')
        ax.set_xlabel('k / d')
        ax.set_ylabel('MSE')
        ax.set_title(f'Cond (U=class)  σ={sigma}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = 'figures/rf_gmm_ntrain_sweep.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
