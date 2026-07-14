"""
RF random-feature MMSE sweep over k for GMM target distribution.

Target: 3-component GMM in d=8 with weights [0.5, 0.3, 0.2].
Conditioning: U = one-hot component label (label_noise configurable).

Compares at each (k, sigma):
  - exact_mmse_uncond  : posterior Wiener MMSE (computed once, sigma-only)
  - exact_mmse_cond    : per-class analytic Wiener MMSE (= cond Wiener for Gaussian components)
  - linear_wiener      : unconditional linear Wiener from mixture covariance
  - cond_wiener        : conditional linear Wiener (per-class average)
  - rf_uncond_cf       : RF empirical closed-form (in-sample)
  - rf_cond_cf         : RF cond empirical closed-form
  - rf_uncond_theory   : RF theory via Stein + Hermite (n<=3)
  - rf_cond_theory     : RF cond theory via Stein + Hermite (n<=3)

Outputs:
  figures/rf_gmm_sweep.png
  tables/rf_gmm_sweep.npz

Usage:
  python scripts/rf_gmm_sweep.py          # k <= 1024 draft
  K_MAX=32768 python scripts/rf_gmm_sweep.py   # full sweep
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

SEED      = 42
N_TRAIN   = int(os.environ.get('N_TRAIN',  '50000'))
N_NOISE   = int(os.environ.get('N_NOISE',  '5'))
LAM       = float(os.environ.get('LAM',    '1e-4'))
K_MAX     = int(os.environ.get('K_MAX',    '1024'))
LABEL_NOISE = float(os.environ.get('LABEL_NOISE', '0.0'))

N_MC_EXACT = int(os.environ.get('N_MC_EXACT', '500000'))   # MC for exact MMSE

# k grid: powers of 2 from 8 up to K_MAX
K_GRID = [2**i for i in range(3, 16) if 2**i <= K_MAX]

# Fixed sigma values to compare at
SIGMA_VALUES = [float(s) for s in os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]

# Fixed k for sigma-sweep panel
K_FOR_SIGMA_SWEEP = int(os.environ.get('K_SIGMA', str(min(512, K_MAX))))

SIGMA_GRID = np.logspace(np.log10(0.05), np.log10(20.0), 20)


# ---------------------------------------------------------------------------
# Define GMM
# ---------------------------------------------------------------------------

def make_gmm(seed: int = SEED) -> GaussianMixture:
    """3-component GMM in d=8 with structured means and covariances."""
    rng = np.random.default_rng(seed)
    d = D

    # Means: each component has a distinct direction
    means = np.zeros((N_CLASSES, d))
    means[0, 0] =  2.0
    means[1, 0] = -1.0; means[1, 1] =  1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2

    # Covariances: different spread per component
    # Component 0: elongated along dim 0
    S0 = np.diag([1.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    # Component 1: elongated along dims 1-2
    S1 = np.diag([0.4, 1.0, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4])
    # Component 2: random PD with seed
    A = rng.standard_normal((d, d)) * 0.3
    S2 = A @ A.T + 0.5 * np.eye(d)

    return GaussianMixture(
        weights=np.array(WEIGHTS),
        means=means,
        covs=np.stack([S0, S1, S2]),
    )


# ---------------------------------------------------------------------------
# Theory MMSE using Stein for Cov + Hermite for Sigma_phi noise
# ---------------------------------------------------------------------------

def mmse_theory(gmm, Theta, Gamma, x0, U, trace_p0, sigma, lam, conditional=True):
    """
    Compute RF theory MMSE via:
      Cov(x0, phi^U): Stein closed-form (conditional) or no-Gamma (unconditional)
      Sigma_phi:      data part from MC samples + Hermite noise terms

    Returns scalar MMSE.
    """
    N, d = x0.shape
    k = Theta.shape[0]

    # ---- Cov(x0, phi^U) via Stein ----
    if conditional:
        Cov_x0_phi = gmm.cov_x0_phi_stein_relu(Theta, Gamma, sigma)  # (d, k)
    else:
        # Unconditional: use Gamma=0 (no class conditioning in features)
        Gamma_zero = np.zeros_like(Gamma)
        Cov_x0_phi = gmm.cov_x0_phi_stein_relu(Theta, Gamma_zero, sigma)  # (d, k)

    # ---- Sigma_phi: data part from MC samples ----
    if conditional:
        M = x0 @ Theta.T + U @ Gamma.T   # (N, k)
    else:
        M = x0 @ Theta.T                  # (N, k)

    s = sigma * np.linalg.norm(Theta, axis=1)  # (k,)
    z = M / np.maximum(s[None, :], 1e-12)
    Phi_z = scipy_norm.cdf(z)   # (N, k)
    phi_z = scipy_norm.pdf(z)   # (N, k)

    G = M * Phi_z + s[None, :] * phi_z  # (N, k)  — expected features g_j(x0, U)
    G_c = G - G.mean(0)
    Sig_data = G_c.T @ G_c / N          # (k, k)

    # Hermite noise terms (n=1,2,3)
    C1 = s[None, :] * Phi_z             # (N, k)
    C2 = s[None, :] * phi_z / 2.0       # (N, k)
    C3 = M * phi_z / 6.0                # (N, k)

    norms_T = np.linalg.norm(Theta, axis=1)
    Theta_n = Theta / norms_T[:, None]
    rho = Theta_n @ Theta_n.T
    rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)

    Sig_noise = (rho * (C1.T @ C1 / N)
                 + 2.0 * rho**2 * (C2.T @ C2 / N)
                 + 6.0 * rho**3 * (C3.T @ C3 / N))

    Sigma_phi = Sig_data + Sig_noise + lam * np.eye(k)

    # ---- MMSE = Tr(Sigma_p0) - Tr(Cov Sigma_phi^{-1} Cov^T) ----
    try:
        A = np.linalg.solve(Sigma_phi, Cov_x0_phi.T)   # (k, d)
        explained = float(np.trace(Cov_x0_phi @ A))
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(Sigma_phi, Cov_x0_phi.T, rcond=None)[0]
        explained = float(np.trace(Cov_x0_phi @ A))

    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Empirical MMSE (streaming CovAccum for memory efficiency)
# ---------------------------------------------------------------------------

class CovAccum:
    """Streaming accumulator for Cov(x0, phi) and Sigma_phi."""
    def __init__(self, d, k):
        self.sum_phi     = np.zeros(k, dtype=np.float64)
        self.sum_x0_phiT = np.zeros((d, k), dtype=np.float64)
        self.sum_phiphiT = np.zeros((k, k), dtype=np.float64)
        self.n = 0

    def add(self, phi, x0_c):
        """phi: (N, k), x0_c: (N, d) centered."""
        phi = phi.astype(np.float64)
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
        return Cov, Sig, mu_phi.astype(np.float32)


def mmse_cf(Cov_x0_phi, Sigma_phi, trace_p0):
    """L = Tr(Sigma_p0) - Tr(Cov Sigma_phi^{-1} Cov^T)."""
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
    rng = np.random.default_rng(SEED)
    print(f"Config: d={D}, C={N_CLASSES}, K_MAX={K_MAX}, N_TRAIN={N_TRAIN}, "
          f"N_NOISE={N_NOISE}, label_noise={LABEL_NOISE}")

    # ---- GMM ----
    gmm = make_gmm(SEED)
    print(f"GMM global mean: {gmm.mu.round(3)}")
    print(f"GMM Tr(Sigma_p0): {np.trace(gmm.Sigma):.4f}")

    # ---- Training data (fixed across k values) ----
    print(f"Sampling {N_TRAIN} training points ...")
    x0_train, labels_train, U_train = gmm.sample(N_TRAIN, label_noise=LABEL_NOISE, rng=rng)
    mu_x0 = x0_train.mean(0)
    X0_c  = x0_train - mu_x0
    trace_p0 = float(np.trace(X0_c.T @ X0_c / (N_TRAIN - 1)))
    print(f"Empirical Tr(Sigma_p0): {trace_p0:.4f}")

    # ---- Baseline curves (independent of k) ----
    print("Computing baselines (exact MMSE, Wiener) ...")
    baselines_uncond = {sg: {} for sg in SIGMA_VALUES}
    baselines_cond   = {sg: {} for sg in SIGMA_VALUES}
    for sigma in SIGMA_VALUES:
        baselines_uncond[sigma]['linear_wiener']   = gmm.mmse_uncond_wiener(sigma)
        baselines_uncond[sigma]['exact_mmse']      = gmm.mmse_uncond_exact(sigma, N_mc=N_MC_EXACT, rng=np.random.default_rng(SEED+1))
        baselines_cond[sigma]['cond_wiener']       = gmm.mmse_cond_wiener(sigma)
        baselines_cond[sigma]['exact_mmse_cond']   = gmm.mmse_cond_exact(sigma)
        print(f"  sigma={sigma:.2f}: wiener={baselines_uncond[sigma]['linear_wiener']:.3f}, "
              f"exact={baselines_uncond[sigma]['exact_mmse']:.3f}, "
              f"cond_wiener={baselines_cond[sigma]['cond_wiener']:.3f}")

    # ---- Results containers: dict[sigma][method] = list over k ----
    results = {
        sg: {key: [] for key in [
            'rf_uncond_cf', 'rf_cond_cf',
            'rf_uncond_theory', 'rf_cond_theory',
        ]}
        for sg in SIGMA_VALUES
    }

    # ---- Sweep over k ----
    print(f"\nSweeping k = {K_GRID} ...")
    for k in tqdm(K_GRID, desc='k'):
        Theta = rng.standard_normal((k, D)) / np.sqrt(D)    # (k, d)
        Gamma = rng.standard_normal((k, N_CLASSES)) / np.sqrt(N_CLASSES)

        for sigma in SIGMA_VALUES:
            # Theory (Stein + Hermite)
            rf_u_th = mmse_theory(gmm, Theta, Gamma, x0_train, U_train,
                                   trace_p0, sigma, LAM, conditional=False)
            rf_c_th = mmse_theory(gmm, Theta, Gamma, x0_train, U_train,
                                   trace_p0, sigma, LAM, conditional=True)
            results[sigma]['rf_uncond_theory'].append(rf_u_th)
            results[sigma]['rf_cond_theory'].append(rf_c_th)

            # Empirical CF (streaming over N_NOISE draws)
            accum_u = CovAccum(D, k)
            accum_c = CovAccum(D, k)
            for _ in range(N_NOISE):
                Z = rng.standard_normal(x0_train.shape) * sigma
                Y = x0_train + Z
                phi_u = np.maximum(Y @ Theta.T, 0)                         # (N, k) relu
                phi_c = np.maximum(Y @ Theta.T + U_train @ Gamma.T, 0)    # (N, k)
                accum_u.add(phi_u, X0_c)
                accum_c.add(phi_c, X0_c)

            Cov_u, Sig_u, _ = accum_u.covariances(LAM)
            Cov_c, Sig_c, _ = accum_c.covariances(LAM)

            results[sigma]['rf_uncond_cf'].append(mmse_cf(Cov_u, Sig_u, trace_p0))
            results[sigma]['rf_cond_cf'].append(mmse_cf(Cov_c, Sig_c, trace_p0))

    # ---- Also compute sigma sweep for fixed k (for supplemental panel) ----
    k_fix = K_FOR_SIGMA_SWEEP
    print(f"\nSigma sweep at k={k_fix} ...")
    Theta_fix = rng.standard_normal((k_fix, D)) / np.sqrt(D)
    Gamma_fix = rng.standard_normal((k_fix, N_CLASSES)) / np.sqrt(N_CLASSES)
    sigma_results = {key: [] for key in [
        'linear_wiener', 'cond_wiener', 'exact_mmse', 'exact_mmse_cond',
        'rf_uncond_cf', 'rf_cond_cf', 'rf_uncond_theory', 'rf_cond_theory',
    ]}
    for sigma in tqdm(SIGMA_GRID, desc='sigma'):
        sigma_results['linear_wiener'].append(gmm.mmse_uncond_wiener(sigma))
        sigma_results['cond_wiener'].append(gmm.mmse_cond_wiener(sigma))
        sigma_results['exact_mmse'].append(gmm.mmse_uncond_exact(sigma, N_mc=100_000, rng=np.random.default_rng(SEED+10)))
        sigma_results['exact_mmse_cond'].append(gmm.mmse_cond_exact(sigma))

        sigma_results['rf_uncond_theory'].append(
            mmse_theory(gmm, Theta_fix, Gamma_fix, x0_train, U_train,
                        trace_p0, sigma, LAM, conditional=False))
        sigma_results['rf_cond_theory'].append(
            mmse_theory(gmm, Theta_fix, Gamma_fix, x0_train, U_train,
                        trace_p0, sigma, LAM, conditional=True))

        accum_u = CovAccum(D, k_fix)
        accum_c = CovAccum(D, k_fix)
        for _ in range(N_NOISE):
            Z = rng.standard_normal(x0_train.shape) * sigma
            Y = x0_train + Z
            phi_u = np.maximum(Y @ Theta_fix.T, 0)
            phi_c = np.maximum(Y @ Theta_fix.T + U_train @ Gamma_fix.T, 0)
            accum_u.add(phi_u, X0_c)
            accum_c.add(phi_c, X0_c)
        Cov_u, Sig_u, _ = accum_u.covariances(LAM)
        Cov_c, Sig_c, _ = accum_c.covariances(LAM)
        sigma_results['rf_uncond_cf'].append(mmse_cf(Cov_u, Sig_u, trace_p0))
        sigma_results['rf_cond_cf'].append(mmse_cf(Cov_c, Sig_c, trace_p0))

    # ---- Save ----
    os.makedirs('tables', exist_ok=True)
    save_dict = {'k_grid': np.array(K_GRID), 'sigma_grid': SIGMA_GRID,
                 'k_for_sigma_sweep': k_fix, 'sigma_values': np.array(SIGMA_VALUES),
                 'trace_p0': trace_p0}
    for sg in SIGMA_VALUES:
        for key, vals in results[sg].items():
            save_dict[f'{key}_s{sg}'] = np.array(vals)
    for key, vals in sigma_results.items():
        save_dict[f'sweep_{key}'] = np.array(vals)
    np.savez('tables/rf_gmm_sweep.npz', **save_dict)
    print("Saved tables/rf_gmm_sweep.npz")

    # ---- Plot ----
    plot(results, sigma_results, baselines_uncond, baselines_cond,
         K_GRID, SIGMA_VALUES, SIGMA_GRID, k_fix, trace_p0)


def plot(results, sigma_results, baselines_uncond, baselines_cond,
         K_GRID, SIGMA_VALUES, SIGMA_GRID, k_fix, trace_p0):
    COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, len(SIGMA_VALUES)))
    k_over_d = np.array(K_GRID) / D

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'RF Denoiser on GMM (d={D}, C={N_CLASSES}, weights={WEIGHTS})\n'
        f'N_train={N_TRAIN}, N_noise={N_NOISE}, label_noise={LABEL_NOISE}',
        fontsize=11,
    )

    # Panel (0,0): Unconditional loss vs k/d
    ax = axes[0, 0]
    for sg, color in zip(SIGMA_VALUES, COLORS):
        ax.axhline(baselines_uncond[sg]['linear_wiener'], color=color, lw=1.5, ls=':',
                   label=f'σ={sg} Wiener')
        ax.axhline(baselines_uncond[sg]['exact_mmse'], color=color, lw=1.5, ls='--',
                   label=f'σ={sg} Exact MMSE')
        ax.plot(k_over_d, results[sg]['rf_uncond_cf'],     color=color, lw=2,   ls='-',
                label=f'σ={sg} RF CF')
        ax.plot(k_over_d, results[sg]['rf_uncond_theory'], color=color, lw=2,   ls='-.',
                label=f'σ={sg} RF Theory')
    ax.set_xscale('log')
    ax.set_xlabel('k / d'); ax.set_ylabel('MSE')
    ax.set_title('Unconditional: L_σ vs k/d')
    ax.grid(True, alpha=0.3)
    # Compact legend (only one set of linestyles)
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color='k', ls=':', lw=1.5, label='Linear Wiener (baseline)'),
        Line2D([0], [0], color='k', ls='--', lw=1.5, label='Exact GMM MMSE'),
        Line2D([0], [0], color='k', ls='-', lw=2, label='RF Empirical CF'),
        Line2D([0], [0], color='k', ls='-.', lw=2, label='RF Theory (Stein+Hermite n≤3)'),
    ] + [Line2D([0],[0], color=c, lw=2, label=f'σ={s}')
         for s, c in zip(SIGMA_VALUES, COLORS)]
    ax.legend(handles=legend_els, fontsize=7, ncol=2)

    # Panel (0,1): Conditional loss vs k/d
    ax = axes[0, 1]
    for sg, color in zip(SIGMA_VALUES, COLORS):
        ax.axhline(baselines_cond[sg]['cond_wiener'], color=color, lw=1.5, ls=':')
        ax.axhline(baselines_cond[sg]['exact_mmse_cond'], color=color, lw=1.5, ls='--')
        ax.plot(k_over_d, results[sg]['rf_cond_cf'],     color=color, lw=2, ls='-')
        ax.plot(k_over_d, results[sg]['rf_cond_theory'], color=color, lw=2, ls='-.')
    ax.set_xscale('log')
    ax.set_xlabel('k / d'); ax.set_ylabel('MSE')
    ax.set_title('Conditional (U=class): L_{σ,U} vs k/d')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_els, fontsize=7, ncol=2)

    # Panel (1,0): Unconditional loss vs sigma (fixed k)
    ax = axes[1, 0]
    ax.plot(SIGMA_GRID, sigma_results['linear_wiener'],    'k:',  lw=1.5, label='Linear Wiener')
    ax.plot(SIGMA_GRID, sigma_results['exact_mmse'],       'k--', lw=1.5, label='Exact GMM MMSE')
    ax.plot(SIGMA_GRID, sigma_results['rf_uncond_cf'],     'b-',  lw=2,   label='RF CF (empirical)')
    ax.plot(SIGMA_GRID, sigma_results['rf_uncond_theory'], 'b-.', lw=2,   label='RF Theory')
    ax.axhline(trace_p0, color='gray', ls=':', lw=1, label='Tr(Σ_p0)')
    ax.set_xscale('log')
    ax.set_xlabel('σ'); ax.set_ylabel('MSE')
    ax.set_title(f'Unconditional: L_σ vs σ  (k={k_fix})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel (1,1): Conditional loss vs sigma (fixed k)
    ax = axes[1, 1]
    ax.plot(SIGMA_GRID, sigma_results['cond_wiener'],       'k:',  lw=1.5, label='Cond Wiener')
    ax.plot(SIGMA_GRID, sigma_results['exact_mmse_cond'],   'k--', lw=1.5, label='Exact cond MMSE')
    ax.plot(SIGMA_GRID, sigma_results['rf_cond_cf'],        'r-',  lw=2,   label='RF cond CF')
    ax.plot(SIGMA_GRID, sigma_results['rf_cond_theory'],    'r-.', lw=2,   label='RF cond Theory')
    ax.axhline(trace_p0, color='gray', ls=':', lw=1, label='Tr(Σ_p0)')
    ax.set_xscale('log')
    ax.set_xlabel('σ'); ax.set_ylabel('MSE')
    ax.set_title(f'Conditional: L_{{σ,U}} vs σ  (k={k_fix})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = 'figures/rf_gmm_sweep.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
