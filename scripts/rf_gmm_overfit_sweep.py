"""
rf_gmm_overfit_sweep.py

For each N_train in {8, 64, 128, 256, 1024, 50000}:
  Treat those N_train samples as the TARGET DISTRIBUTION (empirical measure).
  Show MMSE curves vs k at fixed sigma values.

Methods compared (all using only the finite training data, NOT the analytical GMM):
  1. RF empirical (direct regression): fit W* to x0_train + noise, eval on x0_train + fresh noise
  2. RF theory (empirical-Stein): Stein formula for Cov + Hermite for Sigma_phi, closed-form
  3. Empirical linear Wiener: Tr(sigma^2 * Sigma_emp / (Sigma_emp + sigma^2 I))
  4. NW MMSE (Nadaraya-Watson): optimal denoiser for the finite dataset (for N <= NW_MAX_N)
  5. (Reference) GMM population baselines: linear Wiener, exact MMSE, cond Wiener

Key insight: when k >> N_train, the RF denoiser can overfit/memorize the N_train samples.
NW MMSE is the oracle lower bound (infinite k, optimal kernel smoother).

Produces one figure per N_train: 2 rows (uncond/cond) x 4 cols (sigma).

Outputs:
  figures/rf_gmm_overfit_N{n}.png   — one per N_train
  tables/rf_gmm_overfit_sweep.npz

Usage:
  python scripts/rf_gmm_overfit_sweep.py
  K_MAX=4096 N_NOISE=5 python scripts/rf_gmm_overfit_sweep.py
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

N_TRAIN_VALUES = [int(x) for x in
    os.environ.get('N_TRAIN_VALUES', '8,64,128,256,1024,50000').split(',')]

# k grid: powers of 2 from D up to K_MAX
K_GRID = [2**i for i in range(3, 16) if 2**i <= K_MAX]

SIGMA_VALUES = [float(s) for s in
    os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]

# Nadaraya-Watson is O(N^2) — only feasible for small N
NW_MAX_N    = int(os.environ.get('NW_MAX_N',    '2000'))
N_NOISE_NW  = int(os.environ.get('N_NOISE_NW',  '20'))
N_MC_EXACT  = int(os.environ.get('N_MC_EXACT',  '200000'))


# ---------------------------------------------------------------------------
# GMM definition (same as rf_gmm_sweep.py)
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
# Empirical baselines (use only x0_train, not GMM analytical)
# ---------------------------------------------------------------------------

def mmse_linear_emp(x0_train, sigma):
    """Linear Wiener MMSE using empirical sample covariance."""
    N = x0_train.shape[0]
    mu_emp = x0_train.mean(0)
    Sigma_emp = (x0_train - mu_emp[None, :]).T @ (x0_train - mu_emp[None, :]) / max(N - 1, 1)
    eigvals = np.linalg.eigvalsh(Sigma_emp)
    eigvals = np.maximum(eigvals, 0)
    sig2 = sigma ** 2
    return float((sig2 * eigvals / (eigvals + sig2)).sum())


def mmse_nw(x0_train, sigma, n_noise=N_NOISE_NW, rng=None):
    """
    Nadaraya-Watson MMSE on the empirical distribution.

    Evaluates E_{x0~emp, z~N(0,I)}[||D_NW(x0+sigma*z) - x0||^2]
    where D_NW is the kernel smoother with bandwidth sigma:
      D_NW(y) = sum_j x_j * exp(-||y-x_j||^2/(2*sigma^2)) / normalizer

    Complexity: O(N^2 * d * n_noise).  Only call for N <= NW_MAX_N.
    """
    if rng is None:
        rng = np.random.default_rng()
    N, d = x0_train.shape
    total_mse = 0.0
    for _ in range(n_noise):
        z = rng.standard_normal((N, d))
        y = x0_train + sigma * z  # (N, d)
        # log_w[i, j] = -||y_i - x_j||^2 / (2*sigma^2)
        diff = y[:, None, :] - x0_train[None, :, :]   # (N, N, d)
        log_w = -np.einsum('ijk,ijk->ij', diff, diff) / (2 * sigma**2)  # (N, N)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        D_star = w @ x0_train   # (N, d)
        total_mse += float(np.sum((D_star - x0_train)**2))
    return total_mse / (n_noise * N)


# ---------------------------------------------------------------------------
# RF direct regression MMSE  (the "RF computation" Binxu wants)
# ---------------------------------------------------------------------------

def direct_rf_mmse(x0_train, U_train, Theta, Gamma, sigma, lam, n_noise, rng,
                   conditional=True):
    """
    Fit RF denoiser W* to x0_train + training noise.
    Evaluate on x0_train + independent fresh noise.
    Returns average MSE per training sample (sum over d, mean over N).

    Uses Woodbury when N < k (far overparameterized) for efficiency.
    """
    N, d = x0_train.shape
    k = Theta.shape[0]
    use_woodbury = (N < k)

    total_mse = 0.0
    for _ in range(n_noise):
        # --- Training noise ---
        z_tr = rng.standard_normal((N, d))
        y_tr = x0_train + sigma * z_tr
        if conditional:
            pre_tr = y_tr @ Theta.T + U_train @ Gamma.T
        else:
            pre_tr = y_tr @ Theta.T
        phi_tr = np.maximum(pre_tr, 0)   # (N, k)

        # --- Evaluation noise (independent) ---
        z_ev = rng.standard_normal((N, d))
        y_ev = x0_train + sigma * z_ev
        if conditional:
            pre_ev = y_ev @ Theta.T + U_train @ Gamma.T
        else:
            pre_ev = y_ev @ Theta.T
        phi_ev = np.maximum(pre_ev, 0)   # (N, k)

        # --- Solve ---
        if use_woodbury:
            # W = phi_tr.T @ (phi_tr @ phi_tr.T / N + lam I)^{-1} @ x0 / N
            # x0_hat = phi_ev @ W = (phi_ev @ phi_tr.T) @ alpha / N
            # alpha = (phi_tr @ phi_tr.T / N + lam I)^{-1} @ x0  (N, d)
            A = phi_tr @ phi_tr.T / N + lam * np.eye(N)   # (N, N)
            alpha = np.linalg.solve(A, x0_train)           # (N, d)
            x0_hat = (phi_ev @ phi_tr.T) @ alpha / N       # (N, d)
        else:
            # W = (phi_tr.T @ phi_tr / N + lam I)^{-1} @ (phi_tr.T @ x0 / N)
            A = phi_tr.T @ phi_tr / N + lam * np.eye(k)   # (k, k)
            W = np.linalg.solve(A, phi_tr.T @ x0_train / N)  # (k, d)
            x0_hat = phi_ev @ W                             # (N, d)

        total_mse += float(np.sum((x0_hat - x0_train)**2))

    return total_mse / (n_noise * N)


# ---------------------------------------------------------------------------
# RF theory: empirical Stein formula (no GMM analytical)
# ---------------------------------------------------------------------------

def mmse_theory_emp(x0_train, U_train, mu_emp, Theta, Gamma, trace_p0, sigma, lam,
                    conditional=True):
    """
    RF theory MMSE using empirical-Stein Cov + Hermite Sigma_phi.

    For the empirical distribution (Dirac masses at x0_train):
      Cov(x0_i, phi_j) = (1/N) sum_n (x0_n - mu)_i * c0(m_nj, s_j)
      where m_nj = theta_j^T x0_n + gamma_j^T U_n  (conditional)
            s_j  = sigma * ||theta_j||

    Sigma_phi uses the Hermite expansion (data + noise parts), also fully empirical.
    """
    N, d = x0_train.shape
    k = Theta.shape[0]
    X0_c = x0_train - mu_emp[None, :]
    s = sigma * np.linalg.norm(Theta, axis=1)   # (k,)

    if conditional:
        M = x0_train @ Theta.T + U_train @ Gamma.T   # (N, k)
    else:
        M = x0_train @ Theta.T                        # (N, k)

    # Empirical Stein Cov(x0, phi) = (1/N) (X0 - mu)^T * c0(M, s)
    G = _c0(M, s[None, :])          # (N, k) = E_z[phi_j | x0_n, U_n]
    Cov_x0_phi = X0_c.T @ G / N    # (d, k)

    # Sigma_phi (Hermite expansion, empirical)
    z = M / np.maximum(s[None, :], 1e-12)
    Phi_z = scipy_norm.cdf(z)
    phi_z = scipy_norm.pdf(z)

    G_c = G - G.mean(0)
    Sig_data = G_c.T @ G_c / N    # (k, k)

    C1 = s[None, :] * Phi_z        # (N, k)
    C2 = s[None, :] * phi_z / 2.0  # (N, k)
    C3 = M * phi_z / 6.0           # (N, k)
    norms_T = np.linalg.norm(Theta, axis=1)
    Theta_n = Theta / norms_T[:, None]
    rho = np.clip(Theta_n @ Theta_n.T, -1 + 1e-6, 1 - 1e-6)   # (k, k)
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
# Main
# ---------------------------------------------------------------------------

def main():
    rng_global = np.random.default_rng(SEED)
    print(f"Config: d={D}, C={N_CLASSES}, K_MAX={K_MAX}, N_NOISE={N_NOISE}")
    print(f"N_TRAIN_VALUES: {N_TRAIN_VALUES}")
    print(f"SIGMA_VALUES: {SIGMA_VALUES}")
    print(f"NW_MAX_N={NW_MAX_N},  N_NOISE_NW={N_NOISE_NW}")

    gmm = make_gmm(SEED)

    # GMM population baselines (for reference only)
    print("\nComputing GMM population baselines (reference) ...")
    pop_baselines = {}
    for sigma in SIGMA_VALUES:
        pop_baselines[sigma] = {
            'linear_wiener': gmm.mmse_uncond_wiener(sigma),
            'cond_wiener':   gmm.mmse_cond_wiener(sigma),
            'exact_mmse':    gmm.mmse_uncond_exact(sigma, N_mc=N_MC_EXACT,
                                 rng=np.random.default_rng(SEED + 1)),
        }
        print(f"  sigma={sigma:.2f}: "
              f"wiener={pop_baselines[sigma]['linear_wiener']:.3f}, "
              f"exact={pop_baselines[sigma]['exact_mmse']:.3f}, "
              f"cond_wiener={pop_baselines[sigma]['cond_wiener']:.3f}")

    # Shared projection cache (same projections across all N_train)
    Theta_cache = {}
    Gamma_cache = {}
    rng_proj = np.random.default_rng(SEED + 100)
    for k in K_GRID:
        Theta_cache[k] = rng_proj.standard_normal((k, D)) / np.sqrt(D)
        Gamma_cache[k] = rng_proj.standard_normal((k, N_CLASSES)) / np.sqrt(N_CLASSES)

    all_results = {}

    for N_train in N_TRAIN_VALUES:
        print(f"\n=== N_train={N_train} ===")
        rng = np.random.default_rng(SEED + N_train)

        # Fixed training set
        x0_train, labels_train, U_train = gmm.sample(N_train, rng=rng)
        mu_emp = x0_train.mean(0)
        X0_c   = x0_train - mu_emp[None, :]
        trace_p0_emp = float(np.trace(X0_c.T @ X0_c / max(N_train - 1, 1)))
        print(f"  Empirical Tr(Sigma_p0): {trace_p0_emp:.4f}  (GMM true: {np.trace(gmm.Sigma):.4f})")

        # ---- Empirical baselines (only depend on x0_train, not k) ----
        emp_baselines = {}
        for sigma in SIGMA_VALUES:
            bl = {
                'linear_wiener': mmse_linear_emp(x0_train, sigma),
            }
            if N_train <= NW_MAX_N:
                rng_nw = np.random.default_rng(SEED + N_train + 777)
                bl['nw_mmse'] = mmse_nw(x0_train, sigma, n_noise=N_NOISE_NW, rng=rng_nw)
                print(f"  sigma={sigma:.2f}: emp_lin_wiener={bl['linear_wiener']:.3f}, "
                      f"nw_mmse={bl['nw_mmse']:.3f}")
            else:
                bl['nw_mmse'] = None
                print(f"  sigma={sigma:.2f}: emp_lin_wiener={bl['linear_wiener']:.3f}, "
                      f"nw_mmse=N/A (N too large)")
            emp_baselines[sigma] = bl

        # ---- Per-k results ----
        results = {sg: {key: [] for key in [
            'rf_direct_uncond', 'rf_direct_cond',
            'rf_theory_uncond', 'rf_theory_cond',
        ]} for sg in SIGMA_VALUES}

        for k in tqdm(K_GRID, desc=f'k (N={N_train})'):
            Theta = Theta_cache[k]
            Gamma = Gamma_cache[k]

            for sigma in SIGMA_VALUES:
                # RF direct regression (the "computation")
                rng_dir = np.random.default_rng(SEED + N_train + k + int(sigma * 1000))
                rf_u_dir = direct_rf_mmse(x0_train, U_train, Theta, Gamma, sigma,
                                          LAM, N_NOISE, rng_dir, conditional=False)
                rng_dir2 = np.random.default_rng(SEED + N_train + k + int(sigma * 1000) + 1)
                rf_c_dir = direct_rf_mmse(x0_train, U_train, Theta, Gamma, sigma,
                                          LAM, N_NOISE, rng_dir2, conditional=True)
                results[sigma]['rf_direct_uncond'].append(rf_u_dir)
                results[sigma]['rf_direct_cond'].append(rf_c_dir)

                # RF theory (empirical Stein)
                rf_u_th = mmse_theory_emp(x0_train, U_train, mu_emp, Theta, Gamma,
                                          trace_p0_emp, sigma, LAM, conditional=False)
                rf_c_th = mmse_theory_emp(x0_train, U_train, mu_emp, Theta, Gamma,
                                          trace_p0_emp, sigma, LAM, conditional=True)
                results[sigma]['rf_theory_uncond'].append(rf_u_th)
                results[sigma]['rf_theory_cond'].append(rf_c_th)

        all_results[N_train] = {
            'results':       results,
            'emp_baselines': emp_baselines,
            'trace_p0_emp':  trace_p0_emp,
        }

        # Save intermediate figure
        plot_one(N_train, results, emp_baselines, pop_baselines, trace_p0_emp,
                 K_GRID, SIGMA_VALUES)

    # ---- Save tables ----
    os.makedirs('tables', exist_ok=True)
    save_dict = {
        'k_grid':        np.array(K_GRID),
        'sigma_values':  np.array(SIGMA_VALUES),
        'n_train_values': np.array(N_TRAIN_VALUES),
    }
    for N_train, d in all_results.items():
        save_dict[f'trace_p0_emp_N{N_train}'] = d['trace_p0_emp']
        for sg in SIGMA_VALUES:
            for key, vals in d['results'][sg].items():
                save_dict[f'{key}_N{N_train}_s{sg}'] = np.array(vals)
            bl = d['emp_baselines'][sg]
            save_dict[f'emp_lin_wiener_N{N_train}_s{sg}'] = bl['linear_wiener']
            if bl['nw_mmse'] is not None:
                save_dict[f'nw_mmse_N{N_train}_s{sg}'] = bl['nw_mmse']
    for sg in SIGMA_VALUES:
        for bkey, bval in pop_baselines[sg].items():
            save_dict[f'pop_{bkey}_s{sg}'] = bval
    np.savez('tables/rf_gmm_overfit_sweep.npz', **save_dict)
    print("\nSaved tables/rf_gmm_overfit_sweep.npz")


# ---------------------------------------------------------------------------
# Plotting: one figure per N_train
# ---------------------------------------------------------------------------

def plot_one(N_train, results, emp_baselines, pop_baselines, trace_p0_emp,
             K_GRID, SIGMA_VALUES):
    k_over_d = np.array(K_GRID) / D
    n_sigma = len(SIGMA_VALUES)

    fig, axes = plt.subplots(2, n_sigma, figsize=(5 * n_sigma, 8))
    fig.suptitle(
        f'RF Denoiser — treating N_train={N_train} samples as the distribution\n'
        f'd={D}, C={N_CLASSES},  Tr(Σ_emp)={trace_p0_emp:.3f},  λ={LAM}',
        fontsize=11,
    )

    for col, sigma in enumerate(SIGMA_VALUES):
        bl_emp = emp_baselines[sigma]
        bl_pop = pop_baselines[sigma]

        for row, (mode, direct_key, theory_key, pop_key) in enumerate([
            ('Uncond', 'rf_direct_uncond', 'rf_theory_uncond', 'linear_wiener'),
            ('Cond (U=class)', 'rf_direct_cond', 'rf_theory_cond', 'cond_wiener'),
        ]):
            ax = axes[row, col]

            # --- GMM population baselines (dashed, for reference) ---
            ax.axhline(bl_pop['linear_wiener'], color='gray', ls=':', lw=1.2,
                       label='Pop. linear Wiener', alpha=0.7)
            ax.axhline(bl_pop['exact_mmse'], color='gray', ls='--', lw=1.2,
                       label='Pop. exact MMSE', alpha=0.7)
            if row == 1:
                ax.axhline(bl_pop['cond_wiener'], color='gray', ls='-.', lw=1.2,
                           label='Pop. cond Wiener', alpha=0.7)

            # --- Empirical linear Wiener (orange) ---
            ax.axhline(bl_emp['linear_wiener'], color='darkorange', ls='-', lw=1.5,
                       label='Emp. linear Wiener')

            # --- NW MMSE (green) — oracle for finite dataset ---
            if bl_emp['nw_mmse'] is not None:
                ax.axhline(bl_emp['nw_mmse'], color='forestgreen', ls='-', lw=1.5,
                           label='NW MMSE (oracle)')

            # --- RF direct regression (blue, solid) ---
            ax.plot(k_over_d, results[sigma][direct_key], color='steelblue', lw=2,
                    ls='-', label='RF empirical (direct)')

            # --- RF theory empirical-Stein (red, dashed) ---
            ax.plot(k_over_d, results[sigma][theory_key], color='crimson', lw=1.5,
                    ls='--', label='RF theory (emp-Stein)')

            ax.set_xscale('log')
            ax.set_xlabel('k / d')
            ax.set_ylabel('MSE')
            ax.set_title(f'{mode}  σ={sigma}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=-0.01)

            if row == 0 and col == n_sigma - 1:
                ax.legend(fontsize=7, loc='upper right', ncol=1)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = f'figures/rf_gmm_overfit_N{N_train}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == '__main__':
    main()
