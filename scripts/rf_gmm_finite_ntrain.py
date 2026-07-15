"""
rf_gmm_finite_ntrain.py

For each N_train in {8, 64, 128, 256, 1024, 50000}:
  Treats the N_train samples as the TARGET distribution.
  Produces one figure per N_train with 2x4 subplots (uncond/cond x sigma).

  Methods shown:
    - Linear Wiener (empirical Σ_p0)                            — horizontal dashed
    - NW exact MMSE (Nadaraya-Watson on training data, N≤1024)  — horizontal solid
    - GMM exact MMSE  (population Bayes optimal, reference)     — horizontal dotted
    - RF empirical CF (fit to training data, eval w/ fresh noise)  — solid curve
    - RF theory Joint Gaussian (Section 3 of newfile2_article.pdf) — dash-dot curve

  The theory uses ONLY second-order empirical statistics (Σ_p0, C_xU, Σ_U)
  computed from the N_train samples — no GMM component knowledge.

Usage:
  python scripts/rf_gmm_finite_ntrain.py
  K_MAX=8192 python scripts/rf_gmm_finite_ntrain.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

from core.gmm import GaussianMixture, mmse_theory_joint_gaussian

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

D         = 8
N_CLASSES = 3
WEIGHTS   = [0.5, 0.3, 0.2]
SEED      = 42
N_NOISE   = int(os.environ.get('N_NOISE', '10'))    # noise draws for empirical CF
N_NOISE_NW = int(os.environ.get('N_NOISE_NW', '30'))  # noise draws for NW MMSE
LAM       = float(os.environ.get('LAM', '1e-4'))
K_MAX     = int(os.environ.get('K_MAX', '4096'))
NW_MAX_N  = int(os.environ.get('NW_MAX_N', '1024'))   # skip NW for N > this

N_TRAIN_VALUES = [int(x) for x in os.environ.get('N_TRAIN_VALUES', '8,64,128,256,1024,50000').split(',')]
SIGMA_VALUES   = [float(s) for s in os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]
K_GRID = [2**i for i in range(3, 16) if 2**i <= K_MAX]

N_MC_EXACT = int(os.environ.get('N_MC_EXACT', '200000'))


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
    return GaussianMixture(weights=np.array(WEIGHTS), means=means,
                           covs=np.stack([S0, S1, S2]))


# ---------------------------------------------------------------------------
# Nadaraya-Watson MMSE on finite training set
# ---------------------------------------------------------------------------

def nw_mmse(x0: np.ndarray, sigma: float, n_noise: int, rng) -> float:
    """
    Optimal denoiser for the EMPIRICAL distribution (Dirac mixture at x0_i).
    D*(y) = Σ_j x0_j K(y, x0_j) / Σ_j K(y, x0_j),  K = Gaussian kernel (bandwidth sigma).
    Evaluated at y_i = x0_i + sigma*z_i with fresh z.
    """
    N, d = x0.shape
    assert N <= NW_MAX_N, "N too large for NW; skip before calling"
    total_sq = 0.0
    for _ in range(n_noise):
        z = rng.standard_normal((N, d))
        y = x0 + sigma * z                                # (N, d)
        diff = y[:, None, :] - x0[None, :, :]            # (N, N, d)
        log_w = -np.einsum('ijk,ijk->ij', diff, diff) / (2.0 * sigma ** 2)  # (N, N)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        D_star = w @ x0                                   # (N, d)
        total_sq += np.sum((D_star - x0) ** 2)
    return total_sq / (n_noise * N)


# ---------------------------------------------------------------------------
# Empirical closed-form MMSE accumulator
# ---------------------------------------------------------------------------

class CovAccum:
    def __init__(self, d, k):
        self.sum_phi     = np.zeros(k)
        self.sum_x0_phiT = np.zeros((d, k))
        self.sum_phiphiT = np.zeros((k, k))
        self.n = 0

    def add(self, phi, x0_c):
        phi   = phi.astype(np.float64)
        x0_c  = x0_c.astype(np.float64)
        self.sum_phi     += phi.sum(0)
        self.sum_x0_phiT += x0_c.T @ phi
        self.sum_phiphiT += phi.T @ phi
        self.n           += phi.shape[0]

    def mmse(self, trace_p0, lam):
        n      = self.n
        mu_phi = self.sum_phi / n
        Cov    = self.sum_x0_phiT / n
        Sig    = self.sum_phiphiT / n - np.outer(mu_phi, mu_phi) + lam * np.eye(self.sum_phiphiT.shape[0])
        try:
            A = np.linalg.solve(Sig.astype(np.float64), Cov.T.astype(np.float64))
            explained = float(np.trace(Cov @ A))
        except np.linalg.LinAlgError:
            A = np.linalg.lstsq(Sig, Cov.T, rcond=None)[0]
            explained = float(np.trace(Cov @ A))
        return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng_main = np.random.default_rng(SEED)
    print(f"Config: d={D}, C={N_CLASSES}, K_MAX={K_MAX}, N_NOISE={N_NOISE}, "
          f"N_NOISE_NW={N_NOISE_NW}, NW_MAX_N={NW_MAX_N}")
    print(f"N_TRAIN_VALUES: {N_TRAIN_VALUES}")
    print(f"SIGMA_VALUES: {SIGMA_VALUES}")
    print(f"K_GRID: {K_GRID}")

    gmm = make_gmm(SEED)
    print(f"GMM Tr(Sigma_p0): {np.trace(gmm.Sigma):.4f}")

    # Shared random projections cache (same Theta/Gamma across all N_train for fair comparison)
    rng_proj = np.random.default_rng(SEED + 100)
    Theta_cache = {}
    Gamma_cache = {}
    for k in K_GRID:
        Theta_cache[k] = rng_proj.standard_normal((k, D)) / np.sqrt(D)
        Gamma_cache[k] = rng_proj.standard_normal((k, N_CLASSES)) / np.sqrt(N_CLASSES)

    # Population GMM baselines (independent of N_train, used for reference lines)
    print("\nComputing GMM population baselines ...")
    gmm_baselines = {}
    rng_mc = np.random.default_rng(SEED + 1)
    for sigma in SIGMA_VALUES:
        gmm_baselines[sigma] = {
            'linear_wiener': gmm.mmse_uncond_wiener(sigma),
            'exact_mmse':    gmm.mmse_uncond_exact(sigma, N_mc=N_MC_EXACT, rng=rng_mc),
            'cond_wiener':   gmm.mmse_cond_wiener(sigma),
            'cond_exact':    gmm.mmse_cond_exact(sigma),
        }
        b = gmm_baselines[sigma]
        print(f"  sigma={sigma:.2f}: wiener={b['linear_wiener']:.3f}, "
              f"exact={b['exact_mmse']:.3f}, cond_wiener={b['cond_wiener']:.3f}")

    all_results = {}

    for N_train in N_TRAIN_VALUES:
        print(f"\n=== N_train={N_train} ===")
        rng_data = np.random.default_rng(SEED + N_train)
        x0_tr, labels_tr, U_tr = gmm.sample(N_train, rng=rng_data)
        rng_eval = np.random.default_rng(SEED + N_train + 999)

        # Empirical second-order statistics (for theory)
        mu_x0 = x0_tr.mean(0)
        mu_U  = U_tr.mean(0)
        denom = max(N_train - 1, 1)
        X0_c  = x0_tr - mu_x0
        U_c   = U_tr  - mu_U
        Sigma_p0_emp = X0_c.T @ X0_c / denom          # (d, d)
        C_xU_emp     = X0_c.T @ U_c  / denom          # (d, C)
        Sigma_U_emp  = U_c.T  @ U_c  / denom          # (C, C)
        trace_p0 = float(np.trace(Sigma_p0_emp))
        print(f"  Empirical Tr(Sigma_p0): {trace_p0:.4f}  (population: {np.trace(gmm.Sigma):.4f})")

        # Per-sigma baselines (only depend on empirical Sigma_p0, not k)
        emp_baselines = {}
        for sigma in SIGMA_VALUES:
            eigvals = np.linalg.eigvalsh(Sigma_p0_emp)
            lin_wiener_emp = float(np.sum(sigma**2 * eigvals / (eigvals + sigma**2)))
            nw = None
            if N_train <= NW_MAX_N:
                nw = nw_mmse(x0_tr, sigma, N_NOISE_NW, rng_eval)
            emp_baselines[sigma] = {'linear_wiener': lin_wiener_emp, 'nw': nw}
            nw_str = f"{nw:.3f}" if nw is not None else "N/A"
            print(f"  sigma={sigma:.2f}: emp_linear={lin_wiener_emp:.3f}, nw={nw_str}")

        # Sweep over k
        res = {sigma: {'rf_emp_uncond': [], 'rf_emp_cond': [],
                       'rf_theory_uncond': [], 'rf_theory_cond': []}
               for sigma in SIGMA_VALUES}

        for k in tqdm(K_GRID, desc=f'k (N={N_train})'):
            Theta = Theta_cache[k]
            Gamma = Gamma_cache[k]

            for sigma in SIGMA_VALUES:
                # --- RF theory (jointly Gaussian, Section 3) ---
                # Unconditional: Gamma=zeros, C_xU=None
                th_u = mmse_theory_joint_gaussian(
                    Sigma_p0_emp, mu_x0,
                    Theta, np.zeros_like(Gamma), sigma,
                    C_xU=None, Sigma_U=None, mu_U=None,
                    lam=LAM,
                )
                # Conditional: use empirical cross-covariance
                th_c = mmse_theory_joint_gaussian(
                    Sigma_p0_emp, mu_x0,
                    Theta, Gamma, sigma,
                    C_xU=C_xU_emp, Sigma_U=Sigma_U_emp, mu_U=mu_U,
                    lam=LAM,
                )
                res[sigma]['rf_theory_uncond'].append(th_u)
                res[sigma]['rf_theory_cond'].append(th_c)

                # --- RF empirical CF (fresh noise draws on training data) ---
                acc_u = CovAccum(D, k)
                acc_c = CovAccum(D, k)
                for _ in range(N_NOISE):
                    Z  = rng_eval.standard_normal(x0_tr.shape) * sigma
                    Y  = x0_tr + Z
                    pu = np.maximum(Y @ Theta.T, 0)
                    pc = np.maximum(Y @ Theta.T + U_tr @ Gamma.T, 0)
                    acc_u.add(pu, X0_c)
                    acc_c.add(pc, X0_c)
                res[sigma]['rf_emp_uncond'].append(acc_u.mmse(trace_p0, LAM))
                res[sigma]['rf_emp_cond'].append(acc_c.mmse(trace_p0, LAM))

        all_results[N_train] = {
            'res': res,
            'emp_baselines': emp_baselines,
            'trace_p0': trace_p0,
        }

        # Save per-N_train table
        os.makedirs('tables', exist_ok=True)
        save_d = {'k_grid': np.array(K_GRID), 'sigma_values': np.array(SIGMA_VALUES),
                  'trace_p0': trace_p0, 'N_train': N_train}
        for sg in SIGMA_VALUES:
            for key, vals in res[sg].items():
                save_d[f'{key}_s{sg}'] = np.array(vals)
            save_d[f'emp_linear_s{sg}'] = emp_baselines[sg]['linear_wiener']
            if emp_baselines[sg]['nw'] is not None:
                save_d[f'nw_s{sg}'] = emp_baselines[sg]['nw']
        np.savez(f'tables/rf_gmm_finite_N{N_train}.npz', **save_d)
        print(f"  Saved tables/rf_gmm_finite_N{N_train}.npz")

        # Plot figure for this N_train
        plot_one(N_train, res, emp_baselines, gmm_baselines, trace_p0)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one(N_train, res, emp_baselines, gmm_baselines, trace_p0):
    k_over_d = np.array(K_GRID) / D
    COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, len(SIGMA_VALUES)))

    fig, axes = plt.subplots(2, len(SIGMA_VALUES), figsize=(5 * len(SIGMA_VALUES), 9))
    fig.suptitle(
        f'RF Denoiser — Finite Dataset (N_train={N_train}, d={D}, C={N_CLASSES})\n'
        f'Theory: Section 3 jointly Gaussian (x0,U) using empirical Σ_p0, C_xU, Σ_U',
        fontsize=10,
    )

    row_labels = ['Unconditional', 'Conditional (U=class)']
    row_keys_emp    = ['rf_emp_uncond',    'rf_emp_cond']
    row_keys_theory = ['rf_theory_uncond', 'rf_theory_cond']
    row_gmm_lin     = ['linear_wiener',    'cond_wiener']
    row_gmm_exact   = ['exact_mmse',       'cond_exact']

    for row, (row_label, emp_key, th_key, gmm_lin_key, gmm_exact_key) in enumerate(
            zip(row_labels, row_keys_emp, row_keys_theory, row_gmm_lin, row_gmm_exact)):
        for col, (sigma, color) in enumerate(zip(SIGMA_VALUES, COLORS)):
            ax = axes[row, col]
            eb = emp_baselines[sigma]
            gb = gmm_baselines[sigma]

            # GMM population reference (dotted)
            ax.axhline(gb[gmm_exact_key], color='gray', lw=1, ls=':',
                       label='GMM pop. Bayes optimal')

            # Empirical linear Wiener (dashed)
            ax.axhline(eb['linear_wiener'], color=color, lw=1.5, ls='--',
                       label='Emp. linear Wiener')

            # NW exact MMSE for finite dataset (solid horizontal)
            if eb['nw'] is not None:
                ax.axhline(eb['nw'], color=color, lw=2, ls='-',
                           label='NW exact (emp. dist.)')

            # RF empirical CF (circles)
            ax.plot(k_over_d, res[sigma][emp_key], color=color, lw=2,
                    marker='o', ms=4, ls='-', label='RF empirical CF')

            # RF theory JG (dash-dot)
            ax.plot(k_over_d, res[sigma][th_key], color=color, lw=2,
                    ls='-.', label='RF theory (Joint Gauss.)')

            ax.set_xscale('log')
            ax.set_title(f'{row_label}\nσ={sigma}', fontsize=9)
            ax.set_xlabel('k / d')
            ax.set_ylabel('MSE')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = f'figures/rf_gmm_finite_N{N_train}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == '__main__':
    main()
