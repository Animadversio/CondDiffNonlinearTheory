"""
Random-Feature Denoiser: Theory vs Empirical (newfile2.tex validation)

Compares three ways to compute L_sigma for phi(y) = relu(Theta y):

  1. Theoretical (Gaussian/Stein): uses sample mu_x0 and Sigma_p0 as Gaussian
     parameters, then applies Stein's lemma for Cov(x0,phi) and the Hermite
     expansion for Sigma_phi (truncated at n=2).

  2. Empirical closed-form: sample covariance estimates on CIFAR training data
     plugged into L = Tr(Sigma_p0) - Tr(C Sigma_phi^{-1} C^T). Current code.

  3. Empirical direct: compute W* = C_hat Sigma_phi_hat^{-1} from train data,
     evaluate ||W* phi(y_test) + b* - x0_test||^2 on held-out test set.

Also computes the conditional versions with phi^U(y,U) = relu(Theta y + Gamma U).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STORE    = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
DATASETS = os.path.join(STORE, 'Datasets')
device   = 'cuda' if torch.cuda.is_available() else 'cpu'

N_TRAIN   = 10000   # training samples (use first N from train split)
N_TEST    = 5000    # test samples (from test split)
K         = 512     # random feature dim
N_NOISE   = 5       # noise draws per image for empirical estimate
LAM       = 1e-4    # ridge regularization
N_SIGMA   = 25
SIGMA_MIN = 0.01
SIGMA_MAX = 100.0
N_CLASSES = 10

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cifar10_flat(split, n_max, device):
    ds = torchvision.datasets.CIFAR10(
        root=DATASETS, train=(split == 'train'), download=False,
        transform=T.ToTensor())
    xs, labs = [], []
    for img, lab in ds:
        xs.append(img.flatten())
        labs.append(lab)
        if len(xs) >= n_max:
            break
    x = torch.stack(xs).to(device)     # (N, 3072)
    y = torch.tensor(labs, device=device, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# MMSE closed-form (empirical covariance estimate)
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_empirical_closedform(Phi, X0, lam):
    """L = Tr(Sig_x0) - Tr(C Sig_phi^{-1} C^T) via sample covariances."""
    N, k = Phi.shape
    N2, d = X0.shape
    assert N == N2
    X0_c  = X0  - X0.mean(0)
    Phi_c = Phi - Phi.mean(0)
    trace_p0 = float((X0_c ** 2).sum() / (N - 1))
    if k <= N:
        Sigma_phi = (Phi_c.T @ Phi_c) / (N - 1) + lam * torch.eye(k, device=Phi.device)
        Cov       = (X0_c.T  @ Phi_c) / (N - 1)      # (d, k)
        A         = torch.linalg.solve(Sigma_phi, Cov.T)
        explained = float(torch.trace(Cov @ A))
    else:
        K_mat = (Phi_c @ Phi_c.T) / (N - 1) + lam * torch.eye(N, device=Phi.device)
        M     = torch.linalg.solve(K_mat, X0_c)
        explained = float((X0_c * M).sum()) / (N - 1)
    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Empirical direct: compute W* from train, evaluate on test
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_empirical_direct(Phi_train, X0_train, Phi_test, X0_test, lam):
    """Compute W* from train covariances, evaluate MSE on test data."""
    N, k = Phi_train.shape
    X0_c   = X0_train  - X0_train.mean(0)
    Phi_c  = Phi_train - Phi_train.mean(0)
    mu_x0  = X0_train.mean(0)
    mu_phi = Phi_train.mean(0)
    if k <= N:
        Sigma_phi = (Phi_c.T @ Phi_c) / (N - 1) + lam * torch.eye(k, device=Phi_train.device)
        Cov       = (X0_c.T  @ Phi_c) / (N - 1)
        W_star    = torch.linalg.solve(Sigma_phi, Cov.T).T  # (d, k)
    else:
        K_mat  = (Phi_c @ Phi_c.T) / (N - 1) + lam * torch.eye(N, device=Phi_train.device)
        M      = torch.linalg.solve(K_mat, X0_c)  # (N, d)
        W_star = (Phi_c.T @ M) / (N - 1)          # (k, d)
        W_star = W_star.T                           # (d, k)
    b_star = mu_x0 - W_star @ mu_phi   # (d,)
    # Evaluate on test
    pred = (Phi_test - mu_phi) @ W_star.T + mu_x0  # (N_test, d)
    # sum over d dims, mean over test samples — matches Tr(Sigma_residual) scale
    mse  = float(((pred - X0_test) ** 2).sum(1).mean(0))
    return mse



# ---------------------------------------------------------------------------
# Exact theory using data distribution (no Gaussian p_0 assumption)
# ---------------------------------------------------------------------------
# Key insight from newfile2.tex: the E_{x0,U}[...] integrals can be computed
# empirically from the actual data distribution. Only the E_Z[...] (noise)
# integral needs Hermite expansion — and Z IS Gaussian so that's exact.
#
# Sigma_phi = Cov_{x0}(g) + rho * (C1^T C1 / N) + 2*rho^2 * (C2^T C2 / N)
# where rho_ij = sigma^2 theta_i theta_j / (s_i s_j) is the NOISE-ONLY correlation.
# The data variation is captured exactly in Cov_{x0}(g), no Gaussian approx needed.

@torch.no_grad()
def mmse_theory_data_uncond(Theta, x0_train, trace_p0, sigma, lam):
    """
    Exact theory for uncond RF: uses actual data distribution for x0 expectations.
    No Gaussian assumption on p0. Only Hermite expansion is for noise Z (exact).

    g_j(x0) = E_Z[relu(theta_j x0 + s_j xi)] = M_j Phi(z_j) + s_j phi(z_j)
    Cov(x0, phi)  = X0_c^T G_c / N            (empirical, exact)
    Sigma_phi = Cov_{x0}(G) + rho * (C1^T C1 / N) + 2*rho^2 * (C2^T C2 / N)
    """
    N, d = x0_train.shape
    k = Theta.shape[0]
    dev = x0_train.device

    mu_x0 = x0_train.mean(0)
    X0_c  = x0_train - mu_x0                   # (N, d)

    # Pre-activation mean for each sample: M_j^(n) = theta_j x0^(n)
    M = x0_train @ Theta.T                      # (N, k)

    # Noise std per feature: s_j = sigma * ||theta_j||
    s = sigma * (Theta * Theta).sum(1).sqrt()   # (k,)
    z = M / s.unsqueeze(0)                      # (N, k)

    z_np  = z.cpu().numpy()
    Phi_z = torch.tensor(scipy_norm.cdf(z_np), device=dev, dtype=x0_train.dtype)  # (N,k)
    phi_z = torch.tensor(scipy_norm.pdf(z_np), device=dev, dtype=x0_train.dtype)  # (N,k)

    # g_j^(n) = c0 = M_j Phi(z) + s_j phi(z)   (N, k)
    G = M * Phi_z + s.unsqueeze(0) * phi_z     # (N, k)

    # Hermite coefficients (per sample per feature)
    C1 = s.unsqueeze(0) * Phi_z                # (N, k)   c1 = s Phi(z)
    C2 = s.unsqueeze(0) * phi_z / 2.0          # (N, k)   c2 = s phi(z)/2

    # Cov(x0, phi) = X0_c^T G_c / N
    G_c = G - G.mean(0)                        # (N, k)
    Cov_x0_phi = X0_c.T @ G_c / N             # (d, k)

    # Data part of Sigma_phi: Cov_{x0}(G)
    Sig_data = G_c.T @ G_c / N                 # (k, k)

    # Noise-only correlation rho_ij = sigma^2 theta_i theta_j / (s_i s_j)
    ThetaThetaT = Theta @ Theta.T              # (k, k)
    rho = sigma**2 * ThetaThetaT / (s.unsqueeze(1) * s.unsqueeze(0))  # (k, k)
    rho = rho.clamp(-1+1e-7, 1-1e-7)

    # Noise part: Hermite n=1,2 terms
    EC1C1 = C1.T @ C1 / N                     # (k, k)   E[c1_i c1_j]
    EC2C2 = C2.T @ C2 / N                     # (k, k)   E[c2_i c2_j]
    Sig_noise = rho * EC1C1 + 2.0 * rho**2 * EC2C2

    Sigma_phi = Sig_data + Sig_noise + lam * torch.eye(k, device=dev, dtype=x0_train.dtype)

    A         = torch.linalg.solve(Sigma_phi, Cov_x0_phi.T)   # (k, d)
    explained = float(torch.trace(Cov_x0_phi @ A))
    return max(0.0, trace_p0 - explained)


@torch.no_grad()
def mmse_theory_data_cond(Theta, Gamma, x0_train, U_train, trace_p0, sigma, lam):
    """
    Exact theory for conditional RF: phi^U_j = relu(theta_j x0 + gamma_j U + s_j xi).
    Uses actual (x0, U) data distribution for all x0 expectations.
    """
    N, d = x0_train.shape
    k = Theta.shape[0]
    dev = x0_train.device

    mu_x0 = x0_train.mean(0)
    X0_c  = x0_train - mu_x0

    # Pre-activation mean: M_j^(n) = theta_j x0^(n) + gamma_j U^(n)
    M = x0_train @ Theta.T + U_train @ Gamma.T   # (N, k)

    s = sigma * (Theta * Theta).sum(1).sqrt()     # (k,)  noise-only std
    z = M / s.unsqueeze(0)                        # (N, k)

    z_np  = z.cpu().numpy()
    Phi_z = torch.tensor(scipy_norm.cdf(z_np), device=dev, dtype=x0_train.dtype)
    phi_z = torch.tensor(scipy_norm.pdf(z_np), device=dev, dtype=x0_train.dtype)

    G  = M * Phi_z + s.unsqueeze(0) * phi_z      # (N, k)  g_j^(n)
    C1 = s.unsqueeze(0) * Phi_z                   # (N, k)
    C2 = s.unsqueeze(0) * phi_z / 2.0             # (N, k)

    G_c = G - G.mean(0)
    Cov_x0_phi = X0_c.T @ G_c / N                 # (d, k)

    Sig_data  = G_c.T @ G_c / N                   # (k, k)

    ThetaThetaT = Theta @ Theta.T                 # (k, k)
    rho = sigma**2 * ThetaThetaT / (s.unsqueeze(1) * s.unsqueeze(0))
    rho = rho.clamp(-1+1e-7, 1-1e-7)

    EC1C1 = C1.T @ C1 / N
    EC2C2 = C2.T @ C2 / N
    Sig_noise = rho * EC1C1 + 2.0 * rho**2 * EC2C2

    Sigma_phi = Sig_data + Sig_noise + lam * torch.eye(k, device=dev, dtype=x0_train.dtype)

    A         = torch.linalg.solve(Sigma_phi, Cov_x0_phi.T)
    explained = float(torch.trace(Cov_x0_phi @ A))
    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    print("Loading CIFAR-10 ...")
    x0_train, y_train = load_cifar10_flat('train', N_TRAIN, device)
    x0_test,  y_test  = load_cifar10_flat('test',  N_TEST,  device)
    d = x0_train.shape[1]
    U_train = F.one_hot(y_train, N_CLASSES).float()
    U_test  = F.one_hot(y_test,  N_CLASSES).float()

    print("Pre-computing data statistics ...")
    X0_c    = x0_train - x0_train.mean(0)
    Sigma_p0 = X0_c.T @ X0_c / (N_TRAIN - 1)
    trace_p0 = float(Sigma_p0.diagonal().sum())
    # Linear Wiener baseline needs eigdecomp
    eigvals, _ = torch.linalg.eigh(Sigma_p0)
    eigvals = eigvals.clamp(min=0)

    # Fixed random feature matrices
    Theta = torch.randn(K, d, device=device) / (d ** 0.5)    # (k, d)
    Gamma = torch.randn(K, N_CLASSES, device=device) / (N_CLASSES ** 0.5)  # (k, 10)

    sigma_grid = np.logspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), N_SIGMA)

    res = {k: [] for k in [
        'linear_wiener',
        'rf_uncond_empirical_cf',   # empirical closed-form (train covariances)
        'rf_uncond_empirical_dir',  # empirical direct (W* on test)
        'rf_uncond_theory',         # theory: empirical p0, Hermite for noise only
        'rf_cond_empirical_cf',
        'rf_cond_empirical_dir',
        'rf_cond_theory',
    ]}

    print(f"Sweeping {N_SIGMA} sigma values ...")
    for sigma in tqdm(sigma_grid):

        # --- Linear Wiener baseline (analytic) ---
        inv_y = 1.0 / (eigvals + sigma**2 + 1e-6)
        L_wiener = float(trace_p0 - (eigvals**2 * inv_y).sum())
        res['linear_wiener'].append(max(0.0, L_wiener))

        # --- Accumulate feature statistics over n_noise draws ---
        Phi_train_list, X0_rep_list = [], []
        Phi_cond_train_list = []
        for _ in range(N_NOISE):
            Z   = torch.randn_like(x0_train) * sigma
            Y   = x0_train + Z                         # (N, d) noisy train
            phi_u = F.relu(Y @ Theta.T)                # (N, k)
            phi_c = F.relu(Y @ Theta.T + U_train @ Gamma.T)  # (N, k)
            Phi_train_list.append(phi_u)
            X0_rep_list.append(x0_train)
            Phi_cond_train_list.append(phi_c)
        Phi_tr = torch.cat(Phi_train_list, 0)          # (N*n_noise, k)
        Phi_c_tr = torch.cat(Phi_cond_train_list, 0)
        X0_rep   = torch.cat(X0_rep_list, 0)           # (N*n_noise, d)

        # Test features (single noise draw, fresh)
        Z_test = torch.randn_like(x0_test) * sigma
        Y_test = x0_test + Z_test
        Phi_test = F.relu(Y_test @ Theta.T)
        Phi_c_test = F.relu(Y_test @ Theta.T + U_test @ Gamma.T)

        # --- Uncond empirical closed-form ---
        L_ecf = mmse_empirical_closedform(Phi_tr, X0_rep, LAM)
        res['rf_uncond_empirical_cf'].append(L_ecf)

        # --- Uncond empirical direct ---
        L_edir = mmse_empirical_direct(Phi_tr, X0_rep, Phi_test, x0_test, LAM)
        res['rf_uncond_empirical_dir'].append(L_edir)

        # --- Uncond theory: empirical p0, Hermite for noise only ---
        L_th = mmse_theory_data_uncond(Theta, x0_train, trace_p0, sigma, LAM)
        res['rf_uncond_theory'].append(L_th)

        # --- Cond empirical closed-form ---
        L_c_ecf = mmse_empirical_closedform(Phi_c_tr, X0_rep, LAM)
        res['rf_cond_empirical_cf'].append(L_c_ecf)

        # --- Cond empirical direct ---
        L_c_edir = mmse_empirical_direct(Phi_c_tr, X0_rep, Phi_c_test, x0_test, LAM)
        res['rf_cond_empirical_dir'].append(L_c_edir)

        # --- Cond theory: empirical (x0,U) distribution, Hermite for noise only ---
        L_c_th = mmse_theory_data_cond(Theta, Gamma, x0_train, U_train, trace_p0, sigma, LAM)
        res['rf_cond_theory'].append(L_c_th)

    # --- Save ---
    os.makedirs('tables', exist_ok=True)
    np.savez('tables/rf_theory_vs_empirical_cifar10.npz',
             sigma=sigma_grid, **{k: np.array(v) for k, v in res.items()})

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Random-Feature Denoiser: Theory vs Empirical (CIFAR-10)\n'
                 r'$\phi(y) = \mathrm{relu}(\Theta y)$, '
                 r'$\phi^U(y,U) = \mathrm{relu}(\Theta y + \Gamma U)$, '
                 f'k={K}', fontsize=11)

    sigma = sigma_grid

    # Panel 1: Unconditional
    ax = axes[0]
    ax.plot(sigma, res['linear_wiener'],              'k-',   lw=2,   label='Linear Wiener (analytic)')
    ax.plot(sigma, res['rf_uncond_empirical_cf'],     'b-',   lw=2,   label='RF uncond: Empirical closed-form')
    ax.plot(sigma, res['rf_uncond_empirical_dir'],    'b-o',  lw=1.5, ms=4, label='RF uncond: Empirical direct (W* on test)')
    ax.plot(sigma, res['rf_uncond_theory'],           'b--',  lw=2,   label='RF uncond: Theory (Hermite n≤2 for noise)')
    ax.set_xscale('log')
    ax.set_xlabel('sigma'); ax.set_ylabel('MSE loss')
    ax.set_title('Unconditional: L_sigma')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Conditional
    ax = axes[1]
    ax.plot(sigma, res['linear_wiener'],              'k-',   lw=2,   label='Linear Wiener (analytic)')
    ax.plot(sigma, res['rf_cond_empirical_cf'],       'r-',   lw=2,   label='RF cond: Empirical closed-form')
    ax.plot(sigma, res['rf_cond_empirical_dir'],      'r-o',  lw=1.5, ms=4, label='RF cond: Empirical direct (W* on test)')
    ax.plot(sigma, res['rf_cond_theory'],             'r--',  lw=2,   label='RF cond: Theory (Hermite n≤2 for noise)')
    ax.set_xscale('log')
    ax.set_xlabel('sigma'); ax.set_ylabel('MSE loss')
    ax.set_title('Conditional: L_sigma,U')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = 'figures/rf_theory_vs_empirical_cifar10.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")

    # Print summary
    mid = N_SIGMA // 2
    print(f"\n=== Summary at sigma={sigma_grid[mid]:.3f} ===")
    for key, v in res.items():
        print(f"  {key:40s}: {v[mid]:.4f}")


if __name__ == '__main__':
    main()
