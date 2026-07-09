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
# Theoretical (Gaussian/Stein): unconditional
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_theory_uncond(Theta, eigvals, eigvecs, mu_x0, trace_p0, sigma, lam):
    """
    L_theory using Stein's lemma + Hermite expansion (truncated at n=2) for relu.

    Theta: (k, d) random projection
    eigvals, eigvecs: eigdecomp of Sigma_p0 = V Lambda V^T
    mu_x0: (d,) data mean
    """
    k = Theta.shape[0]
    d = Theta.shape[1]
    dev = eigvals.device

    # VtTheta: (d, k) = V^T Theta^T
    VtTheta = eigvecs.T @ Theta.T  # (d, k)

    # m_j = theta_j mu_x0  (k,)
    m = Theta @ mu_x0   # (k,)

    # S_j^2 = theta_j Sigma_p0 theta_j + sigma^2 ||theta_j||^2
    theta_Sigma_theta = (VtTheta * eigvals.unsqueeze(1) * VtTheta).sum(0)  # (k,)
    theta_norm_sq     = (Theta * Theta).sum(1)                              # (k,)
    S_sq = theta_Sigma_theta + sigma ** 2 * theta_norm_sq                  # (k,)
    S    = S_sq.clamp(min=1e-10).sqrt()                                    # (k,)

    # Hermite coefficients for relu (closed-form via Stein's identity):
    #   c0(m,S) = m Phi(m/S) + S phi(m/S)   [= E[relu]]
    #   c1(m,S) = S Phi(m/S)                 [via Stein E[relu xi] = S P(l>0)]
    #   c2(m,S) = S phi(m/S) / 2             [via Stein twice]
    z   = (m / S).cpu().numpy()
    phi_z = torch.tensor(scipy_norm.pdf(z), device=dev, dtype=eigvals.dtype)   # (k,)
    Phi_z = torch.tensor(scipy_norm.cdf(z), device=dev, dtype=eigvals.dtype)   # (k,)

    c1 = S * Phi_z         # (k,)   = c_1(m_j, S_j)
    c2 = S * phi_z / 2.0  # (k,)   = c_2(m_j, S_j)
    c0 = m * Phi_z + S * phi_z  # (k,)

    # alpha^0_j = c1_j / S_j = Phi(m_j/S_j)   [= P(l_j > 0)]
    alpha = Phi_z   # (k,)

    # Cov(x0, phi)_{:j} = Sigma_p0 theta_j alpha_j
    # Cov(x0,phi) = V Lambda (V^T Theta^T) diag(alpha)  — (d, k)
    Cov_x0_phi = eigvecs @ ((eigvals.unsqueeze(1) * VtTheta) * alpha.unsqueeze(0))  # (d,k)

    # Correlation r_tilde_ij = Cov(l_i, l_j) / (S_i S_j)
    # Cov(l_i,l_j) = theta_i Sigma_p0 theta_j + sigma^2 theta_i theta_j
    C_mat = VtTheta.T @ (eigvals.unsqueeze(1) * VtTheta) + sigma**2 * (Theta @ Theta.T)  # (k,k)
    R = C_mat / (S.unsqueeze(1) * S.unsqueeze(0))  # (k,k) correlation
    R = R.clamp(-1+1e-7, 1-1e-7)

    # Sigma_phi via Hermite truncation at n=2:
    #   Sigma_phi,ij = 1! r^1 c1_i c1_j + 2! r^2 c2_i c2_j
    Sigma_phi_th = (1.0 * R   * c1.unsqueeze(1) * c1.unsqueeze(0)
                  + 2.0 * R**2 * c2.unsqueeze(1) * c2.unsqueeze(0))  # (k,k)
    Sigma_phi_th = Sigma_phi_th + lam * torch.eye(k, device=dev, dtype=eigvals.dtype)

    # L_theory = Tr(Sigma_p0) - Tr(Cov Sigma_phi^{-1} Cov^T)
    A         = torch.linalg.solve(Sigma_phi_th, Cov_x0_phi.T)  # (k, d)
    explained = float(torch.trace(Cov_x0_phi @ A))
    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Theoretical (Gaussian/Stein): conditional  phi^U = relu(Theta y + Gamma U)
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_theory_cond(Theta, Gamma, eigvals, eigvecs, Sigma_U, C_xU, mu_x0, mu_U,
                     trace_p0, sigma, lam):
    """
    L_theory conditional using Stein's lemma (jointly Gaussian (x0, U) approx).

    Sigma_U = Cov(U),  C_xU = Cov(x0, U) (d x n_classes)
    """
    k   = Theta.shape[0]
    dev = eigvals.device

    VtTheta = eigvecs.T @ Theta.T   # (d, k)

    # m_tilde_j = theta_j mu_x0 + gamma_j mu_U  (k,)
    m_tilde = Theta @ mu_x0 + Gamma @ mu_U   # (k,)

    # S_tilde_j^2 = theta_j Sigma_p0 theta_j + 2 theta_j C_xU gamma_j
    #             + gamma_j Sigma_U gamma_j + sigma^2 ||theta_j||^2
    theta_Sigma_theta = (VtTheta * eigvals.unsqueeze(1) * VtTheta).sum(0)     # (k,)
    cross_term        = 2.0 * (Theta @ C_xU @ Gamma.T).diagonal()             # (k,)
    gamma_SU_gamma    = (Gamma @ Sigma_U @ Gamma.T).diagonal()                # (k,)
    theta_norm_sq     = (Theta * Theta).sum(1)                                 # (k,)

    S_sq = theta_Sigma_theta + cross_term + gamma_SU_gamma + sigma**2 * theta_norm_sq
    S    = S_sq.clamp(min=1e-10).sqrt()   # (k,)

    z     = (m_tilde / S).cpu().numpy()
    phi_z = torch.tensor(scipy_norm.pdf(z), device=dev, dtype=eigvals.dtype)
    Phi_z = torch.tensor(scipy_norm.cdf(z), device=dev, dtype=eigvals.dtype)

    c1 = S * Phi_z
    c2 = S * phi_z / 2.0
    alpha = Phi_z   # (k,)

    # Cov(x0, phi^U)_{:j} = (Sigma_p0 theta_j + C_xU gamma_j) alpha_j
    # A_U = (Sigma_p0 Theta^T + C_xU Gamma^T) diag(alpha)
    # Sigma_p0 Theta^T = V Lambda VtTheta  (d x k)
    # C_xU Gamma^T                         (d x k)
    SigTheta = eigvecs @ (eigvals.unsqueeze(1) * VtTheta)   # (d, k)
    CxUGamma = C_xU @ Gamma.T                               # (d, k)
    Cov_cond = (SigTheta + CxUGamma) * alpha.unsqueeze(0)   # (d, k)

    # Correlation for cond features:
    # Cov(l_i, l_j) = theta_i Sigma_p0 theta_j + theta_i C_xU gamma_j + gamma_i C_Ux theta_j
    #               + gamma_i Sigma_U gamma_j + sigma^2 theta_i theta_j
    C_mat = (VtTheta.T @ (eigvals.unsqueeze(1) * VtTheta)
           + Theta @ C_xU @ Gamma.T
           + Gamma @ C_xU.T @ Theta.T
           + Gamma @ Sigma_U @ Gamma.T
           + sigma**2 * Theta @ Theta.T)   # (k, k)
    R = C_mat / (S.unsqueeze(1) * S.unsqueeze(0))
    R = R.clamp(-1+1e-7, 1-1e-7)

    Sigma_phi_th = (1.0 * R   * c1.unsqueeze(1) * c1.unsqueeze(0)
                  + 2.0 * R**2 * c2.unsqueeze(1) * c2.unsqueeze(0))
    Sigma_phi_th = Sigma_phi_th + lam * torch.eye(k, device=dev, dtype=eigvals.dtype)

    A         = torch.linalg.solve(Sigma_phi_th, Cov_cond.T)
    explained = float(torch.trace(Cov_cond @ A))
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
    # Eigdecomp of Sigma_p0 (once)
    X0_c    = x0_train - x0_train.mean(0)
    Sigma_p0 = X0_c.T @ X0_c / (N_TRAIN - 1)
    trace_p0 = float(Sigma_p0.diagonal().sum())
    eigvals, eigvecs = torch.linalg.eigh(Sigma_p0)
    eigvals = eigvals.clamp(min=0)
    mu_x0   = x0_train.mean(0)     # (d,)

    # Class label statistics for conditional theory
    U_c     = U_train - U_train.mean(0)
    Sigma_U = U_c.T @ U_c / (N_TRAIN - 1)        # (10, 10)
    C_xU    = X0_c.T @ U_c  / (N_TRAIN - 1)      # (d, 10)
    mu_U    = U_train.mean(0)                      # (10,)

    # Fixed random feature matrices
    Theta = torch.randn(K, d, device=device) / (d ** 0.5)    # (k, d)
    Gamma = torch.randn(K, N_CLASSES, device=device) / (N_CLASSES ** 0.5)  # (k, 10)

    sigma_grid = np.logspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), N_SIGMA)

    res = {k: [] for k in [
        'linear_wiener',
        'rf_uncond_empirical_cf',   # empirical closed-form (train covariances)
        'rf_uncond_empirical_dir',  # empirical direct (W* on test)
        'rf_uncond_theory',         # Gaussian/Stein theory
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

        # --- Uncond theory (Gaussian/Stein) ---
        L_th = mmse_theory_uncond(Theta, eigvals, eigvecs, mu_x0, trace_p0, sigma, LAM)
        res['rf_uncond_theory'].append(L_th)

        # --- Cond empirical closed-form ---
        L_c_ecf = mmse_empirical_closedform(Phi_c_tr, X0_rep, LAM)
        res['rf_cond_empirical_cf'].append(L_c_ecf)

        # --- Cond empirical direct ---
        L_c_edir = mmse_empirical_direct(Phi_c_tr, X0_rep, Phi_c_test, x0_test, LAM)
        res['rf_cond_empirical_dir'].append(L_c_edir)

        # --- Cond theory (Gaussian/Stein, jointly Gaussian (x0,U)) ---
        L_c_th = mmse_theory_cond(Theta, Gamma, eigvals, eigvecs, Sigma_U, C_xU,
                                   mu_x0, mu_U, trace_p0, sigma, LAM)
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
    ax.plot(sigma, res['linear_wiener'],            'k-',   lw=2,   label='Linear Wiener (analytic)')
    ax.plot(sigma, res['rf_uncond_theory'],         'b--',  lw=2,   label='RF uncond: Theory (Stein, Hermite n≤2)')
    ax.plot(sigma, res['rf_uncond_empirical_cf'],   'b-',   lw=2,   label='RF uncond: Empirical closed-form')
    ax.plot(sigma, res['rf_uncond_empirical_dir'],  'b-o',  lw=1.5, ms=4, label='RF uncond: Empirical direct (W* on test)')
    ax.set_xscale('log')
    ax.set_xlabel('sigma'); ax.set_ylabel('MSE loss')
    ax.set_title('Unconditional: L_sigma')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Conditional
    ax = axes[1]
    ax.plot(sigma, res['linear_wiener'],            'k-',   lw=2,   label='Linear Wiener (analytic)')
    ax.plot(sigma, res['rf_cond_theory'],           'r--',  lw=2,   label='RF cond: Theory (Stein, Hermite n≤2)')
    ax.plot(sigma, res['rf_cond_empirical_cf'],     'r-',   lw=2,   label='RF cond: Empirical closed-form')
    ax.plot(sigma, res['rf_cond_empirical_dir'],    'r-o',  lw=1.5, ms=4, label='RF cond: Empirical direct (W* on test)')
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
    for k, v in res.items():
        print(f"  {k:35s}: {v[mid]:.4f}")


if __name__ == '__main__':
    main()
