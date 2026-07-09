"""
Random-Feature Denoiser: Theory vs Empirical (newfile2.tex validation)

Compares three ways to compute L_sigma for phi(y) = relu(Theta y):

  1. Empirical closed-form: stream covariance stats over noise draws, plug into
     L = Tr(Sigma_p0) - Tr(C Sigma_phi^{-1} C^T). Never stores full (N*n_noise, k).

  2. Empirical direct: compute W* = C_hat Sigma_phi_hat^{-1} from train data,
     evaluate ||W* phi(y_test) + b* - x0_test||^2 on held-out test set.

  3. Theory (exact): uses actual data distribution for x0 expectations (no Gaussian
     p0 assumption). Only the noise Z integral uses Hermite expansion (exact since Z
     is Gaussian). See newfile2.tex §1 and docs/rf_methods.md.

K is configurable via env var K (default 512). Use K=8192 for k > d experiments.
"""

import sys, os, argparse
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
# Config (overridable via env vars)
# ---------------------------------------------------------------------------
STORE    = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
DATASETS = os.path.join(STORE, 'Datasets')

def _get_device():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        torch.zeros(1).cuda()
        return 'cuda'
    except RuntimeError:
        return 'cpu'

device = _get_device()

N_TRAIN   = int(os.environ.get('N_TRAIN', '10000'))
N_TEST    = int(os.environ.get('N_TEST',  '5000'))
K         = int(os.environ.get('K',       '512'))
N_NOISE   = int(os.environ.get('N_NOISE', '5'))
LAM       = float(os.environ.get('LAM',   '1e-4'))
N_SIGMA   = int(os.environ.get('N_SIGMA', '25'))
SIGMA_MIN = float(os.environ.get('SIGMA_MIN', '0.01'))
SIGMA_MAX = float(os.environ.get('SIGMA_MAX', '100.0'))
N_CLASSES = 10

print(f"Config: K={K}, N_TRAIN={N_TRAIN}, N_NOISE={N_NOISE}, device={device}")

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
    x = torch.stack(xs).to(device)
    y = torch.tensor(labs, device=device, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# Streaming covariance accumulator — avoids storing full (N*n_noise, k) matrix
# ---------------------------------------------------------------------------
class CovAccum:
    """
    Accumulates sufficient statistics for Cov(x0, phi) and Sigma_phi
    from multiple noise draws without stacking them.

    Statistics accumulated (all in float64 for numerical stability):
      - sum_phi:      (k,)    = sum_n phi_n
      - sum_x0_phiT: (d, k)  = sum_n x0_c_n phi_n^T
      - sum_phiphiT: (k, k)  = sum_n phi_n phi_n^T
      - n_total:     int     = total number of (x0, phi) pairs seen
    """
    def __init__(self, d, k, device):
        self.d = d
        self.k = k
        self.device = device
        self.sum_phi      = torch.zeros(k, device=device, dtype=torch.float64)
        self.sum_x0_phiT  = torch.zeros(d, k, device=device, dtype=torch.float64)
        self.sum_phiphiT  = torch.zeros(k, k, device=device, dtype=torch.float64)
        self.n_total      = 0

    @torch.no_grad()
    def add(self, phi, x0_c):
        """phi: (N, k), x0_c: (N, d) centered x0."""
        phi64 = phi.double()
        x0_c64 = x0_c.double()
        self.sum_phi     += phi64.sum(0)
        self.sum_x0_phiT += x0_c64.T @ phi64
        self.sum_phiphiT += phi64.T @ phi64
        self.n_total     += phi.shape[0]

    @torch.no_grad()
    def covariances(self, lam):
        """Returns (Cov_x0_phi, Sigma_phi) in float32 with ridge lam."""
        n = self.n_total
        mu_phi = self.sum_phi / n                              # (k,)
        Cov_x0_phi = (self.sum_x0_phiT / n).float()           # (d, k)
        Sigma_phi  = (self.sum_phiphiT / n
                      - mu_phi.unsqueeze(1) * mu_phi.unsqueeze(0)
                      + lam * torch.eye(self.k, device=self.device,
                                        dtype=torch.float64)).float()  # (k, k)
        return Cov_x0_phi, Sigma_phi, mu_phi.float()


# ---------------------------------------------------------------------------
# MMSE from pre-computed covariances
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_from_covs(trace_p0, Cov_x0_phi, Sigma_phi):
    """L = Tr(Sigma_p0) - Tr(C Sigma_phi^{-1} C^T)."""
    A = torch.linalg.solve(Sigma_phi, Cov_x0_phi.T)   # (k, d)
    explained = float(torch.trace(Cov_x0_phi @ A))
    return max(0.0, trace_p0 - explained)


# ---------------------------------------------------------------------------
# Empirical direct: compute W* from stats, evaluate on test features
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_empirical_direct(Cov_x0_phi, Sigma_phi, mu_phi, mu_x0,
                           Phi_test, X0_test):
    W_star = torch.linalg.solve(Sigma_phi, Cov_x0_phi.T).T  # (d, k)
    b_star = mu_x0 - W_star @ mu_phi
    pred = (Phi_test - mu_phi) @ W_star.T + mu_x0            # (N_test, d)
    mse  = float(((pred - X0_test) ** 2).sum(1).mean(0))
    return mse


# ---------------------------------------------------------------------------
# Theory: empirical p0 + Hermite for noise (no Gaussian p0 assumption)
# Follows newfile2.tex §1 exactly. See docs/rf_methods.md.
# ---------------------------------------------------------------------------
@torch.no_grad()
def mmse_theory_uncond(Theta, x0_train, trace_p0, sigma, lam):
    """
    Exact theory for unconditional RF phi(y) = relu(Theta y).
    No Gaussian assumption on p0 (uses actual data for E_{x0}).
    Hermite expansion only for noise Z (which IS Gaussian — exact).

    From newfile2.tex §1:
      g_j(x0) = c0(M_j, s_j) = M_j Phi(z_j) + s_j phi(z_j)   [Gaussian-blurred ReLU]
      c1(M_j, s_j) = (1/1!) E[f(xi) He_1(xi)] = s_j Phi(z_j)  [He_1 = xi]
      c2(M_j, s_j) = (1/2!) E[f(xi) He_2(xi)] = s_j phi(z_j)/2 [He_2 = xi^2-1]
      rho_ij = Corr(xi_i, xi_j) = theta_i^T theta_j / (||theta_i|| ||theta_j||)

      Cov(x0, phi)_{ij} = E_{x0}[(x0_i - mu_i) g_j(x0)]
      Sigma_phi,ij = Cov_{x0}(g_i, g_j) + E_{x0}[sum_{n>=1} n! rho^n c_n_i c_n_j]
    """
    N, d = x0_train.shape
    k = Theta.shape[0]
    dev = x0_train.device

    mu_x0 = x0_train.mean(0)
    X0_c  = x0_train - mu_x0                    # (N, d)

    # M_j^(n) = theta_j^T x0^(n) — noiseless pre-activation mean
    M = x0_train @ Theta.T                       # (N, k)

    # s_j = sigma * ||theta_j|| — noise std in pre-activation
    s = sigma * (Theta * Theta).sum(1).sqrt()    # (k,)
    z = M / s.unsqueeze(0)                       # (N, k)

    z_np  = z.cpu().float().numpy()
    Phi_z = torch.tensor(scipy_norm.cdf(z_np), device=dev, dtype=torch.float32)  # (N,k)
    phi_z = torch.tensor(scipy_norm.pdf(z_np), device=dev, dtype=torch.float32)  # (N,k)

    # g_j^(n) = c0(M_j, s_j) = M_j Phi(z_j) + s_j phi(z_j)
    G  = M * Phi_z + s.unsqueeze(0) * phi_z     # (N, k)
    # Hermite coefficients c1, c2 per sample per feature
    C1 = s.unsqueeze(0) * Phi_z                  # (N, k)   c1 = s Phi(z)
    C2 = s.unsqueeze(0) * phi_z / 2.0            # (N, k)   c2 = s phi(z) / 2

    # Cov(x0, phi)_{ij} = E_{x0}[(x0_i - mu_i) g_j(x0)]
    Cov_x0_phi = X0_c.T @ G / N                 # (d, k)   [centered by X0_c, tower property]

    # Sigma_phi = Cov_{x0}(G) + noise Hermite terms
    G_c       = G - G.mean(0)
    Sig_data  = G_c.T @ G_c / N                  # (k, k)   Cov_{x0}(g_i, g_j)

    # rho_ij = theta_i^T theta_j / (||theta_i|| ||theta_j||)
    # [sigma cancels: s_i s_j = sigma^2 ||theta_i|| ||theta_j||]
    norm = (Theta * Theta).sum(1).sqrt()          # (k,)
    rho  = (Theta @ Theta.T) / (norm.unsqueeze(1) * norm.unsqueeze(0))  # (k, k)
    rho  = rho.clamp(-1 + 1e-7, 1 - 1e-7)

    # E_{x0}[n! rho^n c_n_i c_n_j] for n=1,2
    EC1C1 = C1.T @ C1 / N                        # E[c1_i c1_j]
    EC2C2 = C2.T @ C2 / N                        # E[c2_i c2_j]
    Sig_noise = rho * EC1C1 + 2.0 * rho**2 * EC2C2

    Sigma_phi = Sig_data + Sig_noise + lam * torch.eye(k, device=dev)

    return mmse_from_covs(trace_p0, Cov_x0_phi, Sigma_phi)


@torch.no_grad()
def mmse_theory_cond(Theta, Gamma, x0_train, U_train, trace_p0, sigma, lam):
    """
    Theory for conditional RF: phi^U_j = relu(theta_j x0 + gamma_j U + s_j xi).
    Same structure: M^U_j = theta_j x0 + gamma_j U, s_j = sigma ||theta_j||.
    rho_ij = theta_i^T theta_j / (||theta_i|| ||theta_j||) unchanged (no noise from U).
    """
    N, d = x0_train.shape
    k = Theta.shape[0]
    dev = x0_train.device

    mu_x0 = x0_train.mean(0)
    X0_c  = x0_train - mu_x0

    M = x0_train @ Theta.T + U_train @ Gamma.T   # (N, k)
    s = sigma * (Theta * Theta).sum(1).sqrt()     # (k,)
    z = M / s.unsqueeze(0)                        # (N, k)

    z_np  = z.cpu().float().numpy()
    Phi_z = torch.tensor(scipy_norm.cdf(z_np), device=dev, dtype=torch.float32)
    phi_z = torch.tensor(scipy_norm.pdf(z_np), device=dev, dtype=torch.float32)

    G  = M * Phi_z + s.unsqueeze(0) * phi_z
    C1 = s.unsqueeze(0) * Phi_z
    C2 = s.unsqueeze(0) * phi_z / 2.0

    Cov_x0_phi = X0_c.T @ G / N

    G_c       = G - G.mean(0)
    Sig_data  = G_c.T @ G_c / N

    norm = (Theta * Theta).sum(1).sqrt()
    rho  = (Theta @ Theta.T) / (norm.unsqueeze(1) * norm.unsqueeze(0))
    rho  = rho.clamp(-1 + 1e-7, 1 - 1e-7)

    EC1C1 = C1.T @ C1 / N
    EC2C2 = C2.T @ C2 / N
    Sig_noise = rho * EC1C1 + 2.0 * rho**2 * EC2C2

    Sigma_phi = Sig_data + Sig_noise + lam * torch.eye(k, device=dev)

    return mmse_from_covs(trace_p0, Cov_x0_phi, Sigma_phi)


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
    mu_x0   = x0_train.mean(0)
    X0_c    = x0_train - mu_x0
    Sigma_p0 = X0_c.T @ X0_c / (N_TRAIN - 1)
    trace_p0 = float(Sigma_p0.diagonal().sum())
    # Use numpy eigh (avoids MKL SSYEVD bug on CPU with large matrices)
    import numpy as np
    eigvals_np = np.linalg.eigvalsh(Sigma_p0.cpu().numpy())
    eigvals = torch.tensor(eigvals_np, dtype=torch.float32, device=device).clamp(min=0)

    # Pre-compute per-class eigenvalues for conditional linear Wiener baseline
    # L_cond_wiener = (1/C) sum_c sum_k sigma^2 lambda_{c,k} / (lambda_{c,k} + sigma^2)
    # Use kernel trick: eigvals of (Nc x Nc) kernel = eigvals of (d x d) class cov
    print("Pre-computing per-class eigenvalues for conditional Wiener ...")
    class_eigvals = []
    for c in range(N_CLASSES):
        mask = (y_train == c)
        Xc = x0_train[mask]
        Nc = Xc.shape[0]
        Xc_c = (Xc - Xc.mean(0)).cpu().numpy().astype(np.float64)
        # Kernel matrix eigenvalues (Nc x Nc, much cheaper than d x d)
        kern_c = (Xc_c @ Xc_c.T) / (Nc - 1)
        ev = np.linalg.eigvalsh(kern_c)
        class_eigvals.append(torch.tensor(ev, dtype=torch.float32, device=device).clamp(min=0))
    print(f"  Class sizes: {[int((y_train==c).sum()) for c in range(N_CLASSES)]}")

    # Fixed random feature matrices
    Theta = torch.randn(K, d, device=device) / (d ** 0.5)        # (k, d)
    Gamma = torch.randn(K, N_CLASSES, device=device) / (N_CLASSES ** 0.5)  # (k, 10)

    sigma_grid = np.logspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), N_SIGMA)

    res = {k: [] for k in [
        'linear_wiener',
        'linear_wiener_cond',
        'rf_uncond_empirical_cf',
        'rf_uncond_empirical_dir',     # W* on train, eval on test images + fresh noise
        'rf_uncond_empirical_train',   # W* on train, eval on SAME train images + fresh noise (like oracle bayes)
        'rf_uncond_theory',
        'rf_cond_empirical_cf',
        'rf_cond_empirical_dir',
        'rf_cond_empirical_train',
        'rf_cond_theory',
    ]}

    print(f"Sweeping {N_SIGMA} sigma values with K={K}, N_NOISE={N_NOISE} ...")
    for sigma in tqdm(sigma_grid):

        # --- Unconditional linear Wiener (analytic) ---
        inv_y   = 1.0 / (eigvals + sigma**2 + 1e-6)
        L_wiener = float(trace_p0 - (eigvals**2 * inv_y).sum())
        res['linear_wiener'].append(max(0.0, L_wiener))

        # --- Conditional linear Wiener: average over classes ---
        # L_c(sigma) = sum_k sigma^2 * lambda_{c,k} / (lambda_{c,k} + sigma^2)
        sig2 = sigma ** 2
        L_wiener_cond = 0.0
        for ev_c in class_eigvals:
            L_wiener_cond += float((sig2 * ev_c / (ev_c + sig2)).sum())
        L_wiener_cond /= N_CLASSES
        res['linear_wiener_cond'].append(max(0.0, L_wiener_cond))

        # --- Accumulate empirical covariance stats (streaming — no full N*n_noise,k matrix) ---
        accum_u = CovAccum(d, K, device)   # unconditional
        accum_c = CovAccum(d, K, device)   # conditional

        for _ in range(N_NOISE):
            Z = torch.randn_like(x0_train) * sigma
            Y = x0_train + Z
            phi_u = F.relu(Y @ Theta.T)                         # (N, k)
            phi_c = F.relu(Y @ Theta.T + U_train @ Gamma.T)    # (N, k)
            accum_u.add(phi_u, X0_c)
            accum_c.add(phi_c, X0_c)
            del phi_u, phi_c, Y, Z

        Cov_u, Sig_u, mu_phi_u = accum_u.covariances(LAM)
        Cov_c, Sig_c, mu_phi_c = accum_c.covariances(LAM)

        # --- Test features: held-out test images + fresh noise ---
        Z_test = torch.randn_like(x0_test) * sigma
        Phi_test_u = F.relu((x0_test + Z_test) @ Theta.T)
        Phi_test_c = F.relu((x0_test + Z_test) @ Theta.T + U_test @ Gamma.T)
        del Z_test

        # --- Train features: SAME train images + fresh noise (like oracle bayes) ---
        Z_fresh = torch.randn_like(x0_train) * sigma
        Phi_train_fresh_u = F.relu((x0_train + Z_fresh) @ Theta.T)
        Phi_train_fresh_c = F.relu((x0_train + Z_fresh) @ Theta.T + U_train @ Gamma.T)
        del Z_fresh

        # --- Uncond empirical CF ---
        res['rf_uncond_empirical_cf'].append(mmse_from_covs(trace_p0, Cov_u, Sig_u))

        # --- Uncond empirical direct (test images) ---
        res['rf_uncond_empirical_dir'].append(
            mmse_empirical_direct(Cov_u, Sig_u, mu_phi_u, mu_x0, Phi_test_u, x0_test))

        # --- Uncond empirical train+fresh (same x0 pool, fresh noise — like oracle bayes) ---
        res['rf_uncond_empirical_train'].append(
            mmse_empirical_direct(Cov_u, Sig_u, mu_phi_u, mu_x0, Phi_train_fresh_u, x0_train))

        # --- Uncond theory ---
        res['rf_uncond_theory'].append(
            mmse_theory_uncond(Theta, x0_train, trace_p0, sigma, LAM))

        # --- Cond empirical CF ---
        res['rf_cond_empirical_cf'].append(mmse_from_covs(trace_p0, Cov_c, Sig_c))

        # --- Cond empirical direct (test images) ---
        res['rf_cond_empirical_dir'].append(
            mmse_empirical_direct(Cov_c, Sig_c, mu_phi_c, mu_x0, Phi_test_c, x0_test))

        # --- Cond empirical train+fresh (same x0 pool, fresh noise) ---
        res['rf_cond_empirical_train'].append(
            mmse_empirical_direct(Cov_c, Sig_c, mu_phi_c, mu_x0, Phi_train_fresh_c, x0_train))

        # --- Cond theory ---
        res['rf_cond_theory'].append(
            mmse_theory_cond(Theta, Gamma, x0_train, U_train, trace_p0, sigma, LAM))

        del Phi_test_u, Phi_test_c, Phi_train_fresh_u, Phi_train_fresh_c, Cov_u, Sig_u, Cov_c, Sig_c

    # --- Save ---
    os.makedirs('tables', exist_ok=True)
    tag = f'k{K}'
    np.savez(f'tables/rf_theory_vs_empirical_cifar10_{tag}.npz',
             sigma=sigma_grid, **{k: np.array(v) for k, v in res.items()})

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Random-Feature Denoiser: Theory vs Empirical (CIFAR-10, k={K})\n'
                 r'$\phi(y) = \mathrm{relu}(\Theta y)$, '
                 r'$\phi^U(y,U) = \mathrm{relu}(\Theta y + \Gamma U)$, '
                 f'd={x0_train.shape[1]}', fontsize=11)

    sg = sigma_grid

    ax = axes[0]
    ax.plot(sg, res['linear_wiener'],                'k-',  lw=2,   label='Linear Wiener (analytic)')
    ax.plot(sg, res['rf_uncond_empirical_cf'],       'b-',  lw=2,   label='RF uncond: Empirical CF (in-sample)')
    ax.plot(sg, res['rf_uncond_empirical_train'],    'b--', lw=1.5, label='RF uncond: Train x0 + fresh Z (like oracle)')
    ax.plot(sg, res['rf_uncond_empirical_dir'],      'b:',  lw=1.5, label='RF uncond: Test images + fresh Z')
    ax.plot(sg, res['rf_uncond_theory'],             'b-.', lw=2,   label='RF uncond: Theory (Hermite n≤2)')
    ax.set_xscale('log'); ax.set_xlabel('sigma'); ax.set_ylabel('MSE loss')
    ax.set_title(f'Unconditional: L_sigma  (k={K}, d={x0_train.shape[1]})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sg, res['linear_wiener_cond'],           'k-',  lw=2,   label='Cond. Linear Wiener (class eigvals)')
    ax.plot(sg, res['rf_cond_empirical_cf'],         'r-',  lw=2,   label='RF cond: Empirical CF (in-sample)')
    ax.plot(sg, res['rf_cond_empirical_train'],      'r--', lw=1.5, label='RF cond: Train x0 + fresh Z (like oracle)')
    ax.plot(sg, res['rf_cond_empirical_dir'],        'r:',  lw=1.5, label='RF cond: Test images + fresh Z')
    ax.plot(sg, res['rf_cond_theory'],               'r-.', lw=2,   label='RF cond: Theory (Hermite n≤2)')
    ax.set_xscale('log'); ax.set_xlabel('sigma'); ax.set_ylabel('MSE loss')
    ax.set_title(f'Conditional: L_sigma,U  (k={K}, d={x0_train.shape[1]})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = f'figures/rf_theory_vs_empirical_cifar10_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")

    mid = N_SIGMA // 2
    print(f"\n=== Summary at sigma={sigma_grid[mid]:.3f} ===")
    for key, v in res.items():
        print(f"  {key:40s}: {v[mid]:.4f}")


if __name__ == '__main__':
    main()
