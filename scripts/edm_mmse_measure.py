"""
Measure unconditional and conditional MMSE of pretrained EDM denoisers on CIFAR-10.

EDM VE preconditioning: y = x_edm + sigma_edm * Z  where x_edm in [-1,1]
Normalization: x_edm = 2 * x_pixel - 1   =>   sigma_edm = 2 * sigma_pixel

Appends 'edm_uncond' and 'edm_cond' keys to the existing cifar10 npz and
regenerates the figure.
"""

import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/edm')

import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
STORE    = os.environ.get('STORE_DIR',
           '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
CKPT_DIR = os.path.join(STORE, 'Datasets/EDM_datasets/edm_ckpts')
NPZ_PATH = 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz'
FIG_PATH = 'figures/dnn_feature_mmse_cifar10.png'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
N        = 10000
N_NOISE  = 5
BATCH    = 512


# ---------------------------------------------------------------------------
def load_cifar10(n, device):
    ds = torchvision.datasets.CIFAR10(
        root=os.path.join(STORE, 'Datasets'), train=True, download=False,
        transform=T.ToTensor())
    xs, ls = [], []
    for img, label in ds:
        xs.append(img.flatten())
        ls.append(label)
        if len(xs) >= n:
            break
    x0    = torch.stack(xs).to(device)          # (N, 3072) in [0,1]
    labels = torch.tensor(ls, device=device, dtype=torch.long)
    U      = torch.nn.functional.one_hot(labels, 10).float()
    return x0, U


@torch.no_grad()
def measure_edm_mmse(net, x0_pixel, sigma_pixel, class_labels=None,
                     n_noise=N_NOISE, batch_size=BATCH):
    """
    Measure MSE of EDM denoiser at given sigma_pixel.
    x0_pixel : (N, 3072) float32 in [0,1]
    Returns scalar MSE in [0,1] pixel units.
    """
    device    = x0_pixel.device
    N         = len(x0_pixel)
    sigma_edm = 2.0 * sigma_pixel          # pixel [0,1] → EDM [-1,1] scale
    x0_edm    = x0_pixel.reshape(N, 3, 32, 32) * 2.0 - 1.0   # [-1,1]

    total_mse, n_total = 0.0, 0
    for _ in range(n_noise):
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            x_b  = x0_edm[start:end]                         # (B,3,32,32)
            Z    = torch.randn_like(x_b) * sigma_edm
            y_b  = x_b + Z

            lab_b = None
            if class_labels is not None:
                lab_b = class_labels[start:end]

            sigma_vec = torch.full([len(x_b)], sigma_edm, device=device, dtype=torch.float32)
            D_edm    = net(y_b, sigma_vec, class_labels=lab_b)  # (B,3,32,32)
            D_pixel  = ((D_edm + 1.0) / 2.0).reshape(len(x_b), -1)   # (B,3072) in [0,1]

            x0_b    = x0_pixel[start:end]
            total_mse += float(((D_pixel - x0_b) ** 2).sum())
            n_total   += len(x_b)

    return total_mse / n_total


# ---------------------------------------------------------------------------
def main():
    print("Loading CIFAR-10 ...")
    x0, U = load_cifar10(N, DEVICE)
    print(f"  x0 shape {x0.shape}, pixel mean {x0.mean():.3f} std {x0.std():.3f}")

    print("Loading EDM checkpoints ...")
    with open(os.path.join(CKPT_DIR, 'edm-cifar10-32x32-uncond-ve.pkl'), 'rb') as f:
        net_uncond = pickle.load(f)['ema'].to(DEVICE).eval()
    with open(os.path.join(CKPT_DIR, 'edm-cifar10-32x32-cond-ve.pkl'), 'rb') as f:
        net_cond   = pickle.load(f)['ema'].to(DEVICE).eval()
    print("  Uncond label_dim:", net_uncond.label_dim)
    print("  Cond   label_dim:", net_cond.label_dim)

    # Load existing results for sigma grid
    data       = np.load(NPZ_PATH)
    results    = {k: list(data[k]) for k in data.files}
    sigma_grid = np.array(results['sigma'])
    print(f"Sigma grid: {len(sigma_grid)} points, [{sigma_grid[0]:.3f}, {sigma_grid[-1]:.3f}]")

    edm_uncond, edm_cond = [], []

    print("Measuring EDM MMSE ...")
    for sigma in tqdm(sigma_grid):
        lu = measure_edm_mmse(net_uncond, x0, sigma, class_labels=None)
        lc = measure_edm_mmse(net_cond,   x0, sigma, class_labels=U)
        edm_uncond.append(lu)
        edm_cond.append(lc)

    results['edm_uncond'] = np.array(edm_uncond)
    results['edm_cond']   = np.array(edm_cond)

    # --- summary ---
    mid = len(sigma_grid) // 2
    print(f"\n=== EDM summary at sigma={sigma_grid[mid]:.3f} ===")
    print(f"  EDM uncond: {edm_uncond[mid]:.4f}")
    print(f"  EDM cond:   {edm_cond[mid]:.4f}")
    print(f"  Linear Wiener uncond:  {results['linear_uncond'][mid]:.4f}")
    print(f"  Oracle Bayes cond LB:  {results['bayes_cond'][mid]:.4f}")

    # Save
    os.makedirs('tables', exist_ok=True)
    np.savez(NPZ_PATH, **{k: np.array(v) for k, v in results.items()})
    print(f"Saved {NPZ_PATH}")

    # --- Plot ---
    sigma  = sigma_grid
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('CIFAR-10 MMSE Denoiser Loss  (N=10000, σ in pixel [0,1] units)', fontsize=12)

    ax = axes[0]
    ax.plot(sigma, results['bayes_cond'],        'g-',   lw=2.5, label='Oracle Bayes cond (LB, same-class pool)')
    ax.plot(sigma, results['bayes_uncond'],       'b-',   lw=2.5, label='Oracle Bayes uncond (LB, all-class pool)')
    ax.plot(sigma, results['wiener_class_cond'],  'g--',  lw=2,   label='Class-cond Wiener (analytic)')
    ax.plot(sigma, results['linear_cond'],        'k--',  lw=2,   label='Linear+U Wiener (analytic)')
    ax.plot(sigma, results['linear_uncond'],      'k-',   lw=2,   label='Linear Wiener uncond (analytic)')
    ax.plot(sigma, results['edm_uncond'],         'r-o',  lw=2, ms=4, label='EDM UNet uncond (VE)')
    ax.plot(sigma, results['edm_cond'],           'r-s',  lw=2, ms=4, label='EDM UNet cond (VE)')
    ax.plot(sigma, results['resnet_uncond'],      'C0-o', lw=2, ms=4, label='ResNet18 uncond')
    ax.set_xscale('log'); ax.set_xlabel('sigma (pixel units)'); ax.set_ylabel('MSE loss')
    ax.set_title('MMSE Denoiser Loss vs sigma')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    lin_u  = np.array(results['linear_uncond'])
    lin_c  = np.array(results['linear_cond'])
    ax.plot(sigma, lin_u - np.array(results['edm_uncond']),
            'r-o', lw=2, ms=4, label='EDM uncond gain over linear Wiener')
    ax.plot(sigma, lin_c - np.array(results['edm_cond']),
            'r-s', lw=2, ms=4, label='EDM cond gain over Wiener+U')
    if 'linear_plus_dnn' in results:
        ax.plot(sigma, lin_u - np.array(results['linear_plus_dnn']),
                'm-^', lw=2, ms=4, label='[y;ResNet] combined gain over Wiener')
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.set_xscale('log'); ax.set_xlabel('sigma'); ax.set_ylabel('L_linear − L_DNN')
    ax.set_title('DNN gain over linear (positive = DNN better)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2]
    bayes_cond_arr   = np.array(results['bayes_cond'])
    bayes_uncond_arr = np.array(results['bayes_uncond'])
    ax.plot(sigma, lin_u - bayes_uncond_arr,
            'b--', lw=2, label='Oracle Bayes uncond vs linear_uncond')
    ax.plot(sigma, lin_u - bayes_cond_arr,
            'b-',  lw=2, label='Oracle Bayes cond vs linear_uncond')
    ax.plot(sigma, np.abs(bayes_uncond_arr - bayes_cond_arr),
            'b:',  lw=2.5, label='|Oracle Bayes uncond − cond| (oracle cond gain)')
    ax.plot(sigma, lin_u - np.array(results['edm_uncond']),
            'r-o', lw=2, ms=4, label='EDM uncond vs linear_uncond')
    ax.plot(sigma, lin_c - np.array(results['edm_cond']),
            'r-s', lw=2, ms=4, label='EDM cond vs linear_cond')
    ax.plot(sigma, lin_u - np.array(results['wiener_class_cond']),
            'g--', lw=2, label='Class-cond Wiener vs linear_uncond')
    ax.plot(sigma, lin_u - lin_c,
            'k:',  lw=2, label='Linear+U vs linear_uncond')
    ax.set_xscale('log'); ax.set_xlabel('sigma')
    ax.set_ylabel('L_uncond − L_method  (or conditioning gain)')
    ax.set_title('DNN and conditioning gains')
    ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3); ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved {FIG_PATH}")


if __name__ == '__main__':
    main()
