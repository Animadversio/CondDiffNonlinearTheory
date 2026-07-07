"""
DNN Feature MMSE Experiment — MNIST / CIFAR-10 with ResNet18.

Computes denoiser loss L_sigma as a function of sigma for:
  1. Linear baseline (Wiener filter on original pixels)
  2. Conditional linear baseline (pixels + one-hot class label)
  3. ResNet18 features, unconditional
  4. ResNet18 features + U, mode A (concat raw U)
  5. ResNet18 features + U, mode B (random nonlinear U features)
  6. ResNet18 features + U, mode C (+ FiLM-style interaction)

Design:
  - x0 stored at ORIGINAL resolution (32x32x3=3072 for CIFAR, 28x28=784 for MNIST)
    so Cov(x0, phi) is (3072 x 512) — small and fast
  - Encoder input = normalize(resize(x0)) + noise, on GPU
  - All covariance/MMSE math in torch on GPU
  - Images pre-loaded to GPU once; only noise added per sigma

Usage:
    python scripts/dnn_feature_mmse.py --dataset cifar10
    python scripts/dnn_feature_mmse.py --dataset mnist
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.dnn_estimator import build_conditional_features


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    choices=['mnist', 'cifar10'], default='cifar10')
    p.add_argument('--n_samples',  type=int,   default=10000)
    p.add_argument('--n_noise',    type=int,   default=5,
                   help='noise draws per image (increases effective N)')
    p.add_argument('--sigma_min',  type=float, default=0.02)
    p.add_argument('--sigma_max',  type=float, default=2.0)
    p.add_argument('--n_sigma',    type=int,   default=20)
    p.add_argument('--lam',        type=float, default=1e-4,
                   help='ridge regularization')
    p.add_argument('--k_u',        type=int,   default=64,
                   help='U projection dim for mode B/C')
    p.add_argument('--modes',      nargs='+',  default=['A', 'B', 'C'])
    p.add_argument('--batch_size', type=int,   default=1024)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--save_dir',   default='figures')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading — two views: original small (for x0) and encoder-ready (for phi)
# ---------------------------------------------------------------------------

def load_dataset(name, n_samples, device):
    """
    Returns:
        x0_small_gpu  : (N, d_small) float32 GPU — original pixels, flattened
        x0_enc_gpu    : (N, C, 224, 224) float16 GPU — ImageNet-normalized for encoder
        U_gpu         : (N, n_classes) float32 GPU — one-hot labels
        n_classes, d_small
    """
    STORE = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
    DATASETS = os.path.join(STORE, 'Datasets')

    # Transform for encoder (ResNet18: 224x224 ImageNet-normalized)
    enc_tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Transform for x0 ground truth: keep original size, just ToTensor [0,1]
    raw_tf = T.ToTensor()

    if name == 'cifar10':
        root = DATASETS if os.path.exists(os.path.join(DATASETS, 'cifar-10-batches-py')) else '/tmp/cifar10'
        ds_enc = torchvision.datasets.CIFAR10(root=root, train=True, download=(root=='/tmp/cifar10'), transform=enc_tf)
        ds_raw = torchvision.datasets.CIFAR10(root=root, train=True, download=False,                 transform=raw_tf)
        n_classes = 10
    elif name == 'mnist':
        root = DATASETS if os.path.exists(os.path.join(DATASETS, 'MNIST')) else '/tmp/mnist'
        enc_tf_m = T.Compose([T.Resize(224), T.Grayscale(3), T.ToTensor(),
                               T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        ds_enc = torchvision.datasets.MNIST(root=root, train=True, download=(root=='/tmp/mnist'), transform=enc_tf_m)
        ds_raw = torchvision.datasets.MNIST(root=root, train=True, download=False,                transform=raw_tf)
        n_classes = 10

    n_samples = min(n_samples, len(ds_enc))
    idx = np.random.default_rng(0).choice(len(ds_enc), n_samples, replace=False).tolist()

    def collect(ds, indices, bs=512):
        subset = torch.utils.data.Subset(ds, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=bs, num_workers=8,
                                             pin_memory=True, persistent_workers=True)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    print("  Loading encoder inputs (224x224) ...")
    x_enc, labels = collect(ds_enc, idx)
    print("  Loading raw images ...")
    x_raw, _      = collect(ds_raw, idx)

    d_small = x_raw[0].numel()
    x0_small_gpu = x_raw.reshape(n_samples, d_small).to(device=device, dtype=torch.float32)
    x0_enc_gpu   = x_enc.to(device=device, dtype=torch.float16)

    U = torch.zeros(n_samples, n_classes)
    U[torch.arange(n_samples), labels] = 1.0
    U_gpu = U.to(device)

    print(f"  x0 shape: {x0_small_gpu.shape}, enc shape: {x0_enc_gpu.shape}")
    print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    return x0_small_gpu, x0_enc_gpu, U_gpu, n_classes, d_small


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(device):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return encoder.to(device).eval()


# ---------------------------------------------------------------------------
# GPU covariance-based MMSE
# ---------------------------------------------------------------------------

@torch.no_grad()
def mmse_gpu(Phi_t: torch.Tensor, X0_t: torch.Tensor, lam: float) -> dict:
    """
    Compute optimal linear-readout denoiser loss entirely on GPU.

    L = Tr(Sigma_p0) - Tr(Cov(x0,phi) (Sigma_phi + lam I)^{-1} Cov(phi,x0))

    Uses primal form when k < N, dual form otherwise.

    Parameters
    ----------
    Phi_t : (N, k) float32 GPU tensor
    X0_t  : (N, d) float32 GPU tensor
    lam   : ridge

    Returns dict: loss, r2, Sigma_p0_trace (all python floats)
    """
    N, k = Phi_t.shape
    d    = X0_t.shape[1]

    Phi_c = Phi_t - Phi_t.mean(0)
    X0_c  = X0_t  - X0_t.mean(0)

    Sigma_p0_trace = float((X0_c ** 2).sum() / (N - 1))

    if k <= N:
        Sigma_phi  = (Phi_c.T @ Phi_c) / (N - 1)              # (k, k)
        Cov_x0_phi = (X0_c.T  @ Phi_c) / (N - 1)              # (d, k)
        reg = Sigma_phi + lam * torch.eye(k, device=Phi_t.device, dtype=Phi_t.dtype)
        # solve: Sigma_phi A = Cov_x0_phi^T  =>  A = reg^{-1} Cov^T
        A   = torch.linalg.solve(reg, Cov_x0_phi.T)           # (k, d)
        explained = float(torch.trace(Cov_x0_phi @ A))
    else:
        K   = (Phi_c @ Phi_c.T) / (N - 1) + lam * torch.eye(N, device=Phi_t.device, dtype=Phi_t.dtype)
        M   = torch.linalg.solve(K, X0_c)                     # (N, d)
        explained = float((X0_c * M).sum()) / (N - 1)

    loss = Sigma_p0_trace - explained
    r2   = explained / Sigma_p0_trace if Sigma_p0_trace > 0 else 0.0
    return dict(loss=loss, r2=r2, Sigma_p0_trace=Sigma_p0_trace)


@torch.no_grad()
def wiener_gpu(X0_t: torch.Tensor, sigma: float, lam: float = 1e-6) -> dict:
    """
    Analytic Wiener filter loss on GPU via eigendecomposition of Sigma_p0.
    L_wiener = sum_i sigma^2 * lambda_i / (lambda_i + sigma^2)
    """
    N, d  = X0_t.shape
    X0_c  = X0_t - X0_t.mean(0)
    # Use SVD to get eigenvalues of Sigma_p0 efficiently (d can be 3072)
    # eigvals of Sigma_p0 = singular values^2 / (N-1)
    if d <= N:
        Sigma = (X0_c.T @ X0_c) / (N - 1)
        eigvals = torch.linalg.eigvalsh(Sigma)  # ascending
    else:
        # Dual: eigenvalues of X0_c X0_c^T / (N-1)
        K_small = (X0_c @ X0_c.T) / (N - 1)
        eigvals = torch.linalg.eigvalsh(K_small)

    eigvals = eigvals.clamp(min=0)
    explained = float((eigvals ** 2 / (eigvals + sigma ** 2 + lam)).sum())
    total     = float(eigvals.sum())
    loss      = total - explained
    r2        = explained / total if total > 0 else 0.0
    return dict(loss=loss, r2=r2)


@torch.no_grad()
def extract_and_repeat(encoder, x0_enc_gpu, sigma, n_noise, batch_size):
    """
    Add noise to encoder input (GPU), run forward pass, return (N*n_noise, k) features.
    x0_enc_gpu: (N, C, 224, 224) fp16 on GPU
    """
    N = len(x0_enc_gpu)
    phi_list = []
    chunk = max(1, batch_size // n_noise)
    for start in range(0, N, chunk):
        end    = min(start + chunk, N)
        x_b    = x0_enc_gpu[start:end]
        x_rep  = x_b.repeat_interleave(n_noise, dim=0)
        Z      = torch.randn_like(x_rep) * sigma
        y      = x_rep + Z
        with torch.cuda.amp.autocast():
            phi = encoder(y)      # (B*n_noise, k) fp16
        phi_list.append(phi.float())
    return torch.cat(phi_list, dim=0)   # (N*n_noise, k) fp32 GPU


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    torch.manual_seed(args.seed)
    device = args.device

    print(f"Loading {args.dataset} ...")
    x0_small, x0_enc, U_gpu, n_classes, d_small = load_dataset(
        args.dataset, args.n_samples, device
    )
    N = len(x0_small)

    print("Building ResNet18 encoder ...")
    encoder = build_encoder(device)

    sigma_grid = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.n_sigma
    )

    results = {'sigma': sigma_grid, 'linear_uncond': [], 'linear_cond': [], 'resnet_uncond': []}
    for mode in args.modes:
        results[f'resnet_cond_{mode}'] = []

    print(f"Sweeping {args.n_sigma} sigma values ...")
    for sigma in tqdm(sigma_grid):

        # 1. Wiener filter (pixel-space, GPU)
        lin = wiener_gpu(x0_small, sigma)
        results['linear_uncond'].append(lin['loss'])

        # Conditional Wiener: phi = [x0 + noise ; U]
        Z_lin = torch.randn_like(x0_small) * sigma
        Y_lin = x0_small + Z_lin
        Phi_lin_cond = torch.cat([Y_lin, U_gpu], dim=1)
        lin_c = mmse_gpu(Phi_lin_cond, x0_small, lam=args.lam)
        results['linear_cond'].append(lin_c['loss'])

        # 2. ResNet18 features (GPU)
        Phi_gpu = extract_and_repeat(encoder, x0_enc, sigma, args.n_noise, args.batch_size)
        # (N*n_noise, k) features; repeat x0 and U to match
        X0_rep  = x0_small.repeat_interleave(args.n_noise, dim=0)   # (N*n_noise, d_small)
        U_rep   = U_gpu.repeat_interleave(args.n_noise, dim=0)       # (N*n_noise, n_classes)

        res_u = mmse_gpu(Phi_gpu, X0_rep, lam=args.lam)
        results['resnet_uncond'].append(res_u['loss'])

        # 3. Conditional ResNet18 (modes A/B/C)
        # Move Phi to CPU for build_conditional_features, then back to GPU
        Phi_np = Phi_gpu.cpu().numpy()
        U_np   = U_rep.cpu().numpy()
        for mode in args.modes:
            Phi_cond_np = build_conditional_features(Phi_np, U_np, mode=mode, k_u=args.k_u)
            Phi_cond_gpu = torch.from_numpy(Phi_cond_np).to(device=device, dtype=torch.float32)
            res_c = mmse_gpu(Phi_cond_gpu, X0_rep, lam=args.lam)
            results[f'resnet_cond_{mode}'].append(res_c['loss'])

    for k in results:
        if k != 'sigma':
            results[k] = np.array(results[k])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, args):
    os.makedirs(args.save_dir, exist_ok=True)
    sigma = results['sigma']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{args.dataset.upper()} | ResNet18 features | N={args.n_samples} x {args.n_noise} noise draws', fontsize=12)

    ax = axes[0]
    ax.plot(sigma, results['linear_uncond'], 'k--',  lw=2, label='Linear (Wiener)')
    ax.plot(sigma, results['linear_cond'],   'k:',   lw=2, label='Linear + U')
    ax.plot(sigma, results['resnet_uncond'], 'C0-o', lw=2, ms=5, label='ResNet18 uncond')
    colors = ['C1', 'C2', 'C3']
    for i, mode in enumerate(args.modes):
        ax.plot(sigma, results[f'resnet_cond_{mode}'], f'{colors[i]}-s',
                lw=2, ms=5, label=f'ResNet18+U (mode {mode})')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('MMSE loss')
    ax.set_title('Denoiser loss vs sigma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sigma, results['linear_uncond'] - results['resnet_uncond'],
            'C0-o', lw=2, ms=5, label='ResNet gain over linear')
    for i, mode in enumerate(args.modes):
        ax.plot(sigma, results['linear_cond'] - results[f'resnet_cond_{mode}'],
                f'{colors[i]}-s', lw=2, ms=5, label=f'ResNet+U ({mode}) gain over linear+U')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('Loss reduction')
    ax.set_title('DNN gain over linear baseline')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.8)

    ax = axes[2]
    ax.plot(sigma, results['linear_uncond'] - results['linear_cond'],
            'k--', lw=2, label='Linear: cond gain')
    ax.plot(sigma, results['resnet_uncond'] - results[f'resnet_cond_{args.modes[0]}'],
            'C0-o', lw=2, ms=5, label=f'ResNet: cond gain (mode {args.modes[0]})')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('L_sigma - L_{sigma,U}')
    ax.set_title('Conditioning gain from class label U')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    path = os.path.join(args.save_dir, f'dnn_feature_mmse_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    return path


def print_summary(results, args):
    sigma = results['sigma']
    mid = len(sigma) // 2
    s = sigma[mid]
    print(f"\n=== Summary at sigma={s:.3f} ===")
    print(f"  Linear uncond:  {results['linear_uncond'][mid]:.4f}")
    print(f"  Linear cond:    {results['linear_cond'][mid]:.4f}")
    print(f"  ResNet uncond:  {results['resnet_uncond'][mid]:.4f}")
    for mode in args.modes:
        print(f"  ResNet+U ({mode}): {results[f'resnet_cond_{mode}'][mid]:.4f}")


if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {args.device}")
    results = run_experiment(args)
    print_summary(results, args)
    plot_results(results, args)
