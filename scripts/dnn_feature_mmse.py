"""
DNN Feature MMSE Experiment — MNIST / CIFAR-10 with ResNet18.

Computes denoiser loss as a function of sigma for:
  1. Linear baseline (Wiener filter, pixel-space)
  2. Conditional linear baseline (Wiener filter + class label U)
  3. ResNet18 features, unconditional
  4. ResNet18 features + U, mode A (concat)
  5. ResNet18 features + U, mode B (random nonlinear U features)
  6. ResNet18 features + U, mode C (+ interaction)

Usage:
    python scripts/dnn_feature_mmse.py --dataset cifar10
    python scripts/dnn_feature_mmse.py --dataset mnist --n_samples 10000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.dnn_estimator import (
    extract_features,
    build_conditional_features,
    mmse_from_features,
    wiener_filter_loss,
    wiener_filter_cond_loss,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   choices=['mnist', 'cifar10'], default='cifar10')
    p.add_argument('--n_samples', type=int, default=10000,
                   help='number of training samples to use')
    p.add_argument('--n_noise',   type=int, default=5,
                   help='noise draws per image for feature extraction')
    p.add_argument('--sigma_min', type=float, default=0.02)
    p.add_argument('--sigma_max', type=float, default=2.0)
    p.add_argument('--n_sigma',   type=int,   default=20)
    p.add_argument('--lam',       type=float, default=1e-4,
                   help='ridge regularization for covariance inversion')
    p.add_argument('--k_u',       type=int,   default=64,
                   help='U feature dim for mode B/C')
    p.add_argument('--modes',     nargs='+',  default=['A', 'B', 'C'],
                   help='conditioning modes to evaluate')
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--save_dir',  default='figures')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, n_samples: int, device: str):
    """
    Load dataset, return (x0_tensor, U_onehot, class_names).
    x0_tensor: (N, C, H, W) float32, normalized to ImageNet stats if ResNet
    U_onehot : (N, n_classes) float32
    """
    if name == 'cifar10':
        transform = T.Compose([
            T.Resize(224),   # ResNet18 expects 224x224
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]),
        ])
        ds = torchvision.datasets.CIFAR10(
            root='/tmp/cifar10', train=True, download=True, transform=transform
        )
        n_classes = 10
        class_names = ds.classes

    elif name == 'mnist':
        transform = T.Compose([
            T.Resize(224),
            T.Grayscale(3),   # convert to 3-channel for ResNet
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]),
        ])
        ds = torchvision.datasets.MNIST(
            root='/tmp/mnist', train=True, download=True, transform=transform
        )
        n_classes = 10
        class_names = [str(i) for i in range(10)]

    n_samples = min(n_samples, len(ds))
    idx = np.random.default_rng(0).choice(len(ds), n_samples, replace=False)
    subset = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=256, num_workers=4, pin_memory=True)

    x0_list, label_list = [], []
    for x, y in loader:
        x0_list.append(x)
        label_list.append(y)
    x0 = torch.cat(x0_list, dim=0)   # (N, C, H, W)
    labels = torch.cat(label_list).numpy()

    U = np.zeros((len(labels), n_classes), dtype=np.float32)
    U[np.arange(len(labels)), labels] = 1.0   # one-hot

    return x0, U, class_names


# ---------------------------------------------------------------------------
# Build ResNet18 encoder (strip final FC)
# ---------------------------------------------------------------------------

def build_encoder(device: str) -> nn.Module:
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the classification head — use avgpool output (512-dim)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    encoder = encoder.to(device).eval()
    return encoder


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.dataset} ...")
    x0, U, class_names = load_dataset(args.dataset, args.n_samples, args.device)
    X0_flat = x0.reshape(len(x0), -1).numpy()   # (N, d) for Wiener baseline

    print(f"Building ResNet18 encoder ...")
    encoder = build_encoder(args.device)

    sigma_grid = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.n_sigma
    )

    results = {
        'sigma': sigma_grid,
        'linear_uncond': [],
        'linear_cond':   [],
        'resnet_uncond': [],
    }
    for mode in args.modes:
        results[f'resnet_cond_{mode}'] = []

    print(f"Sweeping {args.n_sigma} sigma values ...")
    for sigma in tqdm(sigma_grid):
        # 1. Linear (Wiener) baselines
        lin = wiener_filter_loss(X0_flat, sigma, lam=args.lam)
        results['linear_uncond'].append(lin['loss'])

        lin_c = wiener_filter_cond_loss(X0_flat, U, sigma, lam=args.lam)
        results['linear_cond'].append(lin_c['loss'])

        # 2. ResNet18 features — unconditional
        Phi, X0_rep = extract_features(encoder, x0, sigma,
                                       n_noise=args.n_noise, device=args.device, rng=rng)
        U_rep = np.repeat(U, args.n_noise, axis=0)

        res_u = mmse_from_features(Phi, X0_rep, lam=args.lam)
        results['resnet_uncond'].append(res_u['loss'])

        # 3. ResNet18 features — conditional (modes A/B/C)
        for mode in args.modes:
            Phi_cond = build_conditional_features(Phi, U_rep, mode=mode, k_u=args.k_u)
            res_c = mmse_from_features(Phi_cond, X0_rep, lam=args.lam)
            results[f'resnet_cond_{mode}'].append(res_c['loss'])

    # Convert to arrays
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
    fig.suptitle(f'{args.dataset.upper()} | ResNet18 features | N={args.n_samples}', fontsize=13)

    # Panel 1: all loss curves
    ax = axes[0]
    ax.plot(sigma, results['linear_uncond'], 'k--',  lw=2, label='Linear (Wiener)')
    ax.plot(sigma, results['linear_cond'],   'k:',   lw=2, label='Linear + U (Wiener)')
    ax.plot(sigma, results['resnet_uncond'], 'C0-o', lw=2, label='ResNet18 uncond')
    for mode in args.modes:
        ax.plot(sigma, results[f'resnet_cond_{mode}'], f'C{ord(mode)-64}-s',
                lw=2, label=f'ResNet18 + U (mode {mode})')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('MMSE loss')
    ax.set_title('Denoiser loss vs sigma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: gain of ResNet over linear
    ax = axes[1]
    ax.plot(sigma, results['linear_uncond'] - results['resnet_uncond'],
            'C0-o', lw=2, label='ResNet gain over linear')
    for mode in args.modes:
        ax.plot(sigma,
                results['linear_cond'] - results[f'resnet_cond_{mode}'],
                f'C{ord(mode)-64}-s', lw=2, label=f'ResNet+U (mode {mode}) gain over linear+U')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('Loss reduction')
    ax.set_title('DNN gain over linear baseline')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.8)

    # Panel 3: conditioning gain (L_sigma - L_{sigma,U})
    ax = axes[2]
    ax.plot(sigma, results['linear_uncond'] - results['linear_cond'],
            'k--', lw=2, label='Linear: cond gain')
    ax.plot(sigma, results['resnet_uncond'] - results[f'resnet_cond_{args.modes[0]}'],
            'C0-o', lw=2, label=f'ResNet: cond gain (mode {args.modes[0]})')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('L_sigma - L_{sigma,U}')
    ax.set_title('Conditioning gain from U')
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
    fig_path = plot_results(results, args)
    print(f"\nFigure: {fig_path}")
