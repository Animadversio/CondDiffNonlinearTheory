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
import torch.nn.functional as F
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
    p.add_argument('--sigma_max',  type=float, default=100.0)
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


# ---------------------------------------------------------------------------
# Analytic linear+U MMSE (Wiener filter with [y; U] features)
# ---------------------------------------------------------------------------

@torch.no_grad()
def wiener_linear_u_precompute(x0_gpu: torch.Tensor, U_gpu: torch.Tensor) -> dict:
    """
    Precompute eigendecomposition of Sigma_p0 and cross-covariance C_xU.
    Call once; pass the result to wiener_linear_u_loss() per sigma.
    """
    N = len(x0_gpu)
    X0_c = x0_gpu - x0_gpu.mean(0)
    U_c  = U_gpu  - U_gpu.mean(0)
    Sigma_p0 = X0_c.T @ X0_c / (N - 1)     # (d, d)
    C_xU     = X0_c.T @ U_c  / (N - 1)     # (d, n_c)
    Sigma_U  = U_c.T  @ U_c  / (N - 1)     # (n_c, n_c)
    eigvals, eigvecs = torch.linalg.eigh(Sigma_p0)   # ascending, (d,), (d,d)
    eigvals = eigvals.clamp(min=0)
    VtC = eigvecs.T @ C_xU                  # (d, n_c)
    return dict(eigvals=eigvals, eigvecs=eigvecs, VtC=VtC, Sigma_U=Sigma_U,
                trace_p0=float(eigvals.sum()))


@torch.no_grad()
def wiener_linear_u_loss(precomp: dict, sigma: float, lam: float = 1e-6) -> dict:
    """
    Analytic LMMSE loss for estimator using [y; U] features, always non-negative.

    Derivation: the optimal linear estimator of x0 from [y=x0+σZ; U] has loss

        L = L_wiener_uncond − Tr(Q · Schur⁻¹)

    where:
        a_i    = λᵢ / (λᵢ + σ²)            (signal-to-total ratio per eigenmode)
        Q      = VᵀC · diag((1−a)²) · VᵀC^T   (n_c × n_c, always PSD)
        Schur  = Σ_U − VᵀC · diag(1/(λ+σ²)) · VᵀC^T   (Schur complement, n_c × n_c)

    This is always non-negative and converges to within-class variance at σ→∞.
    At σ→∞ both this and wiener_class_cond converge to the same within-class variance.
    At finite σ, wiener_class_cond ≤ this because per-class slopes are more expressive.
    """
    eigvals = precomp['eigvals']
    VtC     = precomp['VtC']
    Sigma_U = precomp['Sigma_U']
    n_c     = Sigma_U.shape[0]
    device  = eigvals.device
    dtype   = eigvals.dtype

    inv_y = 1.0 / (eigvals + sigma ** 2 + lam)      # (d,)
    a     = eigvals * inv_y                           # λ/(λ+σ²)

    # Unconditional Wiener loss
    L_uncond = precomp['trace_p0'] - float((eigvals ** 2 * inv_y).sum())

    # Q = VtC^T diag((1-a)^2) VtC  [n_c x n_c]
    Q = VtC.T @ (((1 - a) ** 2)[:, None] * VtC)

    # Schur complement of Sigma_y in joint [[Sigma_y, C_xU],[C_xU^T, Sigma_U]]
    CyC   = VtC.T @ (inv_y[:, None] * VtC)          # C_xU^T Sigma_y^{-1} C_xU
    Schur = Sigma_U - CyC + lam * torch.eye(n_c, device=device, dtype=dtype)
    Schur_inv = torch.linalg.inv(Schur)

    u_correction = float(torch.trace(Q @ Schur_inv))
    return dict(loss=L_uncond - u_correction)


# ---------------------------------------------------------------------------
# Class-conditional Wiener filter — analytic per-class eigendecomposition
# ---------------------------------------------------------------------------

@torch.no_grad()
def wiener_class_cond_gpu(
    x0_gpu: torch.Tensor,
    labels: torch.Tensor,
    sigma: float,
    n_classes: int,
    lam: float = 1e-6,
) -> dict:
    """
    Class-conditional analytic Wiener filter MMSE — always non-negative.

    For each class c, computes the analytic LMMSE:
        L_c = sum_i  sigma^2 * lambda_i^c / (lambda_i^c + sigma^2)
    where lambda_i^c are eigenvalues of the within-class covariance Sigma_c.

    This is the same formula as wiener_gpu() applied per class.
    It is provably non-negative and converges to Tr(Sigma_c) at high sigma.

    Averaged over classes weighted by class frequency.
    """
    N = len(x0_gpu)
    total_loss = 0.0

    for c in range(n_classes):
        mask = labels == c
        x0_c = x0_gpu[mask]   # (N_c, d)
        N_c  = len(x0_c)
        if N_c < 2:
            continue
        result = wiener_gpu(x0_c, sigma, lam=lam)
        total_loss += result['loss'] * N_c / N

    return dict(loss=total_loss)


# ---------------------------------------------------------------------------
# Load full train/test x0 at original resolution (for Bayes pool/eval)
# ---------------------------------------------------------------------------

def load_x0_split(name, split, device):
    """
    Load the full train or test split of a dataset at original resolution (32x32).
    Returns:
        x0_gpu : (N, d) float32 GPU
        labels : (N,) long GPU
    """
    STORE = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
    DATASETS = os.path.join(STORE, 'Datasets')
    raw_tf = T.ToTensor()
    train_flag = (split == 'train')

    if name == 'cifar10':
        root = DATASETS if os.path.exists(os.path.join(DATASETS, 'cifar-10-batches-py')) else '/tmp/cifar10'
        ds = torchvision.datasets.CIFAR10(root=root, train=train_flag,
                                           download=(root == '/tmp/cifar10'), transform=raw_tf)
    elif name == 'mnist':
        root = DATASETS if os.path.exists(os.path.join(DATASETS, 'MNIST')) else '/tmp/mnist'
        ds = torchvision.datasets.MNIST(root=root, train=train_flag,
                                         download=(root == '/tmp/mnist'), transform=raw_tf)

    loader = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=8,
                                          pin_memory=True, persistent_workers=True)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
    x_all = torch.cat(xs)
    y_all = torch.cat(ys)
    d = x_all[0].numel()
    x0_gpu = x_all.reshape(len(x_all), d).to(device=device, dtype=torch.float32)
    labels_gpu = y_all.to(device=device)
    return x0_gpu, labels_gpu


# ---------------------------------------------------------------------------
# Exact empirical Bayes-optimal estimator: softmax-weighted conditional mean
# ---------------------------------------------------------------------------

@torch.no_grad()
def bayes_optimal_loss(
    x0_pool: torch.Tensor,
    pool_labels: torch.Tensor,
    x0_eval: torch.Tensor,
    eval_labels: torch.Tensor,
    sigma: float,
    n_classes: int,
    n_eval_per_class: int = 500,
    conditional: bool = True,
) -> dict:
    """
    Oracle empirical Bayes MMSE with self-inclusive pool.

    conditional=True  (class-cond oracle):
        For each eval sample with class c, pool = same-class training samples.
        Softmax over ~N/n_classes pool members.
        MSE → 0 at low σ (self-predict), → within-class variance at σ → ∞.
        Lower bounds E[x₀|y,c] — the class-conditional MMSE.

    conditional=False (unconditional oracle):
        Pool = ALL training samples, no class restriction.
        Softmax over full N pool.
        MSE → 0 at low σ, → total variance at σ → ∞.
        Lower bounds E[x₀|y] — the unconditional MMSE.
    """
    device = x0_pool.device
    total_mse  = 0.0
    total_eval = 0

    if conditional:
        # Loop over classes — pool restricted to same class
        for c in range(n_classes):
            pool_mask = (pool_labels == c).nonzero(as_tuple=True)[0]
            eval_mask = (eval_labels == c).nonzero(as_tuple=True)[0]
            if len(pool_mask) < 2 or len(eval_mask) < 1:
                continue
            if len(eval_mask) > n_eval_per_class:
                eval_mask = eval_mask[:n_eval_per_class]

            x0_p = x0_pool[pool_mask]
            x0_e = x0_eval[eval_mask]
            y_e  = x0_e + torch.randn_like(x0_e) * sigma

            dists_sq = (
                (y_e ** 2).sum(1, keepdim=True)
                + (x0_p ** 2).sum(1, keepdim=True).T
                - 2.0 * (y_e @ x0_p.T)
            ).clamp(min=0)

            w    = torch.softmax(-dists_sq / (2.0 * sigma ** 2), dim=1)
            pred = w @ x0_p
            total_mse  += float(((pred - x0_e) ** 2).sum())
            total_eval += len(eval_mask)

    else:
        # Unconditional: pool = all samples, no class restriction
        # Process in chunks of eval samples to avoid OOM on (N_eval x N_pool) matrix
        n_eval_total = min(n_eval_per_class * n_classes, len(x0_eval))
        eval_idx = torch.arange(n_eval_total, device=device)
        x0_e = x0_eval[eval_idx]
        y_e  = x0_e + torch.randn_like(x0_e) * sigma

        chunk = 200
        for start in range(0, n_eval_total, chunk):
            end    = min(start + chunk, n_eval_total)
            y_ch   = y_e[start:end]
            x0_ch  = x0_e[start:end]
            dists_sq = (
                (y_ch ** 2).sum(1, keepdim=True)
                + (x0_pool ** 2).sum(1, keepdim=True).T
                - 2.0 * (y_ch @ x0_pool.T)
            ).clamp(min=0)
            w    = torch.softmax(-dists_sq / (2.0 * sigma ** 2), dim=1)
            pred = w @ x0_pool
            total_mse  += float(((pred - x0_ch) ** 2).sum())
        total_eval = n_eval_total

    loss = total_mse / total_eval if total_eval > 0 else float('nan')
    return dict(loss=loss, n_eval=total_eval)


# Keep old name as alias
def bayes_optimal_cond_loss(*args, **kwargs):
    return bayes_optimal_loss(*args, **kwargs)


@torch.no_grad()
def extract_and_repeat(encoder, x0_small, sigma, n_noise, batch_size, dataset_name):
    """
    Add noise at PIXEL resolution (same noise model as oracle Bayes), then encode.
    x0_small: (N, d) float32 pixel-space [0,1]
    Returns: (N*n_noise, k) features fp32 GPU
    """
    N, d = x0_small.shape
    device = x0_small.device
    if dataset_name == 'cifar10':
        C, H, W = 3, 32, 32
    elif dataset_name == 'mnist':
        C, H, W = 1, 28, 28
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    x0_img = x0_small.reshape(N, C, H, W)

    phi_list = []
    chunk = max(1, batch_size // n_noise)
    for start in range(0, N, chunk):
        end   = min(start + chunk, N)
        x_b   = x0_img[start:end]
        x_rep = x_b.repeat_interleave(n_noise, dim=0)      # (B*n_noise, C, H, W)
        Z     = torch.randn_like(x_rep) * sigma
        Y_img = x_rep + Z                                   # noise at pixel scale
        # clamp → 3ch → resize → ImageNet-normalise
        Y_enc = Y_img.clamp(0.0, 1.0)
        if C == 1:
            Y_enc = Y_enc.expand(-1, 3, -1, -1).contiguous()
        Y_enc = F.interpolate(Y_enc.float(), size=224, mode='bilinear', align_corners=False)
        Y_enc = (Y_enc - mean) / std
        with torch.cuda.amp.autocast():
            phi = encoder(Y_enc.half())
        phi_list.append(phi.float())
    return torch.cat(phi_list, dim=0)   # (N*n_noise, k) fp32 GPU


# ---------------------------------------------------------------------------
# Combined [y_pixel ; phi_enc] estimator via Schur complement
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_combined_stats(encoder, x0_small, sigma, n_noise, batch_size, dataset_name):
    """
    Add noise at original (32x32) pixel resolution; accumulate statistics
    needed for the Schur-based combined MMSE without materialising the
    full (N*n_noise, d) noisy-image matrix.

    Returns a dict with:
        Phi_total     : (N, k)   sum of encoder outputs over n_noise draws
        YTphi_total   : (d, k)   sum of Y_m^T @ Phi_m  over draws
        PhiTphi_total : (k, k)   sum of Phi_m^T @ Phi_m over draws
        NM            : int      N * n_noise
    """
    N, d = x0_small.shape
    device = x0_small.device

    if dataset_name == 'cifar10':
        C, H, W = 3, 32, 32
    elif dataset_name == 'mnist':
        C, H, W = 1, 28, 28
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    x0_img = x0_small.reshape(N, C, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    # We don't know k until the first encoder call; initialise after
    Phi_total     = None
    YTphi_total   = None
    PhiTphi_total = None

    for _ in range(n_noise):
        Z     = torch.randn_like(x0_img) * sigma
        Y_img = x0_img + Z                          # (N, C, H, W), may exceed [0,1]
        Y_flat = Y_img.reshape(N, d)                # keep UNclamped for cross-cov

        # Prepare encoder input: clamp → expand to 3ch → resize → ImageNet-norm
        Y_enc = Y_img.clamp(0.0, 1.0)
        if C == 1:
            Y_enc = Y_enc.expand(-1, 3, -1, -1).contiguous()
        Y_enc = F.interpolate(Y_enc, size=224, mode='bilinear', align_corners=False)
        Y_enc = (Y_enc - mean) / std

        phi_chunks = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            with torch.cuda.amp.autocast():
                phi = encoder(Y_enc[start:end].half())
            phi_chunks.append(phi.float())
        Phi_m = torch.cat(phi_chunks, 0)   # (N, k)
        k = Phi_m.shape[1]

        if Phi_total is None:
            Phi_total     = torch.zeros(N, k, device=device)
            YTphi_total   = torch.zeros(d, k, device=device)
            PhiTphi_total = torch.zeros(k, k, device=device)

        Phi_total     += Phi_m
        YTphi_total   += Y_flat.T @ Phi_m          # (d, k)
        PhiTphi_total += Phi_m.T  @ Phi_m          # (k, k)

    return dict(Phi_total=Phi_total, YTphi_total=YTphi_total,
                PhiTphi_total=PhiTphi_total, NM=N * n_noise)


@torch.no_grad()
def mmse_combined_schur(stats, x0_small, eigvals, eigvecs, sigma, lam=1e-4):
    """
    MMSE for combined [y_pixel ; phi_enc(y)] feature via Schur complement.

    Decomposition:
        L_combined = L_Wiener(analytic)  −  Gain(phi_enc | y_pixel)

    where the gain uses partial covariance (Schur complement of Sigma_y):
        C_part    = Cov(x0, phi) − Sigma_x0 Sigma_y^{-1} Cov(y, phi)    (d × k)
        S_phi     = Var(phi)    − Cov(y,phi)^T Sigma_y^{-1} Cov(y,phi)   (k × k)
        Gain      = Tr(C_part (S_phi + lam I)^{-1} C_part^T)

    Sigma_y^{-1} applied analytically via eigendecomp of Sigma_x0.
    Only a k×k system (k=512) needs to be solved — never d×d.

    Properties:
        L_combined ≤ L_Wiener        (Gain ≥ 0 always)
        L_combined ≤ L_ResNet_uncond (combined feature space ⊇ phi alone)
    """
    NM = stats['NM']
    N  = len(x0_small)
    device = x0_small.device

    Phi_total     = stats['Phi_total']       # (N, k)
    YTphi_total   = stats['YTphi_total']     # (d, k)
    PhiTphi_total = stats['PhiTphi_total']   # (k, k)
    k = Phi_total.shape[1]

    mu_x0  = x0_small.mean(0)               # (d,)
    mu_phi = Phi_total.sum(0) / NM          # (k,) global mean across all draws

    # --- Centred sample covariances (unbiased denominator NM-1) ---
    # Cov(x0, phi) = E[x0 phi^T] - mu_x0 mu_phi^T
    CxPhi  = x0_small.T @ Phi_total / NM   # (d, k)
    CxPhi  = CxPhi - mu_x0[:, None] * mu_phi[None, :]

    # Cov(y, phi) = E[y phi^T] - mu_y mu_phi^T,  mu_y = mu_x0 (E[Z]=0)
    CyPhi  = YTphi_total / NM               # (d, k)
    CyPhi  = CyPhi - mu_x0[:, None] * mu_phi[None, :]

    # Var(phi) = E[phi phi^T] - mu_phi mu_phi^T
    SigPhi = PhiTphi_total / NM             # (k, k)
    SigPhi = SigPhi - mu_phi[:, None] * mu_phi[None, :]

    # --- Analytic Sigma_y^{-1} = V diag(1/(lambda+sigma^2)) V^T ---
    inv_y = 1.0 / (eigvals + sigma ** 2 + lam)           # (d,)

    VtCyPhi      = eigvecs.T @ CyPhi                      # (d, k)
    InvSy_CyPhi  = eigvecs @ (inv_y[:, None] * VtCyPhi)  # (d, k)

    # --- Partial covariance: Cov(x0, phi | y) ---
    a = eigvals * inv_y                                    # lambda/(lambda+sigma^2), (d,)
    Sigma_x0_InvSy_CyPhi = eigvecs @ (a[:, None] * VtCyPhi)   # (d, k)
    C_part = CxPhi - Sigma_x0_InvSy_CyPhi                 # (d, k)

    # --- Schur complement: Var(phi | y) ---
    S_phi = SigPhi - CyPhi.T @ InvSy_CyPhi               # (k, k)

    # --- Gain = Tr(C_part (S_phi + lam I)^{-1} C_part^T) ---
    reg  = S_phi + lam * torch.eye(k, device=device)
    A    = torch.linalg.solve(reg, C_part.T)              # (k, d)
    gain = float((C_part * A.T).sum())                    # = Tr(C_part A^T)

    # --- Analytic Wiener loss ---
    trace_p0 = float(eigvals.sum())
    L_wiener  = trace_p0 - float((eigvals ** 2 * inv_y).sum())

    return dict(loss=max(0.0, L_wiener - gain), L_wiener=L_wiener, gain=gain)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    torch.manual_seed(args.seed)
    rng    = np.random.default_rng(args.seed)
    device = args.device

    print(f"Loading {args.dataset} ...")
    x0_small, x0_enc, U_gpu, n_classes, d_small = load_dataset(
        args.dataset, args.n_samples, device
    )
    N      = len(x0_small)
    labels = U_gpu.argmax(dim=1)          # (N,) integer class labels

    print("Building ResNet18 encoder ...")
    encoder = build_encoder(device)

    # Load full train/test splits for Bayes pool/eval (avoids self-prediction)
    print("Loading full train set as Bayes pool ...")
    x0_pool_full, pool_labels_full = load_x0_split(args.dataset, 'train', device)
    print(f"  Pool: {x0_pool_full.shape} ({len(x0_pool_full)} samples)")
    print("Loading full test set as Bayes eval ...")
    x0_eval_full, eval_labels_full = load_x0_split(args.dataset, 'test', device)
    print(f"  Eval: {x0_eval_full.shape} ({len(x0_eval_full)} samples)")

    sigma_grid = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.n_sigma
    )

    # Precompute analytic linear+U quantities (eigendecomp of Sigma_p0, cross-cov C_xU)
    # eigvecs are also used by the combined [y;phi] Schur estimator
    print("Precomputing analytic linear+U Wiener quantities ...")
    linear_u_precomp = wiener_linear_u_precompute(x0_small, U_gpu)
    eigvals_p0 = linear_u_precomp['eigvals']    # (d,)  reused by mmse_combined_schur
    eigvecs_p0 = linear_u_precomp['eigvecs']    # (d,d) reused by mmse_combined_schur

    results = {
        'sigma':              sigma_grid,
        'linear_uncond':      [],
        'linear_cond':        [],   # analytic LMMSE for [y; U] (shared slope A, class bias BU)
        'wiener_class_cond':  [],   # analytic LMMSE per class (per-class slope A^c)
        'bayes_cond':         [],   # oracle Bayes, pool = same class (LB on cond MMSE)
        'bayes_uncond':       [],   # oracle Bayes, pool = all classes (LB on uncond MMSE)
        'resnet_uncond':      [],
        'linear_plus_dnn':    [],   # Schur: L_Wiener − Gain(phi_enc | y_pixel)
    }
    for mode in args.modes:
        results[f'resnet_cond_{mode}'] = []

    print(f"Sweeping {args.n_sigma} sigma values ...")
    for sigma in tqdm(sigma_grid):

        # 1a. Global Wiener filter (pixel-space, GPU) — analytic
        lin = wiener_gpu(x0_small, sigma)
        results['linear_uncond'].append(lin['loss'])

        # 1b. Analytic conditional Wiener: optimal linear estimator from [y; U]
        #     Shared slope A, class-specific bias via B*U. Always non-negative.
        lin_c = wiener_linear_u_loss(linear_u_precomp, sigma, lam=args.lam)
        results['linear_cond'].append(lin_c['loss'])

        # 1c. Class-conditional Wiener via dual-form regression (unbiased, avoids Marchenko-Pastur)
        wcc = wiener_class_cond_gpu(x0_small, labels, sigma, n_classes, lam=args.lam)
        results['wiener_class_cond'].append(wcc['loss'])

        # 1d. Oracle Bayes conditional: pool = same class only  → LB on class-cond MMSE
        bayes_c = bayes_optimal_loss(
            x0_small, labels, x0_small, labels,
            sigma, n_classes, conditional=True,
        )
        results['bayes_cond'].append(bayes_c['loss'])

        # 1e. Oracle Bayes unconditional: pool = all classes  → LB on uncond MMSE
        bayes_u = bayes_optimal_loss(
            x0_small, labels, x0_small, labels,
            sigma, n_classes, conditional=False,
        )
        results['bayes_uncond'].append(bayes_u['loss'])

        # 2. ResNet18 features (GPU) — noise at 32x32 pixel level (consistent with oracle Bayes)
        Phi_gpu = extract_and_repeat(encoder, x0_small, sigma, args.n_noise, args.batch_size, args.dataset)
        X0_rep  = x0_small.repeat_interleave(args.n_noise, dim=0)
        U_rep   = U_gpu.repeat_interleave(args.n_noise, dim=0)

        res_u = mmse_gpu(Phi_gpu, X0_rep, lam=args.lam)
        results['resnet_uncond'].append(res_u['loss'])

        # 3. Conditional ResNet18 (modes A/B/C)
        Phi_np = Phi_gpu.cpu().numpy()
        U_np   = U_rep.cpu().numpy()
        for mode in args.modes:
            Phi_cond_np  = build_conditional_features(Phi_np, U_np, mode=mode, k_u=args.k_u)
            Phi_cond_gpu = torch.from_numpy(Phi_cond_np).to(device=device, dtype=torch.float32)
            res_c = mmse_gpu(Phi_cond_gpu, X0_rep, lam=args.lam)
            results[f'resnet_cond_{mode}'].append(res_c['loss'])

        # 4. Combined [y_pixel; phi_enc] — consistent 32x32 noise, Schur decomposition
        combo_stats = extract_combined_stats(
            encoder, x0_small, sigma, args.n_noise, args.batch_size, args.dataset
        )
        combo = mmse_combined_schur(combo_stats, x0_small, eigvals_p0, eigvecs_p0,
                                    sigma, lam=args.lam)
        results['linear_plus_dnn'].append(combo['loss'])

    for k in results:
        if k != 'sigma':
            results[k] = np.array(results[k])
    return results


def save_results(results, args):
    """Save results to tables/ as .npz for later replotting."""
    os.makedirs('tables', exist_ok=True)
    tag  = f"{args.dataset}_N{args.n_samples}_noise{args.n_noise}_sigma{args.n_sigma}"
    path = os.path.join('tables', f'dnn_feature_mmse_{tag}.npz')
    np.savez(path, **results)
    print(f"Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, args):
    os.makedirs(args.save_dir, exist_ok=True)
    sigma  = results['sigma']
    colors = ['C1', 'C2', 'C3']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f'{args.dataset.upper()} | ResNet18 features | N={args.n_samples} x {args.n_noise} noise draws',
        fontsize=12
    )

    # --- Panel 1: all loss curves ---
    ax = axes[0]
    ax.plot(sigma, results['bayes_cond'],         'g-',   lw=2.5, label='Oracle Bayes cond (same-class pool LB)')
    ax.plot(sigma, results['bayes_uncond'],       'b-',   lw=2.5, label='Oracle Bayes uncond (all-class pool LB)')
    ax.plot(sigma, results['wiener_class_cond'], 'g--',  lw=2,   label='Class-cond Wiener (dual-form)')
    ax.plot(sigma, results['linear_uncond'],     'k--',  lw=2,   label='Linear Wiener (uncond)')
    ax.plot(sigma, results['linear_cond'],       'k:',   lw=2,   label='Linear Wiener + U')
    ax.plot(sigma, results['resnet_uncond'],     'C0-o', lw=2, ms=4, label='ResNet18 uncond')
    ax.plot(sigma, results['linear_plus_dnn'],  'm-^',  lw=2, ms=4, label='Linear+DNN combined [y;phi]')
    for i, mode in enumerate(args.modes):
        ax.plot(sigma, results[f'resnet_cond_{mode}'], f'{colors[i]}-s',
                lw=2, ms=4, label=f'ResNet18+U (mode {mode})')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('MMSE loss')
    ax.set_title('Denoiser loss vs sigma')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: DNN vs linear ---
    ax = axes[1]
    ax.plot(sigma, results['linear_uncond'] - results['resnet_uncond'],
            'C0-o', lw=2, ms=4, label='ResNet gain over Wiener')
    ax.plot(sigma, results['linear_uncond'] - results['linear_plus_dnn'],
            'm-^', lw=2, ms=4, label='Combined [y;phi] gain over Wiener')
    for i, mode in enumerate(args.modes):
        ax.plot(sigma, results['linear_cond'] - results[f'resnet_cond_{mode}'],
                f'{colors[i]}-s', lw=2, ms=4, label=f'ResNet+U ({mode}) gain over Wiener+U')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('Linear loss − DNN loss')
    ax.set_title('DNN gain over linear (positive = DNN better)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=1, ls='--')

    # --- Panel 3: conditioning gain for each method ---
    ax = axes[2]
    # Gains relative to linear_uncond baseline (mixing nonlinearity + conditioning)
    ax.plot(sigma, results['linear_uncond']     - results['bayes_uncond'],
            'b--', lw=2, label='Oracle Bayes uncond vs linear_uncond\n(nonlinearity gain, no class label)')
    ax.plot(sigma, results['linear_uncond']     - results['bayes_cond'],
            'b-',  lw=2, label='Oracle Bayes cond vs linear_uncond\n(nonlinearity + conditioning gain)')
    # Direct oracle conditioning gain: how much does class label help within oracle Bayes
    bayes_cond_arr   = np.array(results['bayes_cond'])
    bayes_uncond_arr = np.array(results['bayes_uncond'])
    ax.plot(sigma, np.abs(bayes_uncond_arr - bayes_cond_arr),
            'b:', lw=2.5, label='|Oracle Bayes uncond − cond|\n(pure conditioning gain within oracle)')
    # Conditioning gains for linear/DNN methods (relative to their own uncond baseline)
    ax.plot(sigma, results['linear_uncond']     - results['wiener_class_cond'],
            'g--', lw=2, label='Class-cond Wiener vs linear_uncond\n(cond gain, analytic per-class)')
    ax.plot(sigma, results['linear_uncond']     - results['linear_cond'],
            'k:',  lw=2, label='Linear+U vs linear_uncond\n(cond gain, shared slope)')
    ax.plot(sigma, results['resnet_uncond']     - results[f'resnet_cond_{args.modes[0]}'],
            'C0-o', lw=2, ms=4, label=f'ResNet+U (mode {args.modes[0]}) vs ResNet uncond\n(cond gain within ResNet)')
    ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('L_method_uncond − L_method_cond')
    ax.set_title('Conditioning gain from class label U\n(positive = class label helps)')
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    path = os.path.join(args.save_dir, f'dnn_feature_mmse_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    return path


def print_summary(results, args):
    sigma = results['sigma']
    mid   = len(sigma) // 2
    s     = sigma[mid]
    print(f"\n=== Summary at sigma={s:.3f} ===")
    print(f"  Oracle Bayes cond (LB):  {results['bayes_cond'][mid]:.4f}  (same-class pool → cond MMSE LB)")
    print(f"  Oracle Bayes uncond (LB):{results['bayes_uncond'][mid]:.4f}  (all-class pool → uncond MMSE LB)")
    print(f"  Class-cond Wiener:      {results['wiener_class_cond'][mid]:.4f}  (dual-form regression)")
    print(f"  Linear uncond (Wiener): {results['linear_uncond'][mid]:.4f}")
    print(f"  Linear cond (Wiener+U): {results['linear_cond'][mid]:.4f}")
    print(f"  ResNet uncond:          {results['resnet_uncond'][mid]:.4f}")
    print(f"  Linear+DNN combined:    {results['linear_plus_dnn'][mid]:.4f}  ([y_pixel; phi_enc], Schur)")
    for mode in args.modes:
        print(f"  ResNet+U ({mode}):        {results[f'resnet_cond_{mode}'][mid]:.4f}")


if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {args.device}")
    results = run_experiment(args)
    print_summary(results, args)
    save_results(results, args)
    plot_results(results, args)
