"""
Patch-GMM empirical Bayes denoisers.

This module models clean image patches with a Gaussian mixture and applies the
closed-form posterior mean for noisy patches:

    x_patch ~ sum_k pi_k N(mu_k, Sigma_k)
    y_patch = x_patch + sigma z

For each noisy patch, E[x_patch | y_patch] is a responsibility-weighted sum of
component posterior means. The image denoiser returns the center-pixel posterior
mean for every overlapping patch location.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class PatchGMM:
    patch_size: int
    n_channels: int
    weights: torch.Tensor
    means: torch.Tensor
    covs: torch.Tensor
    reg_covar: float
    center_slice: slice


def circular_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Return circular image patches as (B*H*W, C*patch_size^2)."""

    if x.ndim != 4:
        raise ValueError(f"x must be (B,C,H,W), got {tuple(x.shape)}")
    if patch_size % 2 != 1:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
    pad = patch_size // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="circular")
    cols = F.unfold(xp, kernel_size=patch_size)
    return cols.transpose(1, 2).reshape(-1, cols.shape[1])


def patch_center_slice(n_channels: int, patch_size: int) -> slice:
    """Slice selecting the C center-pixel coordinates from unfold patch vectors."""

    center = (patch_size * patch_size) // 2
    start = center
    # F.unfold orders as channel-major blocks: [c0 all positions, c1 all positions, ...].
    indices = [c * patch_size * patch_size + center for c in range(n_channels)]
    if indices != list(range(indices[0], indices[-1] + 1)):
        # Caller should use take_center if channels are not contiguous in vector order.
        return slice(-1, -1)
    return slice(start, start + n_channels)


def take_patch_center(patches: torch.Tensor, n_channels: int, patch_size: int) -> torch.Tensor:
    center = (patch_size * patch_size) // 2
    idx = torch.tensor(
        [c * patch_size * patch_size + center for c in range(n_channels)],
        device=patches.device,
        dtype=torch.long,
    )
    return patches.index_select(-1, idx)


@torch.no_grad()
def sample_clean_patches(
    x0: torch.Tensor,
    patch_size: int,
    n_patches: int,
    *,
    batch_size: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Sample clean circular patches uniformly over images and spatial positions."""

    device = x0.device
    n, c, h, w = x0.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    img_idx = torch.randint(n, (n_patches,), device=device, generator=gen)
    pos_idx = torch.randint(h * w, (n_patches,), device=device, generator=gen)
    out = []
    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        imgs = x0.index_select(0, img_idx[start:end])
        patches = circular_patches(imgs, patch_size).reshape(end - start, h * w, c * patch_size * patch_size)
        out.append(patches[torch.arange(end - start, device=device), pos_idx[start:end]])
    return torch.cat(out, dim=0)


@torch.no_grad()
def fit_patch_gmm(
    patches: torch.Tensor,
    n_components: int,
    *,
    n_iter: int = 20,
    reg_covar: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a full-covariance patch GMM with k-means init and EM."""

    device = patches.device
    dtype = patches.dtype
    n, d = patches.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    init_idx = torch.randperm(n, device=device, generator=gen)[:n_components]
    means = patches.index_select(0, init_idx).clone()

    for _ in range(8):
        dist = torch.cdist(patches, means) ** 2
        assign = dist.argmin(dim=1)
        for k in range(n_components):
            mask = assign == k
            if mask.any():
                means[k] = patches[mask].mean(0)

    weights = torch.full((n_components,), 1.0 / n_components, device=device, dtype=dtype)
    covs = torch.eye(d, device=device, dtype=dtype).repeat(n_components, 1, 1)
    eye = torch.eye(d, device=device, dtype=dtype)

    last_ll = None
    for it in range(n_iter):
        log_prob = _log_gaussian_full(patches, means, covs + reg_covar * eye[None])
        log_joint = log_prob + torch.log(weights.clamp_min(1e-12))[None, :]
        log_norm = torch.logsumexp(log_joint, dim=1)
        resp = torch.exp(log_joint - log_norm[:, None])
        nk = resp.sum(0).clamp_min(1e-8)
        weights = nk / n
        means = (resp.T @ patches) / nk[:, None]
        for k in range(n_components):
            xc = patches - means[k]
            covs[k] = (xc.T * resp[:, k][None, :]) @ xc / nk[k] + reg_covar * eye
        ll = float(log_norm.mean())
        if verbose:
            print(f"[patch-gmm] iter={it:02d} avg_loglik={ll:.4f}", flush=True)
        if last_ll is not None and abs(ll - last_ll) < 1e-5:
            break
        last_ll = ll
    return weights, means, covs


def _log_gaussian_full(x: torch.Tensor, means: torch.Tensor, covs: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    k = means.shape[0]
    out = []
    const = d * torch.log(torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.dtype))
    for i in range(k):
        chol = torch.linalg.cholesky(covs[i])
        xc = x - means[i]
        sol = torch.cholesky_solve(xc.T, chol).T
        maha = (xc * sol).sum(1)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        out.append(-0.5 * (const + logdet + maha))
    return torch.stack(out, dim=1)


def make_patch_gmm(
    weights: torch.Tensor,
    means: torch.Tensor,
    covs: torch.Tensor,
    *,
    n_channels: int,
    patch_size: int,
    reg_covar: float,
) -> PatchGMM:
    return PatchGMM(
        patch_size=patch_size,
        n_channels=n_channels,
        weights=weights,
        means=means,
        covs=covs,
        reg_covar=float(reg_covar),
        center_slice=patch_center_slice(n_channels, patch_size),
    )


@torch.no_grad()
def patch_gmm_denoise(
    y: torch.Tensor,
    gmm: PatchGMM,
    sigma: float,
    *,
    batch_patches: int = 65536,
    show_progress: bool = False,
) -> torch.Tensor:
    """Denoise images by center-pixel posterior means under a patch GMM."""

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    b, c, h, w = y.shape
    patches = circular_patches(y, gmm.patch_size)
    total = patches.shape[0]
    center_preds = []
    starts: Iterable[int] = range(0, total, batch_patches)
    if show_progress and tqdm is not None:
        starts = tqdm(starts, desc=f"patch-gmm denoise sigma={sigma:.4g}")
    for start in starts:
        end = min(start + batch_patches, total)
        center_preds.append(_patch_gmm_posterior_center(patches[start:end], gmm, sigma))
    centers = torch.cat(center_preds, dim=0)
    return centers.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def _patch_gmm_posterior_center(y_patch: torch.Tensor, gmm: PatchGMM, sigma: float) -> torch.Tensor:
    k, d = gmm.means.shape
    eye = torch.eye(d, device=y_patch.device, dtype=y_patch.dtype)
    sig2 = float(sigma) ** 2
    logps = []
    post_centers = []
    for i in range(k):
        noisy_cov = gmm.covs[i] + sig2 * eye
        chol = torch.linalg.cholesky(noisy_cov)
        yc = y_patch - gmm.means[i]
        sol = torch.cholesky_solve(yc.T, chol).T
        maha = (yc * sol).sum(1)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        logp = torch.log(gmm.weights[i].clamp_min(1e-12)) - 0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, device=y_patch.device, dtype=y_patch.dtype)) + logdet + maha)
        gain_yc = (gmm.covs[i] @ sol.T).T
        post = gmm.means[i] + gain_yc
        logps.append(logp)
        post_centers.append(take_patch_center(post, gmm.n_channels, gmm.patch_size))
    logps_t = torch.stack(logps, dim=1)
    resp = torch.softmax(logps_t, dim=1)
    centers = torch.stack(post_centers, dim=1)
    return (resp[:, :, None] * centers).sum(1)


@torch.no_grad()
def patch_gmm_mse(
    x0: torch.Tensor,
    gmm: PatchGMM,
    sigma: float,
    *,
    n_noise: int = 1,
    image_batch_size: int = 64,
    batch_patches: int = 65536,
    show_progress: bool = False,
) -> dict[str, float]:
    """Monte-Carlo image-level MSE for the patch-GMM denoiser."""

    start_time = time.perf_counter()
    total = 0.0
    n_total = 0
    for _ in range(n_noise):
        for start in range(0, len(x0), image_batch_size):
            end = min(start + image_batch_size, len(x0))
            xb = x0[start:end]
            yb = xb + torch.randn_like(xb) * float(sigma)
            pred = patch_gmm_denoise(yb, gmm, sigma, batch_patches=batch_patches, show_progress=show_progress)
            total += float(((pred - xb) ** 2).sum())
            n_total += end - start
    if x0.device.type == "cuda":
        torch.cuda.synchronize(x0.device)
    return {"loss": total / max(n_total, 1), "elapsed_seconds": time.perf_counter() - start_time}
