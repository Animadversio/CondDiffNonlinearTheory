"""
2D convolutional random-feature MMSE estimators.

The feature map is fixed:

    phi(y) = relu(conv2d_circular(y, W_rf) + b_rf)

and the denoiser is the optimal linear convolutional readout from phi to x0.
With circular boundary conditions, the readout decouples by 2D Fourier
frequency, so each frequency only needs a small F x F ridge solve where F is
the number of random filters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ConvRFStats:
    """Sufficient statistics for one sigma/filter setting."""

    sigma: float
    n_samples: int
    n_noise: int
    n_total: int
    n_channels: int
    height: int
    width: int
    n_filters: int
    kernel_size: int
    sum_phi: torch.Tensor
    sum_x: torch.Tensor
    sum_phiphi: torch.Tensor
    sum_xphi: torch.Tensor
    sum_x_abs2: torch.Tensor
    elapsed_seconds: float


@dataclass
class DenseWienerPrecomp:
    """Eigendecomposition for the full dense Wiener denoiser."""

    mean: torch.Tensor
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    trace: float


@dataclass
class PatchReadoutStats:
    """Spatial-domain sufficient stats for a shared local readout."""

    sigma: float
    n_samples: int
    n_noise: int
    n_total_images: int
    n_total_patches: int
    n_channels: int
    height: int
    width: int
    n_filters: int
    feature_kernel_size: int
    readout_patch_size: int
    sum_feat: torch.Tensor
    sum_target: torch.Tensor
    sum_featfeat: torch.Tensor
    sum_targetfeat: torch.Tensor
    sum_target_abs2: torch.Tensor
    elapsed_seconds: float


def make_random_conv_filters(
    n_filters: int,
    in_channels: int,
    kernel_size: int,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    bias_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw random convolutional filters with unit expected row norm."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    scale = float(in_channels * kernel_size * kernel_size) ** -0.5
    weight = torch.randn(
        n_filters,
        in_channels,
        kernel_size,
        kernel_size,
        generator=gen,
        dtype=dtype,
    ) * scale
    if bias_std > 0:
        bias = torch.randn(n_filters, generator=gen, dtype=dtype) * float(bias_std)
    else:
        bias = torch.zeros(n_filters, dtype=dtype)
    return weight.to(device), bias.to(device)


def circular_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """2D circular convolution for odd square kernels, preserving H x W."""

    k_h, k_w = weight.shape[-2:]
    if k_h != k_w or k_h % 2 != 1:
        raise ValueError(f"kernel must be odd and square, got {k_h}x{k_w}")
    pad = k_h // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="circular")
    return F.conv2d(x_pad, weight, bias=bias)


def conv_rf_features(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Fixed ReLU random convolutional features."""

    return torch.relu(circular_conv2d(x, weight, bias))


def multiscale_conv_rf_features(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    scales: list[int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Concatenate random conv features computed at several downsampled scales.

    scales are downsampling factors: 1 keeps 32x32, 2 uses 16x16, 4 uses 8x8.
    Features are upsampled back to the input spatial size before concatenation.
    """

    if not (len(weights) == len(biases) == len(scales)):
        raise ValueError("weights, biases, and scales must have equal length")
    h, w = x.shape[-2:]
    feats = []
    for weight, bias, scale in zip(weights, biases, scales):
        if scale == 1:
            xs = x
        else:
            xs = F.avg_pool2d(x, kernel_size=scale, stride=scale)
        fs = conv_rf_features(xs, weight, bias)
        if fs.shape[-2:] != (h, w):
            fs = F.interpolate(fs, size=(h, w), mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)
        feats.append(fs)
    return torch.cat(feats, dim=1)


@torch.no_grad()
def dense_wiener_precompute(x0: torch.Tensor, *, eps: float = 0.0) -> DenseWienerPrecomp:
    """Precompute sample covariance eigensystem for full dense Wiener filtering."""

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (N,C,H,W), got {tuple(x0.shape)}")
    n = x0.shape[0]
    flat = x0.reshape(n, -1)
    mean = flat.mean(0)
    xc = flat - mean
    sigma_x = xc.T @ xc / n
    eigvals, eigvecs = torch.linalg.eigh(sigma_x)
    eigvals = eigvals.clamp(min=eps)
    return DenseWienerPrecomp(mean=mean, eigvals=eigvals, eigvecs=eigvecs, trace=float(eigvals.sum()))


@torch.no_grad()
def dense_wiener_predict(y: torch.Tensor, precomp: DenseWienerPrecomp, sigma: float, lam: float = 1e-6) -> torch.Tensor:
    """Apply the full dense Wiener estimator E_lin[x0 | y] to image batches."""

    b, c, h, w = y.shape
    flat = y.reshape(b, -1)
    yc = flat - precomp.mean[None, :]
    coeff = precomp.eigvals / (precomp.eigvals + float(sigma) ** 2 + float(lam))
    vt_y = yc @ precomp.eigvecs
    pred = precomp.mean[None, :] + (vt_y * coeff[None, :]) @ precomp.eigvecs.T
    return pred.reshape(b, c, h, w)


def dense_wiener_loss_from_precomp(precomp: DenseWienerPrecomp, sigma: float, lam: float = 1e-6) -> float:
    """Analytic full dense Wiener loss for the precomputed sample covariance."""

    inv = 1.0 / (precomp.eigvals + float(sigma) ** 2 + float(lam))
    explained = float((precomp.eigvals ** 2 * inv).sum())
    return max(0.0, precomp.trace - explained)


def circular_unfold2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Circular-padding im2col, returning (B, C*kernel_size^2, H*W)."""

    if kernel_size % 2 != 1:
        raise ValueError(f"readout patch size must be odd, got {kernel_size}")
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="circular")
    return F.unfold(x_pad, kernel_size=kernel_size)


def _as_freq_channels(x: torch.Tensor) -> torch.Tensor:
    """Return unitary FFT as (B, H*W, C)."""

    xf = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return xf.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[-2] * x.shape[-1], x.shape[1])


@torch.no_grad()
def accumulate_conv_rf_stats(
    x0: torch.Tensor,
    sigma: float,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    n_noise: int = 5,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float32,
    stats_dtype: torch.dtype = torch.complex64,
    include_noisy_input: bool = False,
    progress_label: str | None = None,
    show_progress: bool = False,
) -> ConvRFStats:
    """Stream sufficient FFT-domain statistics without storing all features."""

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm is optional for library use.
        tqdm = None

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (N,C,H,W), got {tuple(x0.shape)}")
    device = x0.device
    n, c, h, w = x0.shape
    f_rf = weight.shape[0]
    f = f_rf + c if include_noisy_input else f_rf
    l_freq = h * w
    cdtype = stats_dtype

    sum_phi = torch.zeros(l_freq, f, device=device, dtype=cdtype)
    sum_x = torch.zeros(l_freq, c, device=device, dtype=cdtype)
    sum_phiphi = torch.zeros(l_freq, f, f, device=device, dtype=cdtype)
    sum_xphi = torch.zeros(l_freq, c, f, device=device, dtype=cdtype)
    sum_x_abs2 = torch.zeros(l_freq, device=device, dtype=torch.float64)

    starts = list(range(0, n, batch_size))
    outer: Iterable[int] = range(n_noise)
    if show_progress and tqdm is not None:
        desc = progress_label or f"sigma={sigma:.4g}, features={f}, k={weight.shape[-1]}"
        outer = tqdm(outer, desc=desc)

    start_time = time.perf_counter()
    for _ in outer:
        for start in starts:
            end = min(start + batch_size, n)
            xb = x0[start:end].to(dtype=dtype)
            yb = xb + torch.randn_like(xb) * float(sigma)
            phib = conv_rf_features(yb, weight.to(dtype=dtype), bias.to(dtype=dtype) if bias is not None else None)

            xf = _as_freq_channels(xb).to(cdtype)
            pf = _as_freq_channels(phib).to(cdtype)
            if include_noisy_input:
                yf = _as_freq_channels(yb).to(cdtype)
                pf = torch.cat([yf, pf], dim=-1)

            sum_x += xf.sum(0)
            sum_phi += pf.sum(0)
            sum_x_abs2 += (xf.abs() ** 2).sum(dim=(0, 2)).double()
            sum_phiphi += torch.einsum("blf,blg->lfg", pf, pf.conj())
            sum_xphi += torch.einsum("blc,blf->lcf", xf, pf.conj())

            del xb, yb, phib, xf, pf

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    return ConvRFStats(
        sigma=float(sigma),
        n_samples=n,
        n_noise=int(n_noise),
        n_total=n * int(n_noise),
        n_channels=c,
        height=h,
        width=w,
        n_filters=f,
        kernel_size=weight.shape[-1],
        sum_phi=sum_phi,
        sum_x=sum_x,
        sum_phiphi=sum_phiphi,
        sum_xphi=sum_xphi,
        sum_x_abs2=sum_x_abs2,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def accumulate_dense_wiener_residual_stats(
    x0: torch.Tensor,
    sigma: float,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    wiener: DenseWienerPrecomp,
    *,
    n_noise: int = 5,
    batch_size: int = 256,
    dense_lam: float = 1e-6,
    dtype: torch.dtype = torch.float32,
    stats_dtype: torch.dtype = torch.complex64,
    progress_label: str | None = None,
    show_progress: bool = False,
) -> ConvRFStats:
    """Accumulate stats for conv-RF prediction of the full-Wiener residual.

    The solved model is exactly

        xhat(y) = dense_wiener(y) + conv_readout(phi(y)).

    The target in the covariance formula is

        residual = x0 - dense_wiener(y).
    """

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (N,C,H,W), got {tuple(x0.shape)}")
    device = x0.device
    n, c, h, w = x0.shape
    f = weight.shape[0]
    l_freq = h * w
    cdtype = stats_dtype

    sum_phi = torch.zeros(l_freq, f, device=device, dtype=cdtype)
    sum_r = torch.zeros(l_freq, c, device=device, dtype=cdtype)
    sum_phiphi = torch.zeros(l_freq, f, f, device=device, dtype=cdtype)
    sum_rphi = torch.zeros(l_freq, c, f, device=device, dtype=cdtype)
    sum_r_abs2 = torch.zeros(l_freq, device=device, dtype=torch.float64)

    starts = list(range(0, n, batch_size))
    outer: Iterable[int] = range(n_noise)
    if show_progress and tqdm is not None:
        desc = progress_label or f"sigma={sigma:.4g}, Wiener residual, F={f}, k={weight.shape[-1]}"
        outer = tqdm(outer, desc=desc)

    start_time = time.perf_counter()
    for _ in outer:
        for start in starts:
            end = min(start + batch_size, n)
            xb = x0[start:end].to(dtype=dtype)
            yb = xb + torch.randn_like(xb) * float(sigma)
            pred = dense_wiener_predict(yb, wiener, sigma, lam=dense_lam).to(dtype=dtype)
            rb = xb - pred
            phib = conv_rf_features(yb, weight.to(dtype=dtype), bias.to(dtype=dtype) if bias is not None else None)

            rf = _as_freq_channels(rb).to(cdtype)
            pf = _as_freq_channels(phib).to(cdtype)

            sum_r += rf.sum(0)
            sum_phi += pf.sum(0)
            sum_r_abs2 += (rf.abs() ** 2).sum(dim=(0, 2)).double()
            sum_phiphi += torch.einsum("blf,blg->lfg", pf, pf.conj())
            sum_rphi += torch.einsum("blc,blf->lcf", rf, pf.conj())

            del xb, yb, pred, rb, phib, rf, pf

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    return ConvRFStats(
        sigma=float(sigma),
        n_samples=n,
        n_noise=int(n_noise),
        n_total=n * int(n_noise),
        n_channels=c,
        height=h,
        width=w,
        n_filters=f,
        kernel_size=weight.shape[-1],
        sum_phi=sum_phi,
        sum_x=sum_r,
        sum_phiphi=sum_phiphi,
        sum_xphi=sum_rphi,
        sum_x_abs2=sum_r_abs2,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def accumulate_multiscale_dense_wiener_residual_stats(
    x0: torch.Tensor,
    sigma: float,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    scales: list[int],
    wiener: DenseWienerPrecomp,
    *,
    n_noise: int = 5,
    batch_size: int = 256,
    dense_lam: float = 1e-6,
    dtype: torch.dtype = torch.float32,
    stats_dtype: torch.dtype = torch.complex64,
    progress_label: str | None = None,
    show_progress: bool = False,
) -> ConvRFStats:
    """Accumulate FFT stats for multiscale conv-RF prediction of Wiener residual."""

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (N,C,H,W), got {tuple(x0.shape)}")
    device = x0.device
    n, c, h, w = x0.shape
    f = sum(weight.shape[0] for weight in weights)
    l_freq = h * w
    cdtype = stats_dtype

    sum_phi = torch.zeros(l_freq, f, device=device, dtype=cdtype)
    sum_r = torch.zeros(l_freq, c, device=device, dtype=cdtype)
    sum_phiphi = torch.zeros(l_freq, f, f, device=device, dtype=cdtype)
    sum_rphi = torch.zeros(l_freq, c, f, device=device, dtype=cdtype)
    sum_r_abs2 = torch.zeros(l_freq, device=device, dtype=torch.float64)

    starts = list(range(0, n, batch_size))
    outer: Iterable[int] = range(n_noise)
    if show_progress and tqdm is not None:
        desc = progress_label or f"sigma={sigma:.4g}, multiscale dense residual RF, F={f}, scales={scales}"
        outer = tqdm(outer, desc=desc)

    start_time = time.perf_counter()
    for _ in outer:
        for start in starts:
            end = min(start + batch_size, n)
            xb = x0[start:end].to(dtype=dtype)
            yb = xb + torch.randn_like(xb) * float(sigma)
            pred = dense_wiener_predict(yb, wiener, sigma, lam=dense_lam).to(dtype=dtype)
            rb = xb - pred
            phib = multiscale_conv_rf_features(
                yb,
                [wgt.to(dtype=dtype) for wgt in weights],
                [b.to(dtype=dtype) for b in biases],
                scales,
            )
            rf = _as_freq_channels(rb).to(cdtype)
            pf = _as_freq_channels(phib).to(cdtype)
            sum_r += rf.sum(0)
            sum_phi += pf.sum(0)
            sum_r_abs2 += (rf.abs() ** 2).sum(dim=(0, 2)).double()
            sum_phiphi += torch.einsum("blf,blg->lfg", pf, pf.conj())
            sum_rphi += torch.einsum("blc,blf->lcf", rf, pf.conj())
            del xb, yb, pred, rb, phib, rf, pf

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    return ConvRFStats(
        sigma=float(sigma),
        n_samples=n,
        n_noise=int(n_noise),
        n_total=n * int(n_noise),
        n_channels=c,
        height=h,
        width=w,
        n_filters=f,
        kernel_size=-1,
        sum_phi=sum_phi,
        sum_x=sum_r,
        sum_phiphi=sum_phiphi,
        sum_xphi=sum_rphi,
        sum_x_abs2=sum_r_abs2,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def accumulate_dense_wiener_residual_patch_stats(
    x0: torch.Tensor,
    sigma: float,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    wiener: DenseWienerPrecomp,
    *,
    readout_patch_size: int = 3,
    n_noise: int = 5,
    batch_size: int = 32,
    dense_lam: float = 1e-6,
    dtype: torch.dtype = torch.float32,
    stats_dtype: torch.dtype = torch.float32,
    progress_label: str | None = None,
    show_progress: bool = False,
) -> PatchReadoutStats:
    """Accumulate local spatial patch stats for dense-Wiener residual prediction.

    This solves a shared local readout from patches of conv-RF feature maps:

        residual[p] ~= A * vec(phi[p + local_offsets]).

    Unlike the FFT estimator, this constrains the readout kernel to a finite
    spatial support and pools covariance estimates over all image positions.
    """

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (N,C,H,W), got {tuple(x0.shape)}")
    device = x0.device
    n, c, h, w = x0.shape
    f = weight.shape[0]
    patch_dim = f * readout_patch_size * readout_patch_size

    sum_feat = torch.zeros(patch_dim, device=device, dtype=stats_dtype)
    sum_target = torch.zeros(c, device=device, dtype=stats_dtype)
    sum_featfeat = torch.zeros(patch_dim, patch_dim, device=device, dtype=stats_dtype)
    sum_targetfeat = torch.zeros(c, patch_dim, device=device, dtype=stats_dtype)
    sum_target_abs2 = torch.zeros((), device=device, dtype=torch.float64)

    starts = list(range(0, n, batch_size))
    outer: Iterable[int] = range(n_noise)
    if show_progress and tqdm is not None:
        desc = progress_label or (
            f"sigma={sigma:.4g}, dense residual local RF, F={f}, "
            f"feature_k={weight.shape[-1]}, readout_k={readout_patch_size}"
        )
        outer = tqdm(outer, desc=desc)

    start_time = time.perf_counter()
    n_total_patches = 0
    for _ in outer:
        for start in starts:
            end = min(start + batch_size, n)
            xb = x0[start:end].to(dtype=dtype)
            yb = xb + torch.randn_like(xb) * float(sigma)
            pred = dense_wiener_predict(yb, wiener, sigma, lam=dense_lam).to(dtype=dtype)
            rb = xb - pred
            phib = conv_rf_features(yb, weight.to(dtype=dtype), bias.to(dtype=dtype) if bias is not None else None)

            patches = circular_unfold2d(phib, readout_patch_size).transpose(1, 2)
            patches = patches.reshape(-1, patch_dim).to(stats_dtype)
            target = rb.permute(0, 2, 3, 1).reshape(-1, c).to(stats_dtype)

            sum_feat += patches.sum(0)
            sum_target += target.sum(0)
            sum_featfeat += patches.T @ patches
            sum_targetfeat += target.T @ patches
            sum_target_abs2 += (target.double() ** 2).sum()
            n_total_patches += patches.shape[0]

            del xb, yb, pred, rb, phib, patches, target

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    return PatchReadoutStats(
        sigma=float(sigma),
        n_samples=n,
        n_noise=int(n_noise),
        n_total_images=n * int(n_noise),
        n_total_patches=n_total_patches,
        n_channels=c,
        height=h,
        width=w,
        n_filters=f,
        feature_kernel_size=weight.shape[-1],
        readout_patch_size=readout_patch_size,
        sum_feat=sum_feat,
        sum_target=sum_target,
        sum_featfeat=sum_featfeat,
        sum_targetfeat=sum_targetfeat,
        sum_target_abs2=sum_target_abs2,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def mmse_from_patch_stats(
    stats: PatchReadoutStats,
    *,
    lam: float = 1e-4,
    bias_mode: str = "channel",
) -> dict[str, float]:
    """MMSE of a shared local spatial readout from patch stats.

    The returned loss is image-level MSE (sum over all pixels/channels), not
    per-pixel MSE, so it is directly comparable to the FFT/dense-Wiener curves.
    """

    if bias_mode not in {"channel", "none"}:
        raise ValueError(f"patch stats support bias_mode='channel' or 'none', got {bias_mode!r}")

    m = stats.n_total_patches
    p = stats.sum_feat.shape[0]
    locations = stats.height * stats.width
    sig = stats.sum_featfeat / m
    cov = stats.sum_targetfeat / m
    trace_pixel = stats.sum_target_abs2 / m

    if bias_mode == "channel":
        mean_feat = stats.sum_feat / m
        mean_target = stats.sum_target / m
        sig = sig - mean_feat[:, None] * mean_feat[None, :]
        cov = cov - mean_target[:, None] * mean_feat[None, :]
        trace_pixel = trace_pixel - (mean_target.double() ** 2).sum()

    reg = sig + float(lam) * torch.eye(p, device=stats.sum_feat.device, dtype=sig.dtype)
    sol = torch.linalg.solve(reg, cov.T)
    explained_pixel = ((cov * sol.T).sum()).double()
    loss_pixel = torch.clamp(trace_pixel - explained_pixel, min=0.0)
    trace = float(trace_pixel * locations)
    explained = float(explained_pixel * locations)
    loss = float(loss_pixel * locations)
    return {
        "loss": loss,
        "trace": trace,
        "explained": explained,
        "r2": explained / trace if trace > 0 else 0.0,
        "bias_mode": bias_mode,
    }


@torch.no_grad()
def mmse_from_conv_stats(
    stats: ConvRFStats,
    *,
    lam: float = 1e-4,
    bias_mode: str = "channel",
) -> dict[str, float]:
    """Compute convolutional-readout MMSE from accumulated FFT statistics.

    bias_mode:
      - "channel": true conv-net affine readout, one free bias per output
        channel. Only the DC frequency is centered.
      - "free_spatial": allows arbitrary spatial bias by centering every
        frequency. This is useful for comparison to dense affine estimators.
      - "none": no intercept; all frequencies use raw second moments.
    """

    if bias_mode not in {"channel", "free_spatial", "none"}:
        raise ValueError(f"unknown bias_mode={bias_mode!r}")

    m = stats.n_total
    l_freq, n_filters = stats.sum_phi.shape
    eye = torch.eye(n_filters, device=stats.sum_phi.device, dtype=stats.sum_phiphi.dtype)

    sigma_phi = stats.sum_phiphi / m
    cov_xphi = stats.sum_xphi / m
    trace_freq = stats.sum_x_abs2 / m

    if bias_mode in {"channel", "free_spatial"}:
        mean_phi = stats.sum_phi / m
        mean_x = stats.sum_x / m
        centered_sigma = sigma_phi - mean_phi[:, :, None] * mean_phi.conj()[:, None, :]
        centered_cov = cov_xphi - mean_x[:, :, None] * mean_phi.conj()[:, None, :]
        centered_trace = trace_freq - (mean_x.abs() ** 2).sum(-1).double()

        if bias_mode == "free_spatial":
            sigma_phi = centered_sigma
            cov_xphi = centered_cov
            trace_freq = centered_trace
        else:
            sigma_phi = sigma_phi.clone()
            cov_xphi = cov_xphi.clone()
            trace_freq = trace_freq.clone()
            sigma_phi[0] = centered_sigma[0]
            cov_xphi[0] = centered_cov[0]
            trace_freq[0] = centered_trace[0]

    reg = sigma_phi + float(lam) * eye[None, :, :]
    rhs = cov_xphi.conj().transpose(-1, -2)
    sol = torch.linalg.solve(reg, rhs)
    explained_freq = torch.einsum("lcf,lfc->l", cov_xphi, sol).real.double()
    explained = float(explained_freq.sum())
    trace = float(trace_freq.sum())
    loss = max(0.0, trace - explained)
    return {
        "loss": loss,
        "trace": trace,
        "explained": explained,
        "r2": explained / trace if trace > 0 else 0.0,
        "bias_mode": bias_mode,
    }


@torch.no_grad()
def conv_linear_mmse_fft(
    x0: torch.Tensor,
    sigma: float,
    *,
    lam: float = 1e-6,
    bias_mode: str = "channel",
) -> dict[str, float]:
    """Closed-form MMSE for the best circular-convolutional linear denoiser."""

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (N,C,H,W), got {tuple(x0.shape)}")
    n, c, h, w = x0.shape
    xf = _as_freq_channels(x0)
    l_freq = h * w
    m = n
    sum_x = xf.sum(0)
    sum_xx = torch.einsum("blc,bld->lcd", xf, xf.conj()) / m
    trace_freq = (xf.abs() ** 2).sum(dim=(0, 2)).double() / m

    # y = x + sigma z. With unitary FFT, white pixel noise remains white with
    # variance sigma^2 independently at each frequency/channel.
    sigma_y = sum_xx + (float(sigma) ** 2) * torch.eye(c, device=x0.device, dtype=xf.dtype)[None, :, :]
    cov_xy = sum_xx.clone()

    if bias_mode in {"channel", "free_spatial"}:
        mean_x = sum_x / m
        centered_xx = sum_xx - mean_x[:, :, None] * mean_x.conj()[:, None, :]
        centered_trace = trace_freq - (mean_x.abs() ** 2).sum(-1).double()
        centered_y = centered_xx + (float(sigma) ** 2) * torch.eye(c, device=x0.device, dtype=xf.dtype)[None, :, :]
        if bias_mode == "free_spatial":
            sigma_y = centered_y
            cov_xy = centered_xx
            trace_freq = centered_trace
        elif bias_mode == "channel":
            sigma_y = sigma_y.clone()
            cov_xy = cov_xy.clone()
            trace_freq = trace_freq.clone()
            sigma_y[0] = centered_y[0]
            cov_xy[0] = centered_xx[0]
            trace_freq[0] = centered_trace[0]
    elif bias_mode != "none":
        raise ValueError(f"unknown bias_mode={bias_mode!r}")

    eye = torch.eye(c, device=x0.device, dtype=xf.dtype)
    sol = torch.linalg.solve(sigma_y + float(lam) * eye[None, :, :], cov_xy.conj().transpose(-1, -2))
    explained = float(torch.einsum("lcd,ldc->l", cov_xy, sol).real.double().sum())
    trace = float(trace_freq.sum())
    return {
        "loss": max(0.0, trace - explained),
        "trace": trace,
        "explained": explained,
        "r2": explained / trace if trace > 0 else 0.0,
        "bias_mode": bias_mode,
    }


def stats_to_numpy(stats: ConvRFStats) -> dict[str, np.ndarray | float | int]:
    """Move compact sufficient statistics to CPU numpy for caching."""

    return {
        "sigma": stats.sigma,
        "n_samples": stats.n_samples,
        "n_noise": stats.n_noise,
        "n_total": stats.n_total,
        "n_channels": stats.n_channels,
        "height": stats.height,
        "width": stats.width,
        "n_filters": stats.n_filters,
        "kernel_size": stats.kernel_size,
        "sum_phi": stats.sum_phi.detach().cpu().numpy(),
        "sum_x": stats.sum_x.detach().cpu().numpy(),
        "sum_phiphi": stats.sum_phiphi.detach().cpu().numpy(),
        "sum_xphi": stats.sum_xphi.detach().cpu().numpy(),
        "sum_x_abs2": stats.sum_x_abs2.detach().cpu().numpy(),
        "elapsed_seconds": stats.elapsed_seconds,
    }


def synthetic_smoke_test(device: str = "cpu") -> dict[str, float]:
    """Tiny estimator sanity check used by scripts/conv_rf_cifar10.py."""

    torch.manual_seed(0)
    x = torch.randn(32, 1, 4, 4, device=device)
    weight, bias = make_random_conv_filters(3, 1, 3, seed=1, device=device)
    stats = accumulate_conv_rf_stats(x, 0.2, weight, bias, n_noise=2, batch_size=8)
    rf = mmse_from_conv_stats(stats, lam=1e-5, bias_mode="channel")
    lin = conv_linear_mmse_fft(x, 0.2, lam=1e-6, bias_mode="channel")
    if not np.isfinite(rf["loss"]) or not np.isfinite(lin["loss"]):
        raise RuntimeError("non-finite smoke-test loss")
    if rf["loss"] < -1e-6 or lin["loss"] < -1e-6:
        raise RuntimeError("negative smoke-test loss")
    return {"conv_rf_loss": rf["loss"], "conv_linear_loss": lin["loss"]}
