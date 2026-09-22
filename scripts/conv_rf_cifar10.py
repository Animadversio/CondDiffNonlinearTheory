"""
CIFAR-10 convolutional random-feature MMSE experiment.

This measures the best circular-convolutional linear readout from fixed random
ReLU convolutional features. The readout is solved in closed form by 2D FFT:
one small F x F ridge solve per spatial frequency.

Examples:
    python scripts/conv_rf_cifar10.py --smoke
    python scripts/conv_rf_cifar10.py --n_samples 2000 --n_noise 2 \
        --sigmas 0.3 0.45 0.85 --num_filters 4 16 64 --kernel_sizes 3 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conv_rf_mmse import (  # noqa: E402
    ConvRFStats,
    accumulate_conv_rf_stats,
    accumulate_dense_wiener_residual_stats,
    accumulate_dense_wiener_residual_patch_stats,
    conv_linear_mmse_fft,
    dense_wiener_loss_from_precomp,
    dense_wiener_precompute,
    make_random_conv_filters,
    mmse_from_conv_stats,
    mmse_from_patch_stats,
    synthetic_smoke_test,
)


STORE = os.environ.get(
    "STORE_DIR",
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run a tiny synthetic smoke test and exit")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_noise", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--patch_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.45, 0.65, 0.85, 1.2, 1.7, 2.2])
    p.add_argument("--num_filters", type=int, nargs="+", default=[4, 16, 64])
    p.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 5])
    p.add_argument("--readout_patch_sizes", type=int, nargs="+", default=[])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--dense_lam", type=float, default=1e-6)
    p.add_argument("--bias_std", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stats_dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--artifact_dir", default=os.path.join("tables", "conv_rf_cifar10"))
    p.add_argument("--figure_dir", default="figures")
    p.add_argument("--tag", default="pilot")
    p.add_argument("--baseline_npz", default="tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz")
    return p.parse_args()


def load_cifar10(n_samples: int, device: str, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    data_root = os.path.join(STORE, "Datasets")
    if not os.path.exists(os.path.join(data_root, "cifar-10-batches-py")):
        data_root = "data" if os.path.exists("data") else "/tmp/cifar10"
    ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=T.ToTensor(),
    )
    n_samples = min(n_samples, len(ds))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), n_samples, replace=False).tolist()
    subset = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=512,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        persistent_workers=num_workers > 0,
    )
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x0 = torch.cat(xs).to(device=device, dtype=torch.float32)
    labels = torch.cat(ys).to(device=device)
    return x0, labels


def _slice_stats(stats: ConvRFStats, n_feature_channels: int) -> ConvRFStats:
    return replace(
        stats,
        n_filters=n_feature_channels,
        sum_phi=stats.sum_phi[:, :n_feature_channels],
        sum_phiphi=stats.sum_phiphi[:, :n_feature_channels, :n_feature_channels],
        sum_xphi=stats.sum_xphi[:, :, :n_feature_channels],
    )


def _range_stats(stats: ConvRFStats, start: int, end: int) -> ConvRFStats:
    return replace(
        stats,
        n_filters=end - start,
        sum_phi=stats.sum_phi[:, start:end],
        sum_phiphi=stats.sum_phiphi[:, start:end, start:end],
        sum_xphi=stats.sum_xphi[:, :, start:end],
    )


def _slice_patch_stats(stats, n_filters: int):
    readout_area = stats.readout_patch_size * stats.readout_patch_size
    p = n_filters * readout_area
    return replace(
        stats,
        n_filters=n_filters,
        sum_feat=stats.sum_feat[:p],
        sum_featfeat=stats.sum_featfeat[:p, :p],
        sum_targetfeat=stats.sum_targetfeat[:, :p],
    )


def _init_result_arrays(args: argparse.Namespace) -> dict[str, np.ndarray]:
    shape = (len(args.kernel_sizes), len(args.seeds), len(args.sigmas), len(args.num_filters))
    patch_shape = shape + (len(args.readout_patch_sizes),)
    res = {
        "sigma": np.asarray(args.sigmas, dtype=np.float64),
        "num_filters": np.asarray(args.num_filters, dtype=np.int64),
        "kernel_sizes": np.asarray(args.kernel_sizes, dtype=np.int64),
        "readout_patch_sizes": np.asarray(args.readout_patch_sizes, dtype=np.int64),
        "seeds": np.asarray(args.seeds, dtype=np.int64),
        "conv_linear_channel": np.full(len(args.sigmas), np.nan),
        "conv_linear_free_spatial": np.full(len(args.sigmas), np.nan),
        "dense_wiener": np.full(len(args.sigmas), np.nan),
        "conv_rf_channel": np.full(shape, np.nan),
        "conv_rf_free_spatial": np.full(shape, np.nan),
        "linear_plus_conv_rf_channel": np.full(shape, np.nan),
        "linear_plus_conv_rf_free_spatial": np.full(shape, np.nan),
        "dense_wiener_plus_conv_rf_channel": np.full(shape, np.nan),
        "dense_wiener_residual_trace_channel": np.full(shape, np.nan),
        "dense_wiener_plus_conv_rf_patch_channel": np.full(patch_shape, np.nan),
        "dense_wiener_residual_trace_patch_channel": np.full(patch_shape, np.nan),
        "elapsed_rf_seconds": np.full(shape[:3], np.nan),
        "elapsed_combo_seconds": np.full(shape[:3], np.nan),
        "elapsed_dense_residual_seconds": np.full(shape[:3], np.nan),
        "elapsed_dense_residual_patch_seconds": np.full(shape[:3] + (len(args.readout_patch_sizes),), np.nan),
    }
    return res


def _save_npz(path: str, res: dict[str, np.ndarray], args: argparse.Namespace, extra: dict[str, str | float | int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {
        "n_samples": args.n_samples,
        "n_noise": args.n_noise,
        "batch_size": args.batch_size,
        "patch_batch_size": args.patch_batch_size,
        "num_workers": args.num_workers,
        "lam": args.lam,
        "dense_lam": args.dense_lam,
        "bias_std": args.bias_std,
        "device": str(args.device),
        "stats_dtype": args.stats_dtype,
        **extra,
    }
    np.savez(path, **res, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def _plot(path: str, res: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    sigma = res["sigma"]
    os.makedirs(args.figure_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"CIFAR-10 convolutional RF MMSE | N={args.n_samples}, n_noise={args.n_noise}, tag={args.tag}",
        fontsize=12,
    )

    baseline = None
    if os.path.exists(args.baseline_npz):
        baseline = np.load(args.baseline_npz)

    k0 = 0
    s0 = 0
    filt_to_show = [args.num_filters[0], args.num_filters[len(args.num_filters) // 2], args.num_filters[-1]]
    filt_to_show = sorted(set(filt_to_show))

    ax = axes[0]
    ax.plot(sigma, res["conv_linear_channel"], "k-", lw=2.2, label="conv linear")
    if "dense_wiener" in res:
        ax.plot(sigma, res["dense_wiener"], color="0.25", lw=2.0, ls="--", label="full dense Wiener")
    for nf in filt_to_show:
        fi = list(args.num_filters).index(nf)
        ax.plot(
            sigma,
            res["conv_rf_channel"][k0, s0, :, fi],
            marker="o",
            lw=1.8,
            label=f"conv RF F={nf}",
        )
        ax.plot(
            sigma,
            res["linear_plus_conv_rf_channel"][k0, s0, :, fi],
            marker="^",
            lw=1.5,
            ls="--",
            label=f"[y;RF] F={nf}",
        )
        if "dense_wiener_plus_conv_rf_channel" in res:
            ax.plot(
                sigma,
                res["dense_wiener_plus_conv_rf_channel"][k0, s0, :, fi],
                marker="v",
                lw=1.5,
                ls=":",
                label=f"dense Wiener + RF F={nf}",
            )
    if baseline is not None:
        ax.plot(baseline["sigma"], baseline["linear_uncond"], color="0.35", lw=1.8, ls=":", label="full linear Wiener")
        if "edm_uncond" in baseline:
            ax.plot(baseline["sigma"], baseline["edm_uncond"], color="crimson", lw=1.8, ls="-.", label="EDM uncond")
        if "bayes_uncond" in baseline:
            ax.plot(baseline["sigma"], baseline["bayes_uncond"], color="royalblue", lw=1.6, ls=":", label="oracle softmax uncond")
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("MSE")
    ax.set_title(f"Loss vs sigma, kernel={args.kernel_sizes[k0]}, seed={args.seeds[s0]}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.7)

    ax = axes[1]
    for nf in args.num_filters:
        fi = list(args.num_filters).index(nf)
        gain = res["conv_linear_channel"] - res["linear_plus_conv_rf_channel"][k0, s0, :, fi]
        ax.plot(sigma, gain, marker="o", lw=1.7, label=f"F={nf}")
    if "dense_wiener_plus_conv_rf_channel" in res:
        fi = len(args.num_filters) - 1
        ax.plot(
            sigma,
            res["dense_wiener"] - res["dense_wiener_plus_conv_rf_channel"][k0, s0, :, fi],
            color="C4",
            marker="v",
            lw=2.0,
            ls="--",
            label=f"dense Wiener gain, F={args.num_filters[fi]}",
        )
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("conv linear - [y;RF]")
    ax.set_title("Nonlinear conv-RF gain over conv linear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    target_sigma = 0.85
    si = int(np.argmin(np.abs(sigma - target_sigma)))
    for ki, kernel in enumerate(args.kernel_sizes):
        vals = np.nanmean(res["linear_plus_conv_rf_channel"][ki, :, si, :], axis=0)
        ax.plot(args.num_filters, vals, marker="o", lw=1.8, label=f"k={kernel}")
    ax.axhline(res["conv_linear_channel"][si], color="k", lw=1.5, ls="--", label="conv linear")
    if baseline is not None and "edm_uncond" in baseline:
        bi = int(np.argmin(np.abs(baseline["sigma"] - sigma[si])))
        ax.axhline(float(baseline["edm_uncond"][bi]), color="crimson", lw=1.5, ls="-.", label="EDM uncond")
    if "dense_wiener" in res:
        ax.axhline(res["dense_wiener"][si], color="0.35", lw=1.4, ls=":", label="full dense Wiener")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of random filters")
    ax.set_ylabel("MSE")
    ax.set_title(f"Filter sweep at sigma={sigma[si]:.3g}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {path}", flush=True)


def run(args: argparse.Namespace) -> tuple[str, str]:
    torch.manual_seed(0)
    device = torch.device(args.device)
    stats_dtype = torch.complex64 if args.stats_dtype == "complex64" else torch.complex128
    print(f"[setup] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(device)}", flush=True)
        print(f"[setup] cuda devices={torch.cuda.device_count()}", flush=True)

    x0, labels = load_cifar10(args.n_samples, str(device), args.num_workers)
    del labels
    print(f"[data] x0={tuple(x0.shape)}, mean={x0.mean().item():.4f}, std={x0.std().item():.4f}", flush=True)

    args.num_filters = sorted(args.num_filters)
    max_filters = max(args.num_filters)
    res = _init_result_arrays(args)

    print("[dense] precomputing full dense Wiener eigensystem ...", flush=True)
    dense_precomp = dense_wiener_precompute(x0)
    print(f"[dense] trace={dense_precomp.trace:.4f}", flush=True)

    for si, sigma in enumerate(tqdm(args.sigmas, desc="conv linear")):
        res["conv_linear_channel"][si] = conv_linear_mmse_fft(x0, sigma, lam=args.lam, bias_mode="channel")["loss"]
        res["conv_linear_free_spatial"][si] = conv_linear_mmse_fft(x0, sigma, lam=args.lam, bias_mode="free_spatial")["loss"]
        res["dense_wiener"][si] = dense_wiener_loss_from_precomp(dense_precomp, sigma, lam=args.dense_lam)

    total_settings = len(args.kernel_sizes) * len(args.seeds) * len(args.sigmas)
    done = 0
    start_all = time.perf_counter()
    out_npz = os.path.join(args.artifact_dir, f"conv_rf_cifar10_{args.tag}.npz")
    out_fig = os.path.join(args.figure_dir, f"conv_rf_cifar10_{args.tag}.png")

    for ki, kernel in enumerate(args.kernel_sizes):
        for ei, seed in enumerate(args.seeds):
            weight, bias = make_random_conv_filters(
                max_filters,
                3,
                kernel,
                seed=seed,
                device=device,
                bias_std=args.bias_std,
            )
            for si, sigma in enumerate(args.sigmas):
                setting_start = time.perf_counter()
                print(
                    f"[run] kernel={kernel} seed={seed} sigma={sigma:.4g} maxF={max_filters}",
                    flush=True,
                )
                combo_stats = accumulate_conv_rf_stats(
                    x0,
                    sigma,
                    weight,
                    bias,
                    n_noise=args.n_noise,
                    batch_size=args.batch_size,
                    stats_dtype=stats_dtype,
                    include_noisy_input=True,
                    progress_label=f"sigma={sigma:.4g}, [y;RF], maxF={max_filters}, k={kernel}",
                    show_progress=True,
                )
                dense_resid_stats = accumulate_dense_wiener_residual_stats(
                    x0,
                    sigma,
                    weight,
                    bias,
                    dense_precomp,
                    n_noise=args.n_noise,
                    batch_size=args.batch_size,
                    dense_lam=args.dense_lam,
                    stats_dtype=stats_dtype,
                    progress_label=f"sigma={sigma:.4g}, dense-Wiener residual RF, maxF={max_filters}, k={kernel}",
                    show_progress=True,
                )
                res["elapsed_rf_seconds"][ki, ei, si] = 0.0
                res["elapsed_combo_seconds"][ki, ei, si] = combo_stats.elapsed_seconds
                res["elapsed_dense_residual_seconds"][ki, ei, si] = dense_resid_stats.elapsed_seconds

                for fi, nf in enumerate(args.num_filters):
                    rf_sub = _range_stats(combo_stats, 3, 3 + nf)
                    combo_sub = _slice_stats(combo_stats, 3 + nf)
                    dense_resid_sub = _slice_stats(dense_resid_stats, nf)
                    res["conv_rf_channel"][ki, ei, si, fi] = mmse_from_conv_stats(
                        rf_sub, lam=args.lam, bias_mode="channel"
                    )["loss"]
                    res["conv_rf_free_spatial"][ki, ei, si, fi] = mmse_from_conv_stats(
                        rf_sub, lam=args.lam, bias_mode="free_spatial"
                    )["loss"]
                    res["linear_plus_conv_rf_channel"][ki, ei, si, fi] = mmse_from_conv_stats(
                        combo_sub, lam=args.lam, bias_mode="channel"
                    )["loss"]
                    res["linear_plus_conv_rf_free_spatial"][ki, ei, si, fi] = mmse_from_conv_stats(
                        combo_sub, lam=args.lam, bias_mode="free_spatial"
                    )["loss"]
                    dense_resid = mmse_from_conv_stats(
                        dense_resid_sub, lam=args.lam, bias_mode="channel"
                    )
                    res["dense_wiener_plus_conv_rf_channel"][ki, ei, si, fi] = dense_resid["loss"]
                    res["dense_wiener_residual_trace_channel"][ki, ei, si, fi] = dense_resid["trace"]

                for pi, readout_patch in enumerate(args.readout_patch_sizes):
                    patch_stats = accumulate_dense_wiener_residual_patch_stats(
                        x0,
                        sigma,
                        weight,
                        bias,
                        dense_precomp,
                        readout_patch_size=readout_patch,
                        n_noise=args.n_noise,
                        batch_size=args.patch_batch_size,
                        dense_lam=args.dense_lam,
                        stats_dtype=torch.float32,
                        progress_label=(
                            f"sigma={sigma:.4g}, dense residual local RF, maxF={max_filters}, "
                            f"feature_k={kernel}, readout_k={readout_patch}"
                        ),
                        show_progress=True,
                    )
                    res["elapsed_dense_residual_patch_seconds"][ki, ei, si, pi] = patch_stats.elapsed_seconds
                    for fi, nf in enumerate(args.num_filters):
                        patch_sub = _slice_patch_stats(patch_stats, nf)
                        patch_res = mmse_from_patch_stats(patch_sub, lam=args.lam, bias_mode="channel")
                        res["dense_wiener_plus_conv_rf_patch_channel"][ki, ei, si, fi, pi] = patch_res["loss"]
                        res["dense_wiener_residual_trace_patch_channel"][ki, ei, si, fi, pi] = patch_res["trace"]
                    del patch_stats

                done += 1
                elapsed_all = time.perf_counter() - start_all
                eta = elapsed_all * (total_settings - done) / max(done, 1)
                print(
                    f"[done] setting_seconds={time.perf_counter() - setting_start:.1f} "
                    f"progress={done}/{total_settings} eta_minutes={eta / 60:.1f}",
                    flush=True,
                )
                _save_npz(out_npz, res, args, {"last_completed": done})
                del combo_stats, dense_resid_stats
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    _save_npz(out_npz, res, args, {"last_completed": done})
    _plot(out_fig, res, args)
    return out_npz, out_fig


def main() -> None:
    args = parse_args()
    if args.smoke:
        print("[smoke] running synthetic estimator smoke test", flush=True)
        print(synthetic_smoke_test(args.device), flush=True)
        return
    out_npz, out_fig = run(args)
    print(f"[complete] saved {out_npz}", flush=True)
    print(f"[complete] saved {out_fig}", flush=True)


if __name__ == "__main__":
    main()
