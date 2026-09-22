"""
CIFAR-10 dense-Wiener residual correction with multi-scale conv RF features.

Features are frozen random Conv2D+ReLU layers at scales like 32x32, 16x16,
and 8x8, upsampled to 32x32 and solved with the FFT convolutional readout.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conv_rf_mmse import (
    accumulate_multiscale_dense_wiener_residual_stats,
    dense_wiener_loss_from_precomp,
    dense_wiener_precompute,
    make_random_conv_filters,
    mmse_from_conv_stats,
)


STORE = os.environ.get(
    "STORE_DIR",
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_noise", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.45, 0.65, 0.85, 1.2, 1.7, 2.2])
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--filters_per_scale", type=int, nargs="+", default=[16, 32])
    p.add_argument("--scale_sets", nargs="+", default=["1", "1,2", "1,2,4"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--dense_lam", type=float, default=1e-6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--artifact_dir", default=os.path.join("tables", "conv_rf_cifar10"))
    p.add_argument("--figure_dir", default="figures")
    p.add_argument("--tag", default="pilot")
    return p.parse_args()


def load_cifar10(n: int, device: str) -> torch.Tensor:
    root = os.path.join(STORE, "Datasets")
    if not os.path.exists(os.path.join(root, "cifar-10-batches-py")):
        root = "data" if os.path.exists("data") else "/tmp/cifar10"
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=T.ToTensor())
    n = min(n, len(ds))
    xs = [ds[i][0] for i in range(n)]
    return torch.stack(xs).to(device=device, dtype=torch.float32)


def parse_scale_set(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main() -> None:
    args = parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[setup] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(device)}", flush=True)
    x0 = load_cifar10(args.n_samples, str(device))
    dense_precomp = dense_wiener_precompute(x0)
    sigmas = np.asarray(args.sigmas, dtype=np.float64)
    scale_sets = [parse_scale_set(s) for s in args.scale_sets]
    dense = np.asarray([dense_wiener_loss_from_precomp(dense_precomp, float(s), lam=args.dense_lam) for s in sigmas])

    shape = (len(scale_sets), len(args.filters_per_scale), len(args.seeds), len(sigmas))
    loss = np.full(shape, np.nan)
    elapsed = np.full(shape, np.nan)

    for ai, scales in enumerate(scale_sets):
        for fi, filters in enumerate(args.filters_per_scale):
            for ei, seed in enumerate(args.seeds):
                weights = []
                biases = []
                for si, scale in enumerate(scales):
                    w, b = make_random_conv_filters(
                        filters,
                        3,
                        args.kernel_size,
                        seed=seed + 1000 * scale + 17 * si,
                        device=device,
                    )
                    weights.append(w)
                    biases.append(b)
                for si, sigma in enumerate(sigmas):
                    print(f"[run] scales={scales} F/scale={filters} seed={seed} sigma={sigma:.4g}", flush=True)
                    stats = accumulate_multiscale_dense_wiener_residual_stats(
                        x0,
                        float(sigma),
                        weights,
                        biases,
                        scales,
                        dense_precomp,
                        n_noise=args.n_noise,
                        batch_size=args.batch_size,
                        dense_lam=args.dense_lam,
                        show_progress=True,
                    )
                    res = mmse_from_conv_stats(stats, lam=args.lam, bias_mode="channel")
                    loss[ai, fi, ei, si] = res["loss"]
                    elapsed[ai, fi, ei, si] = stats.elapsed_seconds
                    print(f"[done] loss={res['loss']:.4f} gain={dense[si]-res['loss']:+.4f} elapsed={stats.elapsed_seconds:.1f}s", flush=True)
                    del stats
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    save_results(args, sigmas, scale_sets, dense, loss, elapsed)

    save_results(args, sigmas, scale_sets, dense, loss, elapsed)
    plot_results(args, sigmas, scale_sets, dense, loss)


def save_results(args: argparse.Namespace, sigmas: np.ndarray, scale_sets: list[list[int]], dense: np.ndarray, loss: np.ndarray, elapsed: np.ndarray) -> None:
    path = os.path.join(args.artifact_dir, f"multiscale_conv_rf_cifar10_{args.tag}.npz")
    np.savez(
        path,
        sigma=sigmas,
        scale_sets=np.asarray([",".join(map(str, s)) for s in scale_sets]),
        filters_per_scale=np.asarray(args.filters_per_scale),
        seeds=np.asarray(args.seeds),
        dense_wiener=dense,
        multiscale_dense_wiener_plus_conv_rf=loss,
        elapsed_seconds=elapsed,
        kernel_size=args.kernel_size,
        n_samples=args.n_samples,
        n_noise=args.n_noise,
    )


def plot_results(args: argparse.Namespace, sigmas: np.ndarray, scale_sets: list[list[int]], dense: np.ndarray, loss: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"Multi-scale conv RF residual | kernel={args.kernel_size}, tag={args.tag}", fontsize=12)
    ax = axes[0]
    ax.plot(sigmas, dense, color="black", marker="o", lw=2, label="dense Wiener")
    for ai, scales in enumerate(scale_sets):
        for fi, filters in enumerate(args.filters_per_scale):
            vals = np.nanmean(loss[ai, fi], axis=0)
            ax.plot(sigmas, vals, marker="o", lw=1.7, label=f"scales={scales}, F/scale={filters}")
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("MSE")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for ai, scales in enumerate(scale_sets):
        for fi, filters in enumerate(args.filters_per_scale):
            vals = np.nanmean(loss[ai, fi], axis=0)
            ax.plot(sigmas, dense - vals, marker="o", lw=1.7, label=f"scales={scales}, F/scale={filters}")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("dense Wiener - multiscale RF")
    ax.set_title("Positive means RF residual helps")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(args.figure_dir, f"multiscale_conv_rf_cifar10_{args.tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] saved {out}", flush=True)


if __name__ == "__main__":
    main()
