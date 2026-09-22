"""
CIFAR-10 local empirical Bayes denoising with patch GMMs.

The clean patch prior is a full-covariance Gaussian mixture. Denoising uses the
exact posterior mean E[x_center | noisy_patch] for every image location.
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

from core.conv_rf_mmse import dense_wiener_loss_from_precomp, dense_wiener_precompute
from core.patch_gmm import fit_patch_gmm, make_patch_gmm, patch_gmm_mse, sample_clean_patches


STORE = os.environ.get(
    "STORE_DIR",
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_train_images", type=int, default=5000)
    p.add_argument("--n_eval_images", type=int, default=1000)
    p.add_argument("--n_patches", type=int, default=30000)
    p.add_argument("--patch_sizes", type=int, nargs="+", default=[3, 5])
    p.add_argument("--components", type=int, nargs="+", default=[32, 64])
    p.add_argument("--em_iter", type=int, default=12)
    p.add_argument("--reg_covar", type=float, default=1e-4)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.45, 0.65, 0.85, 1.2, 1.7, 2.2])
    p.add_argument("--n_noise", type=int, default=1)
    p.add_argument("--image_batch_size", type=int, default=64)
    p.add_argument("--batch_patches", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--artifact_dir", default=os.path.join("tables", "patch_gmm_cifar10"))
    p.add_argument("--figure_dir", default="figures")
    p.add_argument("--tag", default="pilot")
    return p.parse_args()


def load_cifar10(train: bool, n: int, device: str) -> torch.Tensor:
    root = os.path.join(STORE, "Datasets")
    if not os.path.exists(os.path.join(root, "cifar-10-batches-py")):
        root = "data" if os.path.exists("data") else "/tmp/cifar10"
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=False, transform=T.ToTensor())
    n = min(n, len(ds))
    xs = []
    for i in range(n):
        x, _ = ds[i]
        xs.append(x)
    return torch.stack(xs).to(device=device, dtype=torch.float32)


def save_gmm(path: str, weights: torch.Tensor, means: torch.Tensor, covs: torch.Tensor, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        weights=weights.detach().cpu().numpy(),
        means=means.detach().cpu().numpy(),
        covs=covs.detach().cpu().numpy(),
        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()},
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[setup] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(device)}", flush=True)

    x_train = load_cifar10(True, args.n_train_images, str(device))
    x_eval = load_cifar10(False, args.n_eval_images, str(device))
    print(f"[data] train={tuple(x_train.shape)} eval={tuple(x_eval.shape)}", flush=True)

    dense_precomp = dense_wiener_precompute(x_train)
    sigmas = np.asarray(args.sigmas, dtype=np.float64)
    dense = np.asarray([dense_wiener_loss_from_precomp(dense_precomp, float(s)) for s in sigmas])
    shape = (len(args.patch_sizes), len(args.components), len(sigmas))
    losses = np.full(shape, np.nan)
    elapsed = np.full(shape, np.nan)

    for pi, patch_size in enumerate(args.patch_sizes):
        patches = sample_clean_patches(
            x_train,
            patch_size,
            args.n_patches,
            batch_size=args.image_batch_size,
            seed=args.seed + 1000 * patch_size,
        )
        print(f"[patches] patch={patch_size} samples={tuple(patches.shape)}", flush=True)
        for ci, n_comp in enumerate(args.components):
            print(f"[fit] patch={patch_size} components={n_comp}", flush=True)
            weights, means, covs = fit_patch_gmm(
                patches,
                n_comp,
                n_iter=args.em_iter,
                reg_covar=args.reg_covar,
                seed=args.seed + n_comp,
                verbose=True,
            )
            gmm = make_patch_gmm(
                weights,
                means,
                covs,
                n_channels=3,
                patch_size=patch_size,
                reg_covar=args.reg_covar,
            )
            gmm_path = os.path.join(args.artifact_dir, f"patch_gmm_{args.tag}_p{patch_size}_k{n_comp}.npz")
            save_gmm(gmm_path, weights, means, covs, {
                "patch_size": patch_size,
                "components": n_comp,
                "n_patches": args.n_patches,
                "reg_covar": args.reg_covar,
            })
            print(f"[fit] saved {gmm_path}", flush=True)
            for si, sigma in enumerate(sigmas):
                print(f"[eval] patch={patch_size} components={n_comp} sigma={sigma:.4g}", flush=True)
                out = patch_gmm_mse(
                    x_eval,
                    gmm,
                    float(sigma),
                    n_noise=args.n_noise,
                    image_batch_size=args.image_batch_size,
                    batch_patches=args.batch_patches,
                    show_progress=False,
                )
                losses[pi, ci, si] = out["loss"]
                elapsed[pi, ci, si] = out["elapsed_seconds"]
                print(f"[eval] loss={out['loss']:.4f} elapsed={out['elapsed_seconds']:.1f}s", flush=True)
                save_results(args, sigmas, dense, losses, elapsed)

    save_results(args, sigmas, dense, losses, elapsed)
    plot_results(args, sigmas, dense, losses)


def save_results(args: argparse.Namespace, sigmas: np.ndarray, dense: np.ndarray, losses: np.ndarray, elapsed: np.ndarray) -> None:
    path = os.path.join(args.artifact_dir, f"patch_gmm_cifar10_{args.tag}.npz")
    np.savez(
        path,
        sigma=sigmas,
        patch_sizes=np.asarray(args.patch_sizes),
        components=np.asarray(args.components),
        dense_wiener=dense,
        patch_gmm=losses,
        elapsed_seconds=elapsed,
        n_train_images=args.n_train_images,
        n_eval_images=args.n_eval_images,
        n_patches=args.n_patches,
        n_noise=args.n_noise,
    )


def plot_results(args: argparse.Namespace, sigmas: np.ndarray, dense: np.ndarray, losses: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"CIFAR-10 Patch-GMM empirical Bayes | tag={args.tag}", fontsize=12)
    ax = axes[0]
    ax.plot(sigmas, dense, color="black", marker="o", lw=2, label="dense Wiener")
    for pi, patch_size in enumerate(args.patch_sizes):
        for ci, n_comp in enumerate(args.components):
            ax.plot(sigmas, losses[pi, ci], marker="o", lw=1.7, label=f"patch {patch_size}x{patch_size}, K={n_comp}")
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("MSE")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for pi, patch_size in enumerate(args.patch_sizes):
        for ci, n_comp in enumerate(args.components):
            ax.plot(sigmas, dense - losses[pi, ci], marker="o", lw=1.7, label=f"patch {patch_size}x{patch_size}, K={n_comp}")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("dense Wiener - patch GMM")
    ax.set_title("Positive means patch GMM better")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(args.figure_dir, f"patch_gmm_cifar10_{args.tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] saved {out}", flush=True)


if __name__ == "__main__":
    main()
