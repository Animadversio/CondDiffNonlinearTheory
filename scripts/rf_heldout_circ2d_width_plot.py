"""Plot held-out 3x3 plain circulant-2D RF loss versus noise by width.

All curves use B=0 (the plain 2-D model), two seeds, 10k train images,
and the same 10k CIFAR-10 held-out images.

    python scripts/rf_heldout_circ2d_width_plot.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = Path("tables/rf_pixel_heldout.npz")
OUT = Path("figures/rf_heldout_circ2d_width_vs_sigma")
SIGS = ("0.127", "0.452", "0.621", "0.853", "1.172", "1.61", "2.212", "5.0")
WIDTHS = (256, 512, 1536, 3072)
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def main():
    tab = np.load(TABLE)
    x = np.array([float(s) for s in SIGS])
    rows = []

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for c, color in zip(WIDTHS, COLORS):
        seeds = []
        for s in SIGS:
            key = f"{s}|{c}|2d"
            if key not in tab:
                raise KeyError(f"missing {key} in {TABLE}")
            a = np.atleast_2d(tab[key])
            if a.shape != (2, 3):
                raise ValueError(f"{key}: expected 2 seeds x [train,resid,test], got {a.shape}")
            seeds.append(a[:, 2])
        seeds = np.stack(seeds)  # noise x seed
        mean = seeds.mean(1)
        lo, hi = seeds.min(1), seeds.max(1)
        ax.plot(x, mean, marker="o", ms=5, lw=2.1, color=color, label=f"c={c}")
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
        for s, vals, m in zip(SIGS, seeds, mean):
            rows.append((s, c, vals[0], vals[1], m, vals.std(ddof=1) / np.sqrt(2)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [str(v) for v in SIGS])
    ax.set_xlabel(r"noise $\sigma$")
    ax.set_ylabel(r"held-out loss $\mathbb{E}\|x_0-\hat{x}_0\|^2$")
    ax.set_title("Plain 3×3 circulant 2-D RF (B=0)\nCIFAR-10 held-out loss, 2 seeds")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="feature width", frameon=False)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    fig.savefig(OUT.with_suffix(".pdf"))
    with OUT.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "c", "test_seed0", "test_seed1", "test_mean", "test_sem"])
        w.writerows(rows)
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.csv')}")
    for s in SIGS:
        vals = [r[4] for r in rows if r[0] == s]
        print(f"sigma={s:>5}: " + "  ".join(f"c={c}: {v:.4f}" for c, v in zip(WIDTHS, vals)))


if __name__ == "__main__":
    main()
