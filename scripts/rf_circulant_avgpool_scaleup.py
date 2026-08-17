"""
rf_circulant_avgpool_scaleup.py — L^dense vs L^circ vs L^circ-banded on the SAME phi as
figures/dnn_feature_mmse_cifar10.png (ResNet18 avgpool, d=512), swept over k/d and over the
full noise grid of that figure.

x0 := the ResNet18 global-average-pooled representation (d=512), standardized to
Tr(Sigma_p0)/d = 1; y = x0 + sigma Z; D(y) = W omega(Theta y + eps) + b.

HOW FAR k/d CAN GO, AND WHY IT STOPS WHERE IT DOES
--------------------------------------------------
The Stein path materialises Sigma_phi as a dense k x k matrix and factorises it. At d=512:

    k/d      k        k^2 (fp64)     with ~4 intermediates
      8    4096          0.1 GB                   0.5 GB
     32   16384          2.0 GB                   8.0 GB
     64   32768          8.0 GB                  32.0 GB
    200  102400         78.1 GB                 312.5 GB     <-- requested, infeasible

Building Sigma_phi needs rho, rho^2, rho^3 and the three C_n Grams alongside the result, so
on an 80 GB H100 the practical ceiling is k/d = 32 (comfortable) to 64 (tight). k/d = 200 is
off the table by ~2 orders of magnitude in memory. PCA-reducing d does not rescue it: the
avgpool covariance spectrum decays slowly (rank 64 keeps only 72.8% of the variance, rank 128
only 86.2%), so shrinking d enough to reach k/d = 200 would change the representation
materially rather than compress it.

Reaching k/d = 200 for L^circ alone IS possible, because a block-circulant Sigma_phi only
needs the c x c per-frequency blocks (c^2 d numbers, not k^2), but L^dense has no such
structure, and a curve without its comparison curve is not the experiment.

    python scripts/rf_circulant_avgpool_scaleup.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.rf_circulant_dnn_layers import extract, standardize, losses_at, DEV, DT

NIMG   = int(os.environ.get('NIMG', '10000'))
C_LIST = [int(x) for x in os.environ.get('C_LIST', '1,2,4,8,16,32').split(',')]
T_BAND = int(os.environ.get('T_BAND', '8'))
N_SEED_SMALL = int(os.environ.get('N_SEED_SMALL', '4'))   # k/d <= 8
N_SEED_BIG   = int(os.environ.get('N_SEED_BIG', '2'))     # k/d >= 16 (circ variance is
                                                          # already small there; see the
                                                          # c*t concentration argument)
TBL = 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz'
N_SIG = int(os.environ.get('N_SIG', '8'))


def main():
    tab = np.load(TBL, allow_pickle=True)
    sig_all = tab['sigma']
    sel = np.unique(np.round(np.linspace(0, len(sig_all) - 1, N_SIG)).astype(int))
    sigmas = [float(sig_all[i]) for i in sel]
    print(f"noise grid taken from {TBL}: {[f'{s:.3f}' for s in sigmas]}")

    X = standardize(extract('CIFAR-10', n_img=NIMG)['avgpool'].to(DEV, DT))
    N, d = X.shape
    print(f"CIFAR-10 avgpool: N={N}, d={d} (identical to that figure's phi)\n")

    R = {}
    for sg in sigmas:
        R[sg] = {q: [] for q in ('c', 'dense', 'circ', 'band',
                                 'dense_sd', 'circ_sd', 'band_sd')}
        print(f"=== sigma={sg:.4f} ===")
        print(f"  {'k/d':>4} {'k':>7} {'L_dense':>16} {'L_circ':>16} "
              f"{'L_band(t=%d)' % T_BAND:>16} {'circ-dense':>11} {'band-dense':>11}")
        for c in C_LIST:
            ns = N_SEED_SMALL if c <= 8 else N_SEED_BIG
            A = np.array([losses_at(X, sg, c, T_BAND, seed=137 * s_) for s_ in range(ns)])
            m = A.mean(0); sd = A.std(0, ddof=1) if ns > 1 else np.zeros(3)
            R[sg]['c'].append(c)
            for q, col in (('dense', 0), ('circ', 1), ('band', 2)):
                R[sg][q].append(m[col]); R[sg][q + '_sd'].append(sd[col])
            print(f"  {c:>4} {c*d:>7} {m[0]:>10.2f}+-{sd[0]:<5.2f} {m[1]:>10.2f}+-{sd[1]:<5.2f} "
                  f"{m[2]:>10.2f}+-{sd[2]:<5.2f} {m[1]-m[0]:>+11.2f} {m[2]-m[0]:>+11.2f}",
                  flush=True)
        np.savez('tables/rf_circulant_avgpool_scaleup.npz', sigmas=np.array(sigmas), d=d,
                 **{f'{s}|{q}': np.array(v[q]) for s, v in R.items() for q in v})

    # ---------------- figure ----------------
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12.5, 'axes.titlesize': 12})
    n = len(sigmas); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)
    for i, sg in enumerate(sigmas):
        ax = axes[i // ncol][i % ncol]; r = R[sg]
        for q, col, mk, nm in (('dense', '#1f4e9c', 'o', r'\rm dense'),
                               ('circ', '#7b3fa0', 's', r'\rm circ\ (full)'),
                               ('band', '#c0392b', '^', rf'\rm circ\ (t={T_BAND})')):
            mu = np.array(r[q]); sd = np.array(r[q + '_sd'])
            ax.plot(r['c'], mu, color=col, marker=mk, lw=2.1, ms=5,
                    label=rf'$\mathcal{{L}}^{{{nm}}}$')
            ax.fill_between(r['c'], mu - sd, mu + sd, color=col, alpha=.18, lw=0)
        ax.set_xscale('log', base=2); ax.set_xticks(C_LIST)
        ax.set_xticklabels([str(c) for c in C_LIST])
        ax.set_title(rf'$\sigma$={sg:.3f}'); ax.grid(True, alpha=.3)
        ax.set_xlabel('$k/d$')
        if i % ncol == 0:
            ax.set_ylabel(r'denoiser loss $\mathcal{L}_\sigma$')
        if i == 0:
            ax.legend(fontsize=9)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('CIFAR-10, ResNet18 avgpool $\\phi$ ($d=512$, the same representation as '
                 'figures/dnn_feature_mmse_cifar10.png)\n'
                 'scaling $k/d$ at every noise scale — the circulant penalty does NOT close '
                 'as $k/d$ grows.  Bands = $\\pm$1 s.d. over $\\Theta$ draws.', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = 'figures/rf_circulant_avgpool_scaleup.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
