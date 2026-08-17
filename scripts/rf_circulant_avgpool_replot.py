"""
rf_circulant_avgpool_replot.py — replot tables/rf_circulant_avgpool_scaleup.npz with
sigma capped at 10 and the EDM denoiser added as a dimensionless reference.

WHY EDM NEEDS A CONVERSION RATHER THAN A DIRECT OVERLAY
-------------------------------------------------------
edm_uncond in tables/dnn_feature_mmse_cifar10_*.npz denoises RAW PIXELS:
    d = 3072,  Tr(Sigma) = 191.52,  per-coordinate SD = 0.2497.
Our L^dense / L^circ curves denoise the ResNet18 AVGPOOL representation:
    d = 512,   Tr(Sigma) = 512 (standardized),  per-coordinate SD = 1.0.
So a given numeric sigma is a 4x different noise-to-signal ratio on the two curves, and the
losses saturate at different values (191.52 vs 512). Plotting them on shared axes without
conversion would be wrong on BOTH axes.

We therefore compare the only thing that is comparable: the dimensionless
    fraction of variance unexplained,  L / Tr(Sigma),  at matched  sigma / per-coordinate SD.
For a panel at our sigma_ours, the equivalent pixel-domain noise is
sigma_pix = sigma_ours * 0.2497; we read edm_uncond there and rescale by our Tr(Sigma) so it
can be drawn on the same axis. The EDM line is a REFERENCE for that dimensionless quantity,
not a competitor on the same task -- it is denoising images, we are denoising an encoding.

    python scripts/rf_circulant_avgpool_replot.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIG_MAX = float(os.environ.get('SIG_MAX', '10.0'))
T_BAND = 8
SRC = 'tables/rf_circulant_avgpool_scaleup.npz'
TBL = 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz'


def main():
    z = np.load(SRC)
    sigmas = [s for s in z['sigmas'] if s <= SIG_MAX]
    d = int(z['d'])
    print(f"kept {len(sigmas)}/{len(z['sigmas'])} noise scales (sigma <= {SIG_MAX}): "
          f"{[f'{s:.3f}' for s in sigmas]}")

    tab = np.load(TBL, allow_pickle=True)
    sig_pix, edm = tab['sigma'], tab['edm_uncond']
    tr_pix = 191.52                      # Tr(Sigma) of raw CIFAR pixels, [0,1] units
    sd_pix = np.sqrt(tr_pix / 3072)      # per-coordinate SD = 0.2497
    tr_ours = float(d)                   # standardized: Tr(Sigma)/d = 1

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12.5, 'axes.titlesize': 12})
    n = len(sigmas); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    for i, sg in enumerate(sigmas):
        ax = axes[i // ncol][i % ncol]
        c = z[f'{sg}|c']
        for q, col, mk, nm in (('dense', '#1f4e9c', 'o', r'\rm dense'),
                               ('circ', '#7b3fa0', 's', r'\rm circ\ (full)'),
                               ('band', '#c0392b', '^', rf'\rm circ\ (t={T_BAND})')):
            mu, sd = z[f'{sg}|{q}'], z[f'{sg}|{q}_sd']
            ax.plot(c, mu, color=col, marker=mk, lw=2.1, ms=5,
                    label=rf'$\mathcal{{L}}^{{{nm}}}$')
            ax.fill_between(c, mu - sd, mu + sd, color=col, alpha=.18, lw=0)
        # EDM, converted to the same dimensionless footing
        sg_pix = sg * sd_pix
        e = float(np.interp(sg_pix, sig_pix, edm))
        ax.axhline(e / tr_pix * tr_ours, color='seagreen', ls='--', lw=2.0,
                   label='EDM (same $L/\\mathrm{Tr}\\Sigma$,\n'
                         rf'  at $\sigma_{{\rm pix}}$={sg_pix:.3f})')
        ax.set_xscale('log', base=2); ax.set_xticks(c)
        ax.set_xticklabels([str(int(v)) for v in c])
        ax.set_title(rf'$\sigma$={sg:.3f}   ($\sigma/\mathrm{{SD}}$={sg:.2f})')
        ax.grid(True, alpha=.3); ax.set_xlabel('$k/d$')
        if i % ncol == 0:
            ax.set_ylabel(r'denoiser loss $\mathcal{L}_\sigma$')
        if i == 0:
            ax.legend(fontsize=8.5)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('CIFAR-10, ResNet18 avgpool $\\phi$ ($d=512$), $\\sigma\\leq10$ — the '
                 'circulant penalty does not close as $k/d$ grows\n'
                 'EDM dashed line is the SAME dimensionless $L/\\mathrm{Tr}\\Sigma$ but for '
                 'denoising raw pixels ($d=3072$), converted at matched $\\sigma$/SD; it is a '
                 'reference, not a competitor on this task', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = 'figures/rf_circulant_avgpool_scaleup_sig10.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
