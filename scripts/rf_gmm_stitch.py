"""
rf_gmm_stitch.py — one coherent multi-panel figure for the finite-N_train GMM sweep.

Naively pasting the four per-N_train PNGs side by side repeats the axis labels, the
titles and the legend four times, so everything shrinks and nothing is readable. Instead
this rebuilds a SINGLE figure from the stored tables:

    rows    = sigma                      (row label on the left)
    columns = N_train x {uncond, cond}   (bold header on the top row)
LANDSCAPE aspect, so it drops into slides/papers without rotating.

with one shared legend for the whole figure, x-labels only on the bottom row and y-labels
only on the left column. Nothing is recomputed — it reads tables/rf_gmm_finite_sample/d{D}/.

    D=8  python scripts/rf_gmm_stitch.py
    D=32 N_TRAIN=16,64,256,1024 python scripts/rf_gmm_stitch.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D            = int(os.environ.get('D', '8'))
N_CLASSES    = 3
N_TRAIN      = [int(x) for x in os.environ.get('N_TRAIN', '16,64,256,1024').split(',')]
SIGMA_VALUES = [float(s) for s in os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]
TBL          = f'tables/rf_gmm_finite_sample/d{D}'
OUT          = f'figures/rf_gmm_finite_sample/d{D}/ALL_N.png'


def load(N):
    d = np.load(os.path.join(TBL, f'N{N}.npz'), allow_pickle=True)
    keys = set(d.files)
    res, emp, pop = {}, {}, {}
    for sg in SIGMA_VALUES:
        suf = f'_s{sg}'
        R = {k[:-len(suf)]: d[k] for k in keys if k.endswith(suf)}
        res[sg] = R
        emp[sg] = {k: (float(R[k]) if k in R else None)
                   for k in ('wiener_emp', 'wiener_cond_emp', 'nw_bayes', 'nw_bayes_cond')}
        pop[sg] = {k: float(R[k]) for k in ('bayes_pop', 'wiener_pop', 'cond_wiener_pop')}
    return d['k_grid'], res, emp, pop, float(d['trace_p0_emp'])


def main():
    """Landscape layout: sigma down the rows, (N_train x cond/uncond) across the columns.
    This is the transpose of the natural per-N figure and gives a wide, slide-friendly
    aspect ratio while still showing every panel exactly once."""
    cols = [(N, uc) for N in N_TRAIN for uc in ('u', 'c')]
    nC, nR = len(cols), len(SIGMA_VALUES)
    fig, axes = plt.subplots(nR, nC, figsize=(2.55 * nC, 3.05 * nR),
                             squeeze=False, sharex=True)
    fig.suptitle(f'RF denoiser on a finite dataset — GMM d={D}, C={N_CLASSES}   '
                 'RF in the green band beats the best linear denoiser   '
                 '(conditional columns use conditional baselines)',
                 fontsize=15, y=0.982)

    cache = {N: load(N) for N in N_TRAIN}
    meta = {'u': ('nw_bayes', 'wiener_pop', 'wiener_emp', 'bayes_pop'),
            'c': ('nw_bayes_cond', 'cond_wiener_pop', 'wiener_cond_emp', 'cond_wiener_pop')}
    handles = {}
    for c, (N, uc) in enumerate(cols):
        kg, res, emp, pop, tr = cache[N]
        kd = np.array(kg) / D
        nw_k, popw_k, lin_k, popb_k = meta[uc]
        for r, sg in enumerate(SIGMA_VALUES):
            ax = axes[r][c]
            R, eb, pb = res[sg], emp[sg], pop[sg]
            lin = eb[lin_k]

            if eb[nw_k] is not None and lin > eb[nw_k]:
                h = ax.axhspan(eb[nw_k], lin, color='green', alpha=.07, zorder=0)
                handles.setdefault('nonlinear-gain zone', h)
            # baselines are column-matched (cond columns use the conditional versions),
            # so each style gets ONE legend entry rather than a near-duplicate pair.
            h, = ax.plot(kd, np.full_like(kd, pb[popb_k]), color='gray', ls='--', lw=1.1)
            handles.setdefault('Bayes MMSE (pop)', h)
            if popb_k != popw_k:
                h, = ax.plot(kd, np.full_like(kd, pb[popw_k]), color='gray', ls=':', lw=1.1)
                handles.setdefault('linear Wiener (pop)', h)
            h, = ax.plot(kd, np.full_like(kd, lin), color='darkorange', ls='-', lw=1.5)
            handles.setdefault('linear Wiener (emp)', h)
            if eb[nw_k] is not None:
                h, = ax.plot(kd, np.full_like(kd, eb[nw_k]), color='forestgreen', lw=1.8)
                handles.setdefault('Bayes MMSE (emp)', h)

            h, = ax.plot(kd, R[f'gmm_pop_{uc}'], color='crimson', lw=1.9, ls='--')
            handles.setdefault('RF theory (pop, N→∞)', h)
            h, = ax.plot(kd, R[f'stein_{uc}'], color='teal', lw=1.7, ls='-.')
            handles.setdefault('RF theory (emp)', h)
            h, = ax.plot(kd, R[f'rf_analytic_{uc}'], color='steelblue', lw=2.1,
                         marker='o', ms=3.2)
            handles.setdefault('RF measured (emp)', h)
            h, = ax.plot(kd, R[f'rf_optridge_{uc}'], color='slategray', lw=1.0, ls=':',
                         marker='.', ms=2.4, alpha=.8)
            handles.setdefault('RF measured (pure-MC opt-λ)', h)

            ax.set_xscale('log'); ax.set_ylim(bottom=-0.01)
            ax.grid(True, alpha=.3); ax.tick_params(labelsize=8.5)
            if r == 0:
                ax.set_title(f"N={N}\n{'Uncond.' if uc == 'u' else 'Cond. (U)'}",
                             fontsize=12.5, fontweight='bold', pad=5)
            if r == nR - 1:
                ax.set_xlabel('k / d', fontsize=11)
            if c == 0:
                ax.set_ylabel('MSE', fontsize=10.5)
                ax.text(-0.38, 0.5, f'σ = {sg}', transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=14, fontweight='bold')

    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=8,
               fontsize=11.5, frameon=True, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0.014, 0.045, 1, 0.972])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f"Saved {OUT}   ({nR} rows x {nC} cols, landscape)")


if __name__ == '__main__':
    main()
