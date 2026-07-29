"""
rf_gmm_stitch.py — one coherent multi-panel figure for the finite-N_train GMM sweep.

Naively pasting the four per-N_train PNGs side by side repeats the axis labels, the
titles and the legend four times, so everything shrinks and nothing is readable. Instead
this rebuilds a SINGLE figure from the stored tables:

    columns = sigma            (one column per noise level, header on the top row only)
    rows    = N_train x {uncond, cond}   (row label on the left, e.g. "N=64 / Conditional")

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
    nS = len(SIGMA_VALUES)
    nR = 2 * len(N_TRAIN)
    fig, axes = plt.subplots(nR, nS, figsize=(4.1 * nS, 2.85 * nR),
                             squeeze=False, sharex=True)
    fig.suptitle(f'RF denoiser on a finite dataset — GMM d={D}, C={N_CLASSES}\n'
                 'RF in the green band beats the best linear denoiser   '
                 '(baselines are row-matched: conditional rows use conditional baselines)',
                 fontsize=15, y=0.997)

    handles = {}
    for bi, N in enumerate(N_TRAIN):
        kg, res, emp, pop, tr = load(N)
        kd = np.array(kg) / D
        rows = [('Unconditional', 'u', 'nw_bayes', 'wiener_pop', 'wiener_emp', 'bayes_pop'),
                ('Conditional (U=class)', 'c', 'nw_bayes_cond', 'cond_wiener_pop',
                 'wiener_cond_emp', 'cond_wiener_pop')]
        for ri, (rname, uc, nw_k, popw_k, lin_k, popb_k) in enumerate(rows):
            r = 2 * bi + ri
            for c, sg in enumerate(SIGMA_VALUES):
                ax = axes[r][c]
                R, eb, pb = res[sg], emp[sg], pop[sg]
                lin = eb[lin_k]

                if eb[nw_k] is not None and lin > eb[nw_k]:
                    h = ax.axhspan(eb[nw_k], lin, color='green', alpha=.07, zorder=0)
                    handles.setdefault('nonlinear-gain zone', h)
                # baselines are ROW-MATCHED (cond rows use the conditional versions),
                # so each style gets ONE legend entry rather than a near-duplicate pair.
                h, = ax.plot(kd, np.full_like(kd, pb[popb_k]), color='gray', ls='--', lw=1.1)
                handles.setdefault('Bayes MMSE (pop)', h)
                if popb_k != popw_k:      # uncond row: Wiener(pop) is a separate line
                    h, = ax.plot(kd, np.full_like(kd, pb[popw_k]), color='gray', ls=':', lw=1.1)
                    handles.setdefault('linear Wiener (pop)', h)
                h, = ax.plot(kd, np.full_like(kd, lin), color='darkorange', ls='-', lw=1.5)
                handles.setdefault('linear Wiener (emp)', h)
                if eb[nw_k] is not None:
                    h, = ax.plot(kd, np.full_like(kd, eb[nw_k]), color='forestgreen', lw=1.8)
                    handles.setdefault('Bayes MMSE (emp)', h)

                h, = ax.plot(kd, R[f'gmm_pop_{uc}'], color='crimson', lw=2, ls='--')
                handles.setdefault('RF theory (pop, N→∞)', h)
                h, = ax.plot(kd, R[f'stein_{uc}'], color='teal', lw=1.8, ls='-.')
                handles.setdefault('RF theory (emp)', h)
                h, = ax.plot(kd, R[f'rf_analytic_{uc}'], color='steelblue', lw=2.2,
                             marker='o', ms=3.5)
                handles.setdefault('RF measured (emp)', h)
                h, = ax.plot(kd, R[f'rf_optridge_{uc}'], color='slategray', lw=1.0, ls=':',
                             marker='.', ms=2.5, alpha=.8)
                handles.setdefault('RF measured (pure-MC opt-λ)', h)

                ax.set_xscale('log')
                ax.set_ylim(bottom=-0.01)
                ax.grid(True, alpha=.3)
                ax.tick_params(labelsize=9)
                if r == 0:
                    ax.set_title(f'σ = {sg}', fontsize=14, pad=8)
                if r == nR - 1:
                    ax.set_xlabel('k / d', fontsize=12)
                if c == 0:
                    ax.set_ylabel('MSE', fontsize=11)
                    ax.text(-0.30, 0.5, f'N={N}\n{"Uncond." if uc == "u" else "Cond. (U)"}',
                            transform=ax.transAxes, rotation=90, va='center', ha='center',
                            fontsize=12, fontweight='bold')

    fig.legend(handles.values(), handles.keys(), loc='lower center',
               ncol=4, fontsize=11.5, frameon=True, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=[0.012, 0.035, 1, 0.975])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f"Saved {OUT}   ({nR} rows x {nS} cols, N_train={N_TRAIN})")


if __name__ == '__main__':
    main()
