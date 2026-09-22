"""Figure: smooth kernel variation buys far more per parameter than periodic sharing.

Left   toll vs W parameters, band-B (solid) against period-p (dashed), one colour per
       sigma.  Same family, same budget axis -- the only difference is WHICH modulation
       frequencies are bought.
Right  toll vs sigma for the nested classes, against the two deficits already measured
       (the Z_3072 toll our RF pays, and the RF's own deficit).

    python scripts/rf_band_relaxation_atlas_plot.py     # needs tables/rf_band_relaxation_atlas.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figures/rf_band_relaxation_atlas.png'
BANDS, PERIODS = [0, 1, 2, 3], [1, 2, 4, 8, 16]
Z3072 = [1.273, 4.070, 8.473, 9.005]      # our RF's group, from rf_equivariance_toll.npz
CIRC_RF = [0.129, 2.927, 8.315, 8.974]    # the block-circulant RF's actual deficit


def main():
    z = np.load('tables/rf_band_relaxation_atlas.npz')
    sg, lin = z['sigmas'], z['linear']
    bt = {b: z[f'band|{b}'] - lin for b in BANDS}
    pt = {p: z[f'period|{p}'] - lin for p in PERIODS}
    bn = {b: int(z[f'band|{b}|np'][0]) for b in BANDS}
    pn = {p: int(z[f'period|{p}|np'][0]) for p in PERIODS}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3))
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(sg)))

    for i, s in enumerate(sg):
        ax.plot([bn[b] for b in BANDS], [bt[b][i] for b in BANDS], 'o-',
                color=cols[i], lw=2.3, ms=7, label=f'$\\sigma$={s}  smooth (band $B$)')
        ax.plot([pn[p] for p in PERIODS], [pt[p][i] for p in PERIODS], 's--',
                color=cols[i], lw=1.8, ms=6, alpha=0.75, label=f'$\\sigma$={s}  periodic (period $p$)')
    for b in BANDS:
        ax.annotate(f'$B$={b}', (bn[b], bt[b][-1]), textcoords='offset points',
                    xytext=(3, -13), fontsize=7.5, color=cols[-1])
    for p in PERIODS:
        ax.annotate(f'$p$={p}', (pn[p], pt[p][-1]), textcoords='offset points',
                    xytext=(3, 5), fontsize=7.5, color=cols[-1], alpha=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('readout parameters  ($9216\\,|G|$)')
    ax.set_ylabel('toll  $\\mathcal{L}-\\mathcal{L}^{\\rm lin}$   (excess over the free Wiener denoiser)')
    ax.set_title('Same family, same budget, different $G$:\n'
                 'the LOW modulation frequencies are where the payoff is', fontsize=10)
    ax.legend(fontsize=6.9, ncol=2, loc='lower left'); ax.grid(alpha=0.25, which='both')
    ax.annotate('', xy=(bn[1], bt[1][-1]), xytext=(pn[16], pt[16][-1]),
                arrowprops=dict(arrowstyle='<->', color='#b2182b', lw=1.6))
    ax.text(3.2e5, 4.6, '$B$=1 beats $p$=16\nwith 28$\\times$ fewer\nparameters',
            fontsize=8, color='#b2182b', ha='center')

    ax2.plot(sg, Z3072, '^:', color='0.35', lw=1.8, ms=8,
             label='$Z_{3072}$ toll (the group our RF uses)')
    ax2.plot(sg, CIRC_RF, 'D-', color='#1a9850', lw=2.2, ms=8,
             label='our block-circulant RF (NONLINEAR)')
    for b, st in zip(BANDS, ['o-', 'o-', 'o-', 'o-']):
        ax2.plot(sg, [bt[b][i] for i in range(len(sg))], st, lw=2,
                 ms=6, color=plt.cm.autumn(b / 4.5),
                 label=f'band $B$={b} linear  ({bn[b]:,} params)')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('$\\sigma$'); ax2.set_ylabel('excess over the free Wiener denoiser')
    ax2.set_title('A band-limited LINEAR readout passes under our\n'
                  'nonlinear circulant RF for $\\sigma\\geq0.45$', fontsize=10)
    ax2.legend(fontsize=7.6, loc='lower right'); ax2.grid(alpha=0.25, which='both')

    os.makedirs('figures', exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=160)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
