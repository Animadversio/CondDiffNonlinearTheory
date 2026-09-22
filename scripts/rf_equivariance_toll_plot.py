"""Figure: the circulant model's deficit is the equivariance toll, and tiling is far worse.

Left   loss vs sigma for the nested classes: free Wiener < circulant RF < best EQUIVARIANT
       linear < best TILED nonlinear (Bayes floor).  The circulant RF sits essentially on
       the equivariant-linear curve, so its whole deficit vs linear is the readout
       constraint -- and the proposed tiled model's FLOOR is an order of magnitude worse.
Right  payoff for relaxing equivariance to period P.  Flat where it is affordable.

    python scripts/rf_equivariance_toll_plot.py      # needs tables/rf_equivariance_toll.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figures/rf_equivariance_toll.png'
PS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]   # P=d is the free Wiener, toll = 0 exactly
                                                     # (omitted: log axis cannot show it)


def main():
    z = np.load('tables/rf_equivariance_toll.npz')
    sg, lin = z['sigmas'], z['linear']
    eq, rf, tb = z['period1d|1'], z['circ_rf'], z['tiled_bayes|8']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    ax.plot(sg, tb - lin, 's--', color='#b2182b', lw=2, ms=8,
            label='BAYES floor of the tiled 8-block model\n(best possible, any nonlinearity)')
    ax.plot(sg, eq - lin, 'o-', color='#2166ac', lw=2.2, ms=8,
            label='best EQUIVARIANT linear denoiser\n(exact min over the whole class)')
    ax.plot(sg, rf - lin, 'D-', color='#1a9850', lw=2.2, ms=8,
            label='block-circulant RF, matched free params\n($c$=6144, 18.9M features)')
    ax.axhline(0, color='k', lw=1.2)
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('$\\sigma$'); ax.set_ylabel('$\\mathcal{L}-\\mathcal{L}^{\\rm lin}$   (excess over the free Wiener denoiser)')
    ax.set_title('The circulant RF sits ON the equivariant-linear curve\n'
                 'raw CIFAR-10, $d$=3072; 0 = plain linear regression', fontsize=10)
    ax.legend(fontsize=8.2, loc='upper left'); ax.grid(alpha=0.25, which='both')
    for i, s in enumerate(sg):
        ax.annotate(f'{rf[i]-eq[i]:+.2f}', (s, rf[i] - lin[i]), textcoords='offset points',
                    xytext=(0, -16), ha='center', fontsize=7.5, color='#1a9850')
    ax.text(0.97, 0.05, 'green labels = what the nonlinearity buys\nover the best equivariant LINEAR map',
            transform=ax.transAxes, ha='right', fontsize=8,
            bbox=dict(fc='w', ec='0.6', alpha=0.9))

    cols = plt.cm.viridis(np.linspace(0, 0.85, len(sg)))
    for i, s in enumerate(sg):
        ax2.plot(PS, [z[f'period1d|{P}'][i] - lin[i] for P in PS], 'o-',
                 color=cols[i], lw=2, ms=6, label=f'$\\sigma$={s}')
    ax2.set_xscale('log', base=2); ax2.set_yscale('log')
    ax2.set_xlabel('$P$   (readout equivariant only up to period $P$;  $W$ params $=c\\,d\\,P$)')
    ax2.set_ylabel('residual toll   $\\mathcal{L}^{(P)}-\\mathcal{L}^{\\rm lin}$')
    ax2.set_title('Relaxing equivariance: no cheap interior point\n'
                  'flat to $P$=8, collapses only at $P$=1024=$d$/3', fontsize=10)
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.25, which='both')
    ax2.axvline(1024, color='0.4', ls=':', lw=1.4)
    ax2.text(1024, ax2.get_ylim()[1] * 0.5, ' $P$=$d$/3: filter may\n depend on position',
             fontsize=7.5, color='0.3', va='top')
    ax2.text(0.32, 0.03, 'not shown: $P$=$d$=3072 is the free\nWiener denoiser, toll $\\equiv$ 0',
             transform=ax2.transAxes, fontsize=7.8, color='0.3')

    os.makedirs('figures', exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=160)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
