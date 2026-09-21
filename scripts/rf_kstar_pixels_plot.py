"""Figure: the Prop-1 threshold on raw CIFAR-10 pixels vs the gap we actually measured.

Left   the theorem's cap on the linear->RF improvement,  gap <= 4 check_eps_w^2 * k
       (one line per sigma), with the measured dense gaps L^lin - L^dense overlaid.
       Every measured point must lie BELOW its line or the bound is violated.
Right  why the N=10^4 points sit as close to the cap as they do: at fixed k the measured
       gap collapses as the number of training images grows, so most of what looks like
       "dense beats linear" at N=10^4 is finite-sample, not a population effect.

Inputs: tables/rf_kstar_pixels.npz (defects), tables/rf_pixel_dense_sweep.npz +
        tables/rf_pixel_featmatch2.npz (N=10^4 losses),
        tables/rf_pixel_dense_nsweep_{10000,20000,40000}.npz (the N sweep).

    python scripts/rf_kstar_pixels_plot.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = 3072
NREF = 10000            # the defect estimate matched to the experiment's sample size
OUT = 'figures/rf_kstar_pixels.png'
SIGS_PLOT = [0.127, 0.452, 1.61, 5.0]
NS = [10000, 20000, 40000]


def load_measured():
    D, LIN = {}, {}
    for f in ('tables/rf_pixel_featmatch2.npz', 'tables/rf_pixel_dense_sweep.npz'):
        for key, v in np.load(f, allow_pickle=True).items():
            if key.startswith('linear'):
                LIN[float(key.split('|')[1])] = float(v); continue
            sg, j, w = key.split('|')
            if w == 'dense':
                D.setdefault(float(sg), {})[int(round(float(j) * d))] = float(np.mean(v))
    return D, LIN


def load_nsweep():
    """(sigma, k) -> {N: gap}."""
    G = {}
    for n in NS:
        f = f'tables/rf_pixel_dense_nsweep_{n}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        lin = {float(k.split('|')[1]): float(v) for k, v in z.items() if k.startswith('linear')}
        for key, v in z.items():
            if key.startswith('linear'):
                continue
            sg, j, _ = key.split('|')
            sg = float(sg); k = int(round(float(j) * d))
            G.setdefault((sg, k), {})[n] = lin[sg] - float(np.mean(v))
    return G


def main():
    z = np.load('tables/rf_kstar_pixels.npz')
    sigmas = list(z['sigmas']); chk = z[f'{NREF}|chk']
    D, LIN = load_measured(); G = load_nsweep()
    os.makedirs('figures', exist_ok=True)
    cols = {s: c for s, c in zip(SIGS_PLOT, plt.cm.viridis(np.linspace(0, 0.85, len(SIGS_PLOT))))}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.3))

    kk = np.logspace(2.5, 5.2, 50)
    for sg in SIGS_PLOT:
        c = chk[sigmas.index(sg)]
        ax.plot(kk, 4 * c * kk, '-', color=cols[sg], lw=2,
                label=f'$\\sigma$={sg}   $4\\check\\varepsilon_w^2$={4*c:.2e}')
        pts = [(k, LIN[sg] - D[sg][k]) for k in sorted(D.get(sg, {}))
               if LIN[sg] - D[sg][k] > 0]
        if pts:
            ax.plot(*zip(*pts), 'o', color=cols[sg], ms=8, mec='k', mew=0.7, zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$k$   (dense RF width)')
    ax.set_ylabel('$\\mathcal{L}^{\\rm lin}_\\sigma-\\mathcal{L}^{\\rm RF}_\\sigma$   (total, not per-coord)')
    ax.set_title('Prop. 1 cap  $\\mathrm{gap}\\leq4\\check\\varepsilon_w^2k$  vs measured dense gap\n'
                 'raw CIFAR-10 pixels, $d$=3072; points = $N$=10$^4$ dense sweep', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.25, which='both')
    ax.text(0.97, 0.06, 'lines = forbidden above\npoints use 16–24% of the budget',
            transform=ax.transAxes, ha='right', fontsize=8.5,
            bbox=dict(fc='w', ec='0.6', alpha=0.9))

    for (sg, k), g in sorted(G.items()):
        if sg not in cols or len(g) < 2:
            continue
        ns = sorted(g)
        ax2.plot(ns, [g[n] for n in ns], 'o-', color=cols[sg], lw=2, ms=7,
                 ls='-' if k == 12288 else '--',
                 label=f'$\\sigma$={sg}, $k/d$={k//d}')
    ax2.axhline(0, color='k', lw=1.1)
    ax2.set_xscale('log'); ax2.set_xticks(NS)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel('$N$ training images (at fixed $k$)')
    ax2.set_ylabel('$\\mathcal{L}^{\\rm lin}_\\sigma-\\mathcal{L}^{\\rm RF}_\\sigma$')
    ax2.set_title('The measured gap is mostly finite-$N$\n'
                  '$\\sigma$=0.127, $k$=12288: 0.82 $\\to$ 0.34 $\\to$ 0.04', fontsize=10)
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(OUT, dpi=160)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
