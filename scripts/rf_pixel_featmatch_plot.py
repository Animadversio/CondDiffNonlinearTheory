"""Figure for the matched-free-parameter comparison on raw CIFAR-10 pixels.

Left  : excess loss over the linear (Wiener) denoiser vs k/d, one colour per sigma.
        Dense = solid circles, circulant = dashed squares.  Plotting L - L_linear puts
        six noise levels whose raw losses span 8 to 160 on one axis, and the crossing
        is where the two curves of a colour intersect.
Right : the crossing k/d vs sigma, read off by log-interpolating the dense curve onto
        the circulant floor.

Sources (all matched free parameters, c = k):
    tables/rf_pixel_featmatch2.npz    k/d = 0.5, 1, 2  dense + circ, all sigma
    tables/rf_pixel_dense_sweep.npz   k/d = 3, 4, 6, 8 dense only
    tables/rf_pixel_parammatch.npz    c = k = 2..256   dense + circ (k/d << 1, shown faint)
The circulant is NOT computed above k/d=2 -- a c=9216 cell is a 13-22 h job -- so its
curve is drawn solid only where measured and the floor is extended as a thin dotted line.

    python scripts/rf_pixel_featmatch_plot.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = 3072
OUT = 'figures/rf_pixel_featmatch_summary.png'
SIGS = [0.127, 0.452, 1.61, 2.459, 3.756, 5.0]


def collect():
    D, C, LIN = {}, {}, {}
    for f in ('tables/rf_pixel_featmatch2.npz', 'tables/rf_pixel_dense_sweep.npz'):
        for k, v in np.load(f, allow_pickle=True).items():
            if k.startswith('linear'):
                LIN[float(k.split('|')[1])] = float(v); continue
            sg, j, w = k.split('|')
            (D if w == 'dense' else C)[(float(sg), float(j))] = float(np.mean(v))
    # small-c matched points: values are [dense, circ, circ_sd] at c = k = tag
    for k, v in np.load('tables/rf_pixel_parammatch.npz', allow_pickle=True).items():
        sg, tag = k.split('|')
        if tag == 'lin':
            LIN.setdefault(float(sg), float(v)); continue
        if tag == 'edm':
            continue
        j = int(tag) / d
        D[(float(sg), j)] = float(v[0]); C[(float(sg), j)] = float(v[1])
    return D, C, LIN


def crossing(D, C, sg):
    """log-interpolate the dense curve onto the circulant floor."""
    floor = C.get((sg, 2.0))
    js = sorted(j for (s, j) in D if s == sg)
    for lo, hi in zip(js, js[1:]):
        if D[(sg, lo)] > floor >= D[(sg, hi)]:
            f = (D[(sg, lo)] - floor) / (D[(sg, lo)] - D[(sg, hi)])
            return lo * (hi / lo) ** f
    return None


def main():
    D, C, LIN = collect()
    os.makedirs('figures', exist_ok=True)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(SIGS)))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={'width_ratios': [1.75, 1]})

    for sg, col in zip(SIGS, cols):
        lin = LIN[sg]
        jd = sorted(j for (s, j) in D if s == sg and j >= 0.5)
        jc = sorted(j for (s, j) in C if s == sg and j >= 0.5)
        ax.plot(jd, [D[(sg, j)] - lin for j in jd], 'o-', color=col, lw=1.9, ms=5,
                label=f'$\\sigma$={sg}')
        ax.plot(jc, [C[(sg, j)] - lin for j in jc], 's--', color=col, lw=1.6, ms=5,
                alpha=0.85)
        # circulant floor extended past the last COMPUTED point (not measured there)
        if jc:
            ax.plot([max(jc), 9], [C[(sg, max(jc))] - lin] * 2, ':', color=col, lw=1.1,
                    alpha=0.55)

    ax.axhline(0, color='k', lw=1.1)
    ax.text(0.53, 0.4, 'linear (Wiener)', fontsize=9, va='bottom')
    ax.set_xscale('log'); ax.set_xticks([0.5, 1, 2, 3, 4, 6, 8])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('$k/d$   (matched free parameters: $c=k$, params $=k\\cdot d$ each side)')
    ax.set_ylabel('$L - L_{\\rm linear}$')
    ax.set_title('Raw CIFAR-10 pixels, $d$=3072, $N$=10$^4$, $t$=8\n'
                 'solid $\\bullet$ dense   dashed $\\blacksquare$ circulant   '
                 'dotted = floor, not computed', fontsize=10)
    ax.legend(fontsize=8.5, ncol=2, loc='upper right')
    ax.grid(alpha=0.25)

    xs = [sg for sg in SIGS if crossing(D, C, sg)]
    ax2.plot(xs, [crossing(D, C, sg) for sg in xs], 'o-', color='crimson', lw=2, ms=7)
    for sg in xs:
        ax2.annotate(f'{crossing(D, C, sg):.2f}', (sg, crossing(D, C, sg)),
                     textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8.5)
    ax2.set_xscale('log'); ax2.set_xlabel('$\\sigma$')
    ax2.set_ylabel('$k/d$ where dense overtakes circulant')
    ax2.set_title('Crossing is non-monotone:\nfalls, then flattens at $k/d\\approx1.6$–$1.7$',
                  fontsize=10)
    ax2.set_ylim(1.3, 3.0); ax2.grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(OUT, dpi=160)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
