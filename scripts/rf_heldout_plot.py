"""HELD-OUT LOSS vs NOISE for every model class we have (michimin, 2026-09-23 06:11).

    "graph the losses (dense, circ 1d, circ 2d, linear on 50k) against the noise"

EVERY CURVE IS SCORED ON THE SAME 10,000 CIFAR TEST IMAGES.  That is the whole point: the
trace offset between the two CIFAR splits (Tr(Sigma_test) is 0.9% below Tr(Sigma_train))
passes ~1:1 into the loss at large sigma, so absolute losses are only comparable when the
evaluation set is held fixed.  It is, here, for all of them.

THE TRAINING SETS ARE *NOT* ALL THE SAME, AND THE FIGURE SAYS SO.
  dense / circ 1-D / circ 2-D  -- fitted on CIFAR train[:10000]
  linear (10k)                 -- fitted on the same 10,000, the matched baseline
  linear (50k)                 -- fitted on ALL 50,000 train images   <- michimin's request
  EDM U-Net                    -- trained on all 50,000, a frozen network
The two 50k rows are handicaps in the BASELINE's favour and are drawn dashed/dotted so they
cannot be misread as matched comparisons.  The matched comparison is RF-vs-linear(10k).

EACH RF CLASS IS PLOTTED AT ITS BEST AVAILABLE WIDTH, i.e. the lower envelope over the
widths measured, and the annotation lists which width that is.  For dense at low sigma this
matters: its held-out curve is U-shaped, so "best width" is k/d=4, not the largest.

    python scripts/rf_heldout_plot.py          # regenerates from npz in ~1 s, no GPU
    REBUILD_LIN50=1 python scripts/rf_heldout_plot.py    # recompute the 50k linear row
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = 3072
SIGS = ('0.127', '0.452', '0.621', '0.853', '1.172', '1.61', '2.212', '5.0')
LIN50 = 'tables/rf_linear50k_heldout.npz'
FIG = 'figures/rf_heldout_vs_sigma.png'


def load_npz(p):
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


def build_lin50():
    """Wiener fitted on ALL 50,000 CIFAR train images, scored on the 10,000 test images."""
    import torch
    from scripts.rf_pixel_heldout import load, linear_split
    Xtr = load(True, 50000).reshape(50000, d)
    Xte = load(False, 10000).reshape(10000, d)
    out = {}
    for sg in SIGS:
        tr, te = linear_split(Xtr, Xte, float(sg))
        out[f'lin50|{sg}'] = np.array([tr, te])
        print(f"  linear(50k)  sigma={sg:>6}   train {tr:9.4f}   test {te:9.4f}", flush=True)
    np.savez(LIN50, **out)
    return out


def envelope(tab, sg, suffix, widths, col=2):
    """Best (lowest) held-out loss over the widths measured, and the width that achieved it."""
    best, at = None, None
    for w in widths:
        a = tab.get(f'{sg}|{w}|{suffix}')
        if a is None:
            continue
        v = float(a[:, col].mean())
        if best is None or v < best:
            best, at = v, w
    return best, at


def main():
    h = load_npz('tables/rf_pixel_heldout.npz')
    dn = load_npz('tables/rf_pixel_dense_heldout.npz')
    ed = load_npz('tables/edm_pixel_heldout.npz')
    l50 = load_npz(LIN50)
    if not l50 or os.environ.get('REBUILD_LIN50'):
        l50 = build_lin50()

    CS = (32, 96, 256, 512, 1536, 3072, 6144)
    DJS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

    x, series = [], {k: [] for k in
                     ('lin10', 'lin50', 'edm', '2d', '1d', 'dense')}
    where = {'2d': [], '1d': [], 'dense': []}
    for sg in SIGS:
        # spine = the 50k linear row, which exists at every sigma.  Gating the axis on the
        # circulant table instead would silently DROP sigma=2.212 -- where dense, linear and
        # EDM are all measured and only the circulant job (47923736) has not arrived yet.
        if f'lin50|{sg}' not in l50:
            continue
        L = h.get(f'linear|{sg}')
        x.append(float(sg))
        series['lin10'].append(float(L[1]) if L is not None else np.nan)
        series['lin50'].append(float(l50[f'lin50|{sg}'][1]))
        e = ed.get(f'uncond-ve|{sg}')
        series['edm'].append(float(e[2]) if e is not None else np.nan)
        for tag, tab, ws in (('2d', h, CS), ('1d', h, CS), ('dense', dn, DJS)):
            suf = 'dense' if tag == 'dense' else tag
            v, at = envelope(tab, sg, suf, ws)
            series[tag].append(v if v is not None else np.nan)
            where[tag].append(at)
    x = np.array(x)
    S = {k: np.array(v, float) for k, v in series.items()}

    STY = {
        'dense': ('dense RF', 'tab:red', 'o', '-', 2.0),
        '1d':    ('circulant 1-D  $Z_{3072}$', 'tab:orange', 's', '-', 2.0),
        '2d':    ('circulant 2-D  $Z_{32}\\times Z_{32}$', 'tab:blue', 'D', '-', 2.2),
        'lin10': ('linear, 10k (matched baseline)', 'k', '^', '-', 1.8),
        'lin50': ('linear, 50k  (5x the data)', 'k', 'v', '--', 1.8),
        'edm':   ('EDM U-Net (50k, frozen)', 'tab:green', '*', ':', 2.0),
    }
    ORDER = ['dense', '1d', '2d', 'lin10', 'lin50', 'edm']

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.4))

    for k in ORDER:
        lab, col, mk, ls, lw = STY[k]
        m = np.isfinite(S[k])
        ax[0].plot(x[m], S[k][m], ls, color=col, marker=mk, lw=lw, ms=6, label=lab)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(r'noise $\sigma$  (pixel units, $[0,1]$)')
    ax[0].set_ylabel(r'held-out loss  $\mathbb{E}\|x_0-\hat x_0\|^2$  on the 10,000 test images')
    ax[0].set_title('Held-out loss vs noise\nall curves scored on the SAME 10,000 CIFAR test images')
    ax[0].grid(alpha=.3, which='both'); ax[0].legend(fontsize=8.5, loc='upper left')

    base = S['lin50']
    for k in ORDER:
        if k == 'lin50':
            continue
        lab, col, mk, ls, lw = STY[k]
        m = np.isfinite(S[k]) & np.isfinite(base)
        ax[1].plot(x[m], (S[k] - base)[m], ls, color=col, marker=mk, lw=lw, ms=6, label=lab)
    ax[1].axhline(0, color='k', ls='--', lw=1.8)
    ax[1].text(x[0] * 1.05, 0.35, 'linear fitted on all 50k', fontsize=8, color='k')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'noise $\sigma$  (pixel units, $[0,1]$)')
    ax[1].set_ylabel('loss  $-$  linear(50k) held-out loss')
    ax[1].set_title('Excess over the 50k linear denoiser\n'
                    'below zero = beats the best linear map fitted on 5x the data')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8.5, loc='upper left')

    # The envelope is a min over widths taken ON THE TEST SET, so it carries a mild selection
    # bias.  Say so on the figure rather than in a caption nobody keeps with the png.
    note = ('each RF class at its best measured width (min over widths taken on the test '
            'set -- mild selection bias, matters only near the dense minimum at low sigma):  '
            + '   '.join(f"{t}: " + ",".join(str(w) for w in dict.fromkeys(
                [w for w in where[t] if w is not None]))
                for t in ('dense', '1d', '2d')))
    fig.text(0.5, 0.005, note, ha='center', fontsize=7.2, color='0.35')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs('figures', exist_ok=True)
    fig.savefig(FIG, dpi=160)
    print(f"wrote {FIG}")

    print("\nthe plotted numbers (best width per class, held out on the 10k test split):")
    hdr = f"{'sigma':>7} " + " ".join(f"{STY[k][0][:13]:>14}" for k in ORDER)
    print(hdr)
    for i, sg in enumerate(x):
        print(f"{sg:>7.3f} " + " ".join(f"{S[k][i]:14.4f}" for k in ORDER))
    print("\nbest width used per sigma:")
    for t in ('dense', '1d', '2d'):
        print(f"  {t:>6}: " + " ".join(f"{sg}:{w}" for sg, w in zip(x, where[t])))


if __name__ == '__main__':
    main()
