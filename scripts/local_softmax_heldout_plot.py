"""MSE vs NOISE: linear / global softmax / local softmax (Lukoianov, Kamb & Ganguli) / EDM U-net.

    michimin, 2026-09-28: "comparison plot comparing linear, softmax global (train=eval),
    softmax local (Lukianov), softmax local (Kamb), UNet EDM ... the MSE vs sigma scaling
    curve for these types"

SOURCES (all CIFAR-10, per-image summed squared error in [0,1] pixel units, sigma in pixel
units; every held-out model uses CIFAR train[:10000] as its pool / fit set):
  tables/local_softmax_heldout.npz      scripts/local_softmax_heldout.py, first 1000 TEST
      wiener|s         linear Wiener W_U (pool covariance)
      ls|s|63          GLOBAL softmax, held out (LS with a whole-image window IS the global
                       empirical-Bayes oracle; asserted in that script's _validate)
      ls|s|k           Kamb & Ganguli LS (position-specific), k selected on VAL
      luk|s|tau|global Lukoianov et al., tau selected on VAL
      edm-uncond-vp|s  EDM U-net (trained on all 50k train images -- a data handicap in its
                       favour, see scripts/edm_pixel_heldout.py)
  (Kamb & Ganguli ELS is NOT drawn: dropped 2026-09-28 at michimin's request after two sigma,
   tables/local_softmax_heldout_els.npz, ~24 min per sigma.)
  tables/bayes_oracle_heldout.npz
      oracle|s [2]     GLOBAL softmax, TRAIN = EVAL: the pool contains the image being
                       denoised (self-inclusive, the definition of the in-sample oracle).
      linear|s [0]     its in-sample Wiener partner, for the ratio panel

VAL = CIFAR train[10000:10000+NVAL], disjoint from pool and test.  Hyper-parameters are never
chosen on test.  A curve is drawn only on the sigma it has; nothing is interpolated.

LEFT: raw MSE vs sigma, log-log, with the two trivial references: the identity D(y)=y
(d sigma^2) and the constant predictor mu_train (E||x0 - mu||^2 on test).  RIGHT: MSE / the
Wiener on THE SAME IMAGES AND NOISE (subset Wiener for the held-out curves, in-sample Wiener
for the train=eval curve), LOG scale on [R_LO, R_HI]; < 1 beats linear.  Points off that
range (global held out far above, train=eval near zero) are pinned to the edge with a
triangle and, above, their value; left-panel zeros are pinned to FLOOR_L the same way.

Writes figures/local_softmax_vs_sigma.{png,pdf} and a table twin .csv.

    python scripts/local_softmax_heldout_plot.py
    python scripts/local_softmax_heldout_plot.py --ylinear     # linear y, *_ylin.*
    python scripts/local_softmax_heldout_plot.py --dataset ffhq32 [--ylinear]   # or afhq32
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = lambda p: os.path.join(ROOT, 'tables', p)
# --ylinear: linear y on both panels (MSE from 0; ratio on [0, 2]), written to *_ylin.*
YLIN = '--ylinear' in sys.argv
# --dataset cifar10 (default) | ffhq32 | afhq32   (tables from scripts/run_local_softmax_faces.sh)
DS = sys.argv[sys.argv.index('--dataset') + 1] if '--dataset' in sys.argv else 'cifar10'
DSNAME = {'cifar10': 'CIFAR-10', 'ffhq32': 'FFHQ 32×32', 'afhq32': 'AFHQ 32×32',
          'ffhq64': 'FFHQ 64×64', 'afhq64': 'AFHQ 64×64'}[DS]
RES = 64 if DS.endswith('64') else 32
GK = str(2 * RES - 1)              # LS window covering the whole image = the global softmax
NFULL = {'ffhq32': '70k', 'ffhq64': '70k', 'afhq32': '15.8k', 'afhq64': '15.8k'}.get(DS, '')
OUT = os.path.join(ROOT, 'figures', 'local_softmax_vs_sigma' + ('' if DS == 'cifar10' else f'_{DS}')
                   + ('_ylin' if YLIN else ''))
SIGS = [0.0002, 0.0005, 0.001, 0.002,                              # very low, 2026-09-28
        0.005, 0.01, 0.02, 0.05,                                   # wide range, 2026-09-28
        0.127, 0.452, 0.621, 0.853, 1.172, 1.610, 2.212, 5.0,     # the project grid
        10.0, 20.0, 40.0]                                          # sigma_edm = 80 = EDM max
d = 3072
FLOOR_L = 1e-6                        # log-axis floor for the pinned train=eval points
EDM_SMIN = 0.001                      # sigma_edm = 0.002 = EDM's sigma_min; below = extrapolation
R_LO, R_HI = 0.25, 30.0               # right panel (log) range; off-scale points pinned
if YLIN:                              # nothing to pin below: 0 is on a linear axis
    FLOOR_L, R_LO, R_HI = -np.inf, -np.inf, 2.0


def load(p):
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


def spell(s):
    return [f'{s:g}', str(s), repr(s), f'{s:.3f}']


def fetch(store, fmt, s):
    for k in spell(s):
        if fmt.format(k) in store:
            return store[fmt.format(k)]
    return None


A = load(T('local_softmax_heldout.npz' if DS == 'cifar10' else f'local_softmax_heldout_{DS}.npz'))
O = {}
if DS == 'cifar10':        # CIFAR's train = eval curve is the oracle table (all 10k train images)
    O = load(T('bayes_oracle_heldout.npz'))
    for f in ('bayes_oracle_heldout_wide.npz', 'bayes_oracle_heldout_wide2.npz'):  # extra sigma,
        O.update({k: v for k, v in load(T(f)).items()                              # same function
                  if k.startswith(('oracle|', 'linear|'))})

# network curves: key -> (table key format, legend label, short label, marker, filled, linestyle)
NETS = ({'edm': ('test:edm-uncond-vp|{}', 'EDM U-net (VP, 50k train)', 'EDM', 's', True, '-')}
        if DS == 'cifar10' else {})
NETDEF = {
    'unet-edm':            (f'EDM U-net (official, all {NFULL}; test seen)', 'EDM', 's', True, '-'),
    'unet-full_longtrain': (f'U-net, all {NFULL}, long (test seen)', f'U-net {NFULL} long', 's', True, '-'),
    'unet-30000_longtrain_split1': ('U-net, 30k [0:30k], 250k steps, held out', 'U-net 30k long',
                                    'o', True, (0, (4, 2))),
    'unet-10000_longtrain_split1': ('U-net, 10k = the pool, 250k steps, held out', 'U-net 10k long',
                                    'o', False, (0, (1.5, 1.5))),
    'unet-full':           ('U-net, all images, 50k steps (test seen)', 'U-net full', 's', True, '-'),
    'unet-30000_split1':   ('U-net, 30k [0:30k], held out', 'U-net 30k', 'o', True, (0, (4, 2))),
    'unet-10000_split1':   ('U-net, 10k = the pool, held out', 'U-net 10k', 'o', False, (0, (1.5, 1.5))),
}
# which stored networks to DRAW (all stay in the .csv): '--nets a,b' or a per-dataset default.
# FFHQ64 stores six; drawn by default = best held-out (30k, 50k steps), pool-matched 10k, EDM.
PLOT_NETS = (sys.argv[sys.argv.index('--nets') + 1].split(',') if '--nets' in sys.argv else
             {'ffhq64': ['unet-30000_split1', 'unet-10000_split1', 'unet-edm']}.get(DS))
for tag in (list(np.asarray(A.get('meta_NETS', [])).ravel()) if DS != 'cifar10' else []):
    lab, short, mk, filled, ls = NETDEF[str(tag)]
    NETS[str(tag)] = (f'test:{tag}|{{}}', lab, short, mk, filled, ls)


def params(store, arm, s):
    """All stored parameter values for arm at sigma s (keys test:{arm}|{s}|{param}...)."""
    for sp in spell(s):
        pre = f'test:{arm}|{sp}|'
        ps = [k[len(pre):] for k in store if k.startswith(pre)]
        if ps:
            return sp, ps
    return None, []


BASE = {'ls': ({'3', '5', '7', '9', '11', '13', '15', '19', '23', '31'} if RES == 32 else
               {'1', '3', '5', '7', '9', '11', '15', '19', '23', '31', '39', '47', '63'}),
        # 64 px: sigma >= 0.02 ran the reduced grid (scripts/run_local_softmax_64.sh)
        'luk': {f'{t}|global' for t in (('0.005', '0.01', '0.02', '0.05', '0.1', '0.2')
                                        if RES == 32 else ('0.02', '0.05', '0.1', '0.2', '0.3'))},
        'lukB': {f'{t}|global' for t in ('0.05', '0.1', '0.2')}}


def selected(store, arm, s, exclude=()):
    """Val-selected (param, test vector, val mean) for a swept arm; None if not run."""
    sp, ps = params(store, arm, s)
    ps = [p for p in ps if p not in exclude]
    # a sweep still being written must not be drawn: selecting among the first few k / tau
    # of an unfinished sigma produced a spurious 1.7x Lukoianov point in a draft render.  A
    # sigma is drawn once its BASE grid is complete; later extras (LS k=1, tau >= 0.3,
    # added for the low-sigma end) join the val selection wherever they exist.
    if not ps or not BASE[arm] <= set(ps):
        return None
    best = min(ps, key=lambda p: store[f'val:{arm}|{sp}|{p}'].mean())
    return best, store[f'test:{arm}|{sp}|{best}']


def mean_se(v):
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


rows = []   # (curve, sigma, mse, se, ratio_to_wiener, ratio_se, param, n_images)
# RATIO REFERENCE: the pool Wiener, unless a full-rank variant-B Wiener exists (64 px): there
# the pool covariance has rank 9999 < d, so the pool Wiener floors at ~0.4 per image as
# sigma -> 0 and would make every other curve look good at low sigma.
REF = 'wienerB' if any(k.startswith('test:wienerB|') for k in A) else 'wiener'
for s in SIGS:
    w0 = fetch(A, 'test:wiener|{}', s)
    w = fetch(A, f'test:{REF}|{{}}', s)
    if w is not None:
        rows.append(('linear', s, *mean_se(w0), float(w0.mean() / w.mean()),
                     float(np.std(w0 - w, ddof=1) / np.sqrt(len(w)) / w.mean()), '', len(w0)))
        if REF == 'wienerB':
            rows.append(('linearB', s, *mean_se(w), 1.0, 0.0, '', len(w)))
        for curve, arm, ex in (('ls', 'ls', (GK,)), ('luk', 'luk', ()), ('lukB', 'lukB', ())):
            r = selected(A, arm, s, ex)
            if r:
                p, v = r
                rat = v / w.mean()
                rows.append((curve, s, *mean_se(v), float(v.mean() / w.mean()),
                             float(np.std(v - w, ddof=1) / np.sqrt(len(v)) / w.mean()),
                             p, len(v)))
        for curve, fmt in [('global_heldout', 'test:ls|{}|' + GK)] + [
                (k, v[0]) for k, v in NETS.items()]:
            v = fetch(A, fmt, s)
            if v is not None:
                rows.append((curve, s, *mean_se(v), float(v.mean() / w.mean()),
                             float(np.std(v - w, ddof=1) / np.sqrt(len(v)) / w.mean()),
                             '', len(v)))
    gi, wi = fetch(A, 'test:global_insample|{}', s), fetch(A, 'test:wiener_insample|{}', s)
    # (train = eval keeps its own in-sample POOL Wiener as reference: it is scored in-sample)
    if gi is not None:      # faces: first 1000 POOL images, pool contains each of them
        rows.append(('global_insample', s, *mean_se(gi), float(gi.mean() / wi.mean()),
                     float('nan'), '', len(gi)))
    o = fetch(O, 'oracle|{}', s)
    if o is not None:
        ins, lin_ins = float(o[2]), float(fetch(O, 'linear|{}', s)[0])
        rows.append(('global_insample', s, ins, float('nan'), ins / lin_ins, float('nan'),
                     '', 10000))

# held-out global cross-check: subset LS k=63 vs the full-test oracle table
for s in SIGS:
    v, o = fetch(A, 'test:ls|{}|' + GK, s), fetch(O, 'oracle|{}', s)
    if v is not None and o is not None:
        print(f"  global held-out sigma={s}: subset {v.mean():8.3f}  full-test table "
              f"{float(o[0]):8.3f}")

# ---------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SERIES = [  # key, label, color, marker, filled, linestyle
    ('linear',          'Linear (Wiener)' + (' — 10k pool cov (rank 9999)' if RES == 64 else ''),
                                                                     MAGENTA, 'v', True, '-'),
    ('linearB',         'Linear (Wiener) — larger full-rank cov',    MAGENTA, 'v', False, (0, (4, 2))),
    ('global_insample', 'Softmax, global — train = eval',            YELLOW,  'D', False, (0, (1.5, 1.5))),
    ('global_heldout',  'Softmax, global — held out',                YELLOW,  'D', True, '-'),
    ('luk',             'Softmax, local — Lukoianov (Wiener mask)',  AQUA,    '^', True, '-'),
    ('lukB',            'Lukoianov — mask from larger covariance',   AQUA,    '^', False, (0, (4, 2))),
    ('ls',              'Softmax, local — Kamb & Ganguli LS',        ORANGE,  'o', True, '-'),
] + [(k, v[1], BLUE, v[3], v[4], v[5]) for k, v in NETS.items()    # one hue: all networks
       if PLOT_NETS is None or k in PLOT_NETS or k == 'edm']
DRAWN = {k for k, *_ in SERIES}
SHORT = {'linear': 'linear', 'linearB': 'linear (B)', 'global_insample': 'global (train=eval)',
         'global_heldout': 'global held out', 'luk': 'Lukoianov', 'ls': 'LS', 'lukB': 'Lukoianov (B)',
         **{k: v[2] for k, v in NETS.items()}}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURF)
sg = np.array(SIGS)
xs = np.geomspace(1.5e-4, 60, 100)
axes[0].plot(xs, d * xs ** 2, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
axes[0].text(*((0.3, 140) if YLIN else (0.16, 60)), 'identity  dσ²', color=MUTED, fontsize=8, rotation=0)
tr = float(np.asarray(O.get('trace_test', A.get('meta_trace_test', [np.nan]))).ravel()[0])
axes[0].axhline(tr, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
axes[0].text(1.3, tr * 1.08, 'constant μ_train', color=MUTED, fontsize=8)
axes[1].axhline(1.0, color=INK2, lw=0.9, zorder=1)

labels = {0: [], 1: []}          # (x, y, text) per panel, spread apart afterwards
for ai, (ax, col) in enumerate(((axes[0], 2), (axes[1], 4))):
    lo, hi = (FLOOR_L, np.inf) if ai == 0 else (R_LO, R_HI)
    for key, lab, c, mk, filled, ls in SERIES:
        pts = sorted((r[1], r[col], r[6]) for r in rows if r[0] == key)
        if not pts:
            continue
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts], float)
        below, above = y < lo, y > hi
        yp = np.clip(y, lo, hi)
        ax.plot(x, yp, color=c, lw=1.6, ls=ls, zorder=3 if filled else 2,
                solid_capstyle='round', dash_capstyle='round',
                label=lab if ai == 0 else None)
        ok = ~(below | above)
        # small markers, thin ring: at 15 sigma the 1.3-pt surface ring cut the lines into
        # apparent dashes wherever grid points sit ~0.13 decades apart
        ext = (x < EDM_SMIN) if key in NETS else np.zeros_like(ok)   # below nets' sigma_min
        ax.scatter(x[ok & ~ext], yp[ok & ~ext], marker=mk, s=20, zorder=4,
                   facecolor=c if filled else SURF, edgecolor=SURF if filled else c,
                   lw=0.6 if filled else 1.1)
        ax.scatter(x[ok & ext], yp[ok & ext], marker=mk, s=22, zorder=4, facecolor=SURF,
                   edgecolor=c, lw=1.1)
        for m, sel in (('v', below), ('^', above)):
            if sel.any():
                ax.scatter(x[sel], yp[sel], marker=m, s=28, zorder=4, facecolor=SURF,
                           edgecolor=c, lw=1.3, clip_on=False)
        labels[ai].append((x[0], yp[0], SHORT[key]))

for ax in axes:
    ax.set_facecolor(SURF)
    ax.set_xscale('log'); ax.set_yscale('linear' if YLIN else 'log')
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.001', '0.01', '0.1', '1', '10'])
    ax.set_xticks(SIGS, minor=True)        # every measured sigma, unlabelled
    ax.set_xticklabels([], minor=True)
    ax.tick_params(axis='x', which='minor', length=3, color=AXIS)
    ax.set_xlim(5e-6 if YLIN else 1.2e-5, 60)
    ax.set_xlabel('noise level σ  (pixel units, [0,1] images)')
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
if YLIN:
    axes[0].set_ylim(0, 210)
    axes[1].set_ylim(0, R_HI * 1.06)
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    axes[1].set_yticklabels(['0', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2'])
else:
    axes[0].set_ylim(FLOOR_L * 0.7, 600)
    axes[0].text(0.012, FLOOR_L * 1.9, '▼ pinned: ≈ 0', color=MUTED, fontsize=7.5)
    axes[1].set_ylim(R_LO * 0.85, R_HI * 1.6)
    axes[1].set_yticks([0.3, 0.5, 1, 2, 5, 10, 20])
    axes[1].set_yticklabels(['0.3', '0.5', '1', '2', '5', '10', '20'])
    axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[0].set_ylabel('MSE  E‖x₀ − D(y)‖²  per image (sum over 3072 coords)')
axes[0].set_title(f'MSE vs noise, {DSNAME}', loc='left')
axes[1].set_ylabel('MSE / Wiener MSE on the same images and noise'
                   + ('\n(reference: full-rank Wiener B)' if REF == 'wienerB' else ''))
axes[1].set_title('Relative to linear  (< 1 beats Wiener)', loc='left')
NLEG = len([1 for k, *_ in SERIES if any(r[0] == k for r in rows)])   # drawn series only
LEGH = 0.025 * ((NLEG + 3) // 4)              # legend height grows with its row count
fig.legend(loc='lower center', ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 0.05))
n_a = int(np.asarray(A.get('meta_NTEST', 0)))
if YLIN:
    axes[1].text(6e-6, R_HI * 1.025, f'▲ > {R_HI:g}×', color=MUTED, fontsize=7.5)
else:
    axes[1].text(1.5e-5, R_HI * 1.25, f'▲ > {R_HI:g}×', color=MUTED, fontsize=7.5)
    axes[1].text(1.5e-5, R_LO * 0.93, f'▼ < {R_LO:g}×', color=MUTED, fontsize=7.5, va='top')
if DS == 'cifar10':
    head = (f'Held out: pool/fit = train[:10000], first {n_a} test images, k and τ picked on '
            f'validation.  EDM: trained on 50k; hollow = σ < {EDM_SMIN:g}, below its training '
            f'range.  train = eval: pool contains the evaluated image.')
else:
    ts = int(np.asarray(A.get('meta_TEST_START', 0)))
    nv = int(np.asarray(A.get('meta_NVAL', 0)))
    head = (f'Held out: pool/fit = images [0:10000], test = [{ts}:{ts + n_a}], k and τ picked '
            f'on the last {nv} images.  Networks: hollow = σ < {EDM_SMIN:g} (below training '
            f'range).  train = eval: pool images [0:{n_a}], pool contains each.')
fig.text(0.01, 0.955, head, fontsize=7.5, color=INK2)
sel = []
for key, name, pre in (('ls', 'LS', 'k'), ('luk', 'Lukoianov', 'τ')):
    cells = sorted((r[1], r[6].split('|')[0]) for r in rows if r[0] == key)
    if cells:
        sel.append(f"{name} {pre} = " + ', '.join(f'{v}' for _, v in cells))
fig.text(0.01, 0.005, 'val-selected, by σ ascending:   ' + ';   '.join(sel),
         fontsize=7.5, color=INK2)
fig.text(0.01, 0.028, 'Lukoianov at σ ≤ 0.001 → 0 is an 8-bit artefact: as σ → 0 its Wiener mask '
         'shrinks to the diagonal, a per-coordinate nearest value among 10k training pixels, '
         'which snaps to the exact 1/255 level once σ ≪ 1/255.', fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.06 + LEGH + 0.03, 1, 0.95))


def spread(ax, items, gap=11):
    """Direct labels at each curve's last point, pushed apart vertically (display px)."""
    if not items:
        return
    tr = ax.transData
    P = sorted(([*tr.transform((x, y)), t, x, y] for x, y, t in items), key=lambda q: q[1])
    ys = [q[1] for q in P]
    for i in range(1, len(ys)):
        if abs(P[i][0] - P[i - 1][0]) < 60 and ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for q, yy in zip(P, ys):
        ax.annotate(q[2], (q[3], q[4]), xytext=(-7, (yy - q[1]) * 72 / fig.dpi),
                    textcoords='offset points', fontsize=7.5, color=INK2, va='center',
                    ha='right')


for ai, ax in enumerate(axes):
    spread(ax, labels[ai])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + '.png', dpi=180, facecolor=SURF)
fig.savefig(OUT + '.pdf', facecolor=SURF)

with open(OUT + '.csv', 'w') as f:
    f.write('curve,sigma,mse,se,ratio_to_wiener,ratio_se,selected_param,n_images\n')
    order = [k for k, *_ in SERIES] + [k for k in NETS if k not in DRAWN]
    for r in sorted(rows, key=lambda r: (order.index(r[0]), r[1])):
        f.write(','.join(str(x) for x in r) + '\n')
print(f"wrote {OUT}.png/.pdf/.csv  ({len(rows)} cells)")
for key, *_ in SERIES:
    cells = sorted((r[1], r[2], r[6]) for r in rows if r[0] == key)
    print(f"  {key:16s} " + '  '.join(f'{s:g}:{m:.2f}{("["+p+"]") if p else ""}'
                                      for s, m, p in cells))
