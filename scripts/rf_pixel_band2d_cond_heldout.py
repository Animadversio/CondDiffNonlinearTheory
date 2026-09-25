"""LABEL-CONDITIONED band-B NONLINEAR RF on Z_32 x Z_32, HELD OUT (michimin, 2026-09-25).

    "add code to enable computing conditional circulant nonlinear denoiser. this shouldn't
     touches the per-frequency decoupling ... i'll run them on the clusters"

    STATUS: written 2026-09-25, NEVER EXECUTED.  Run scripts/run_selftest_cond.sh first.

THE MODEL -- core/rf_circulant2d_band.py, arguments lab= / gam= / lab_test= / class_centre=

    phi_a[u] = relu( (h_a * y)[u] + <gamma_a, U> )        gamma_a ~ N(0, GSCALE^2 I / 10),
                                                          ONE per feature plane (all u share it)
    MODE=vu    D = sum_{a,r} W_{a,r} * (m_r phi_a) + V U + beta     label in phi AND readout
    MODE=feat  D = sum_{a,r} W_{a,r} * (m_r phi_a)       + beta     label only inside phi
    MODE=vu0   D = sum_{a,r} W_{a,r} * (m_r phi_a^0)     + V U + beta  gamma = 0: readout only

MODE=vu is the nonlinear member of the "S" linear class of scripts/rf_cond_toll2d_heldout.py
(shared slope, class mean known), so band_S(B) there is to this run what band_U(B) is to the
unconditional band driver: the linear baseline its nonlinear gain is measured against.
MODE=vu0 isolates what the free V U alone buys; vu - vu0 is what ALSO telling the features the
class adds.  MODE=feat is the dense-RF convention (core/rf_gmm_estimators.py, readout
label-free) and docs/rf_circulant2d_conditional.md sec 1.2; its linear class is U plus a
per-class offset built from the (2B+1)^2 lowest spatial frequencies only (uniform per channel
at B=0), so band_U(B) is only an upper bound on that class's floor.

BASELINES, all held out on the same split and printed per sigma:
    W_U  linear_split (the unconditional Wiener every table uses)
    W_S  class mean known, pooled within-class Wiener      } scripts/rf_cond_toll2d_heldout.py
    W_C  per-class Wiener, averaged over test images       }
and, if tables/rf_cond_toll2d_heldout.npz exists, band_S(B) / band_U(B) at the same sigma.

PAIRED WITH THE UNCONDITIONAL RUNS.  h = filt2d(c, s) is the SAME draw the unconditional band
driver uses at (T2, c, seed); gamma comes from a generator of its own (seed 7700 + 11 s + c),
so it never perturbs h.  Where the unconditional cell exists in UNCOND the paired differential
cond - uncond on the SAME Theta draw is printed.

SIZING.  The conditional path adds only per-image bookkeeping to the Stein assembly, so the
unconditional driver's sizing() is reused unchanged.  class_centre adds one (c, nb, D)
feature-chunk transient in the moment pass (~2 GB at the default sample_chunk); leave that
much headroom when a cell is already near the card's limit.

OUT = tables/rf_pixel_band2d_cond_heldout_{MODE}[_t{T2}].npz, one file per mode, because the
table is rewritten whole after every cell: NEVER run two jobs on the same OUT at once.
Keys '{cell_key}|{MODE}|g{GSCALE}' (vu0: '{cell_key}|vu0'), rows = seeds, columns =
[train, train_resid, test];  'linU|{sg}', 'linS|{sg}', 'linC|{sg}' = [train, test].

    MODE=vu CS=96 BS=0,1 SIGS=0.621,1.61 python scripts/rf_pixel_band2d_cond_heldout.py
    MODE=vu T2=7 CS=256 BS=0,3 SIGS=1.61 python scripts/rf_pixel_band2d_cond_heldout.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from core.rf_circulant2d_band import circulant2d_band_rf_mmse
from scripts.rf_pixel_heldout import load, linear_split, filt2d, DEV, DT, LAM, H, W, CIN, d
from scripts.rf_pixel_band2d_heldout import sizing, cell_key
from scripts.rf_cond_toll2d_heldout import load_labels, conditional_families, NCLS

T2 = int(os.environ.get('T2', '3'))
CS = [int(x) for x in os.environ.get('CS', '96').split(',')]
BS = [int(x) for x in os.environ.get('BS', '0,1').split(',')]
SIGS = [float(x) for x in os.environ.get('SIGS', '0.621,0.853,1.172,1.610').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
NSEED = int(os.environ.get('NSEED', '1'))
MODE = os.environ.get('MODE', 'vu')
GSCALE = float(os.environ.get('GSCALE', '1.0'))
# ONE FILE PER (MODE, T2) BY DEFAULT: the driver rewrites its whole table after every cell, so
# two live jobs sharing an OUT silently delete each other's cells.  Separate files make the
# natural "submit vu and vu0 side by side" safe; jobs of the SAME mode must still not overlap.
OUT = os.environ.get('OUT', f'tables/rf_pixel_band2d_cond_heldout_{MODE}'
                            + ('' if T2 == 3 else f'_t{T2}') + '.npz')
UNCOND = os.environ.get('UNCOND', ','.join(
    ['tables/rf_pixel_band2d_heldout.npz', 'tables/rf_pixel_band2d_heldout_t3fill.npz',
     'tables/rf_pixel_heldout.npz'] if T2 == 3 else
    [f'tables/rf_pixel_band2d_heldout_t{T2}.npz']))
TOLL = os.environ.get('TOLL', 'tables/rf_cond_toll2d_heldout.npz')


def gamma2d(c, s):
    """gamma_a ~ N(0, GSCALE^2 I / n_cls), one per feature plane -- the scale of
    core/rf_circulant.py::build_circulant_gamma -- from a generator of its own, so filt2d's
    draw (and with it the pairing against the unconditional tables) is untouched."""
    g = torch.Generator(device=DEV)
    g.manual_seed(7700 + 11 * s + c)
    return torch.randn(c, NCLS, generator=g, device=DEV, dtype=DT) * (GSCALE / np.sqrt(NCLS))


def cond_key(sg, c, B):
    base = f'{cell_key(sg, c, B)}|{MODE}'
    return base if MODE == 'vu0' else f'{base}|g{GSCALE:g}'


def uncond_cell(tabs, sg, c, B):
    """The unconditional held-out cell on the SAME (T2, c, B, seeds), or None.  At T2=3, B=0
    the plain 2-D table ('{sg}|{c}|2d') is the same model (asserted by the band driver)."""
    keys = [cell_key(sg, c, B)] + ([f'{sg}|{c}|2d'] if (B == 0 and T2 == 3) else [])
    for k in keys:
        for t in tabs:
            if k in t:
                return np.atleast_2d(np.asarray(t[k], float))
    return None


def main():
    from scripts import rf_pixel_heldout as _ph
    assert _ph.T2 == T2, f'tap size disagrees: filt2d uses {_ph.T2}, this driver uses {T2}'
    assert MODE in ('vu', 'feat', 'vu0'), f"MODE must be vu, feat or vu0, got {MODE!r}"
    Xtr = load(True, NIMG)
    Xte = load(False, NTEST)
    Xtr1 = Xtr.reshape(NIMG, d)
    Xte1 = Xte.reshape(NTEST, d)
    ytr = load_labels(True, NIMG, Xtr1)
    yte = load_labels(False, NTEST, Xte1)
    print(f"LABEL-CONDITIONED BAND RF, held out.  MODE={MODE}  GSCALE={GSCALE:g}  "
          f"train={NIMG}  test={NTEST}  c={CS}  B={BS}  taps={T2}x{T2}  seeds={NSEED}  "
          f"-> {OUT}", flush=True)

    store = {}
    if os.path.exists(OUT):
        store = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}
        print(f"resuming from {OUT}: {len(store)} entries already present", flush=True)
    tabs = [dict(np.load(f, allow_pickle=True)) for f in UNCOND.split(',') if os.path.exists(f)]
    toll = dict(np.load(TOLL, allow_pickle=True)) if os.path.exists(TOLL) else None
    tsig = ({float(v): i for i, v in enumerate(np.asarray(toll['sigmas']).ravel())}
            if toll is not None else {})
    # the linear class this RF lives in: S for vu / vu0.  For feat it is U plus a per-class
    # offset in the (2B+1)^2 lowest frequencies (all a band-B readout can make of a constant
    # plane; uniform at B=0), so band_U is an upper bound on its floor.
    lin_fam = 'U' if MODE == 'feat' else 'S'

    # conditional free optima (Wiener only: no band solves here)
    RSC = conditional_families(Xtr1, ytr, Xte1, yte, NCLS, SIGS, [], verbose=False)

    for sg in SIGS:
        lin = {'linU': linear_split(Xtr1, Xte1, sg), 'linS': RSC['S'][('W', sg)],
               'linC': RSC['C'][('W', sg)]}
        for nm, v in lin.items():
            lk = f'{nm}|{sg}'
            if lk in store:       # no RNG: a resume on the same split MUST reproduce it
                dmax = float(np.max(np.abs(np.asarray(store[lk], float) - np.array(v))))
                assert dmax < 1e-9, f'{lk} does not reproduce stored value: {dmax:.3e}'
            store[lk] = np.array(v)
        wU, wS, wC = lin['linU'][1], lin['linS'][1], lin['linC'][1]
        print(f"\n=== sigma={sg}   held-out Wiener  W_U {wU:.4f}   W_S {wS:.4f}   W_C {wC:.4f}"
              f"   (A_S {wU - wS:+.4f}, A_C {wU - wC:+.4f}) ===", flush=True)
        for c in CS:
            for B in BS:
                key = cond_key(sg, c, B)
                npar = CIN * c * H * W * (2 * B + 1) ** 2
                have = len(np.atleast_2d(store[key])) if key in store else 0
                if have >= NSEED:
                    v = np.atleast_2d(store[key])
                    print(f"  B={B} c={c}: already done, {have} seed(s) "
                          f"(test {np.mean(v[:, 2]):.4f})", flush=True)
                    continue
                nf, ns = sizing(c, B)
                rows = [list(r) for r in np.atleast_2d(store[key])] if have else []
                for s in range(have, NSEED):
                    t1 = time.time()
                    gam = None if MODE == 'vu0' else gamma2d(c, s)
                    r = circulant2d_band_rf_mmse(Xtr, filt2d(c, s), sg, T2, B, lam=LAM,
                                                 device=DEV, freq_chunk=nf, super_chunk=ns,
                                                 x0_test=Xte, lab=ytr, gam=gam, lab_test=yte,
                                                 class_centre=(MODE != 'feat'))
                    rows.append([r['train'], r['train_resid'], r['test']])
                    print(f"      [seed {s}: train {r['train']:.4f}  resid "
                          f"{r['train_resid']:.4f}  test {r['test']:.4f}  "
                          f"{time.time() - t1:.0f}s  nf={nf} NS={ns}  "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB]",
                          flush=True)
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                a = np.array(rows)
                store[key] = a
                np.savez(OUT, **store)
                te = float(a[:, 2].mean())
                print(f"  >> {MODE} B={B} c={c} ({npar:,} trained params): test {te:.4f}   "
                      f"vs W_U {te - wU:+.4f}   vs W_S {te - wS:+.4f}   vs W_C {te - wC:+.4f}"
                      f"   own gap {te - float(a[:, 1].mean()):+.4f}", flush=True)
                if toll is not None and sg in tsig and f'{lin_fam}|band2d|{B}' in toll:
                    bl = float(toll[f'{lin_fam}|band2d|{B}'][tsig[sg]][1])
                    what = 'its linear class' if lin_fam == 'S' else 'upper bound on its class'
                    print(f"     [{what}: band_{lin_fam}(B={B}) = {bl:.4f} held out  "
                          f"=> nonlinear gain {bl - te:+.4f}]", flush=True)
                p = uncond_cell(tabs, sg, c, B)
                if p is not None:
                    n = min(len(p), len(a))
                    dte = float(np.mean(a[:n, 2] - p[:n, 2]))
                    print(f"     [cond - uncond at B={B} c={c} T2={T2}, same Theta draw(s) "
                          f"0..{n - 1}: {dte:+.4f} held out  (uncond test "
                          f"{float(np.mean(p[:n, 2])):.4f})]", flush=True)
                else:
                    print(f"     [no unconditional cell at B={B} c={c} T2={T2} in UNCOND: "
                          f"nothing to pair against]", flush=True)
    print("\ndone", flush=True)


if __name__ == '__main__':
    main()
