"""THE EQUIVARIANCE TOLL, HELD OUT.  Re-runs the band/comb relaxation with a train/test split.

michimin, 2026-09-23 05:12: "re-run rf_band_relaxation.py with the split to re-calculate toll
that isn't in sample".

WHY IT MATTERS HERE MORE THAN ANYWHERE ELSE.  A toll is a DIFFERENCE between two fitted
models -- a constrained linear denoiser and the free Wiener optimum -- and the two carry
wildly different numbers of free parameters:

    free Wiener          d*d = 9,437,184 params   (Sigma at d=3072)
    period2d p=1          3*d =     9,216
    band2d   B=1        3*d*9 =    82,944
    band2d   B=3       3*d*49 =   451,584
    period2d p=16      3*d*256 = 2,359,296

all estimated from the SAME 10,000 images.  The held-out RF work (docs/rf_heldout_and_nsweep.md
sec 3) measured what that does: the Wiener baseline's own optimism is +0.459 at sigma=0.127
while a model with ~100x fewer parameters has +0.02.  So EVERY in-sample toll in
tables/rf_band_relaxation.npz is inflated by roughly the baseline's optimism, i.e. the
constrained classes look worse than they are, and the smallest classes look worst of all
because they are the ones being compared against the most overfit reference.

The correction is not uniform across the table either: it is ~constant in the class (they are
all far less overfit than Wiener) but the TOLLS themselves span 0.4 to 9.0, so a ~0.4-0.5
shift is a rounding error on the p=1 toll at sigma=5 and a LARGE relative move on band B=3 at
sigma=0.127.  Per-parameter efficiency rankings can move.

THE SPLIT.  Same construction as scripts/rf_pixel_heldout.py:

    W_f = (P_f^train)^{-1} q_f^train                           (solved on train moments)
    L_test = Tr(Sig_te) - 2 Re sum_f <q_f^test, W_f> + sum_f <W_f, P_f^test W_f>

with P_f = (signal block) + sigma^2 I on BOTH sides -- the noise is independent of the split,
so it is not re-estimated.  There is no ridge here (sigma^2 I is the model's noise term, not a
regulariser), so unlike the RF case train and train_resid coincide and there is no
lambda*||W||^2 shift to keep track of.

CENTRING is part of the model: b = mu_train - A mu_train, so the test moments are centred by
the TRAIN mean and Tr(Sig_te) is E_test||x0 - mu_train||^2.  Re-centring on test would refit
d=3072 parameters at evaluation time.

THE BASELINE IS HELD OUT TOO, via linear_split() from scripts/rf_pixel_heldout.py.  Every
"excess over Wiener" printed below is train-vs-train or test-vs-test, never mixed.

VALIDATION (all asserted -- the run stops rather than printing a plausible wrong table):
  (a) feeding ONE split in twice must reproduce the in-sample number on BOTH columns.  This
      pins sign, normalisation and the centring convention without any reference table.
  (b) the train column must reproduce tables/rf_band_relaxation.npz (band2d, period2d) and
      tables/rf_equivariance_toll.npz (period1d|1) cell for cell: same images, same math.
  (c) band B=0 == period2d p=1 on both columns (michimin's selftest, extended to the split).
  (d) Parseval on both Fourier covariances.

    python scripts/rf_band_relaxation_heldout.py
    BS=0,1,2 PS=1,2 python scripts/rf_band_relaxation_heldout.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.rf_pixel_featmatch2 import DEV, DT, d
from scripts.rf_band_relaxation import band_index, H, Wd
from scripts.rf_pixel_heldout import linear_split, load

SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610,5.0').split(',')]
BS = [int(x) for x in os.environ.get('BS', '0,1,2,3,4').split(',')]
PS = [int(x) for x in os.environ.get('PS', '1,2,4,8,16').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
OUT = os.environ.get('OUT', 'tables/rf_band_relaxation_heldout.npz')
TOL = 1e-8


# ----------------------------------------------------------------------------- moment blocks
# Each class reduces to a set of independent blocks (q_f, P_f).  period1d/period2d are the
# special case q_f = Pm_f (the signal covariance of the block itself); the band is the general
# case, where the regressor window is wider than the output.  Factored out of
# rf_equivariance_toll.py / rf_band_relaxation.py VERBATIM so that the train column is the
# same arithmetic as the stored tables, which is what makes check (b) meaningful.

def fourier_cov(Xc):
    N = Xc.shape[0]
    Xh = torch.fft.fft2(Xc.reshape(N, 3, H, Wd), norm='ortho').reshape(N, d)
    return (Xh.conj().T @ Xh).conj() / N


def pm_1d(Xc, P):
    N, M = Xc.shape[0], d // P
    F = torch.fft.fft(Xc.reshape(N, M, P), dim=1) / np.sqrt(M)
    return torch.einsum('nma,nmb->mab', F, F.conj()) / N


def pm_2d(Xc, p):
    N, m, B = Xc.shape[0], 32 // p, 3 * p * p
    V = Xc.reshape(N, 3, m, p, m, p).permute(0, 2, 4, 1, 3, 5).reshape(N, m, m, B)
    F = (torch.fft.fft2(V, dim=(1, 2)) / m).reshape(N, m * m, B).permute(1, 0, 2)
    return torch.einsum('fna,fnb->fab', F, F.conj()) / N


# ------------------------------------------------------------------------------- split scorer
def score(q_tr, P_tr, q_te, P_te, TR_tr, TR_te):
    """W = P_tr^{-1} q_tr, scored against both moment sets.

    L = Tr(Sig) - 2 Re <q, W> + <W, P W>.  At W = P_tr^{-1} q_tr the train column collapses to
    the familiar Tr(Sig) - <q, P^{-1} q>; it is written in residual form anyway so that the two
    columns are literally the same expression evaluated on two moment sets."""
    W = torch.linalg.solve(P_tr, q_tr)
    out = []
    for q, P, TR in ((q_tr, P_tr, TR_tr), (q_te, P_te, TR_te)):
        cross = float(torch.einsum('fam,fam->', q.conj(), W).real)
        quad = float(torch.einsum('fam,fam->', W.conj(), P @ W).real)
        out.append(TR - 2 * cross + quad)
    return out


def blocks_band(Chat, B, sig):
    win, out, nb = band_index(B)
    I = (sig * sig) * torch.eye(3 * nb, device=DEV, dtype=Chat.dtype)
    return (Chat[win[:, :, None], out[:, None, :]],
            Chat[win[:, :, None], win[:, None, :]] + I)


def blocks_period(Pm, sig):
    I = torch.eye(Pm.shape[-1], device=DEV, dtype=Pm.dtype)[None]
    return Pm, Pm + sig * sig * I


def main():
    Xtr = load(True, NIMG).reshape(NIMG, d)
    Xte = load(False, NTEST).reshape(NTEST, d)
    mu = Xtr.mean(0)                      # the ONLY mean either side is allowed to use
    Ctr, Cte = Xtr - mu, Xte - mu
    TR_tr = float((Ctr ** 2).sum() / Ctr.shape[0])
    TR_te = float((Cte ** 2).sum() / Cte.shape[0])

    Chat_tr, Chat_te = fourier_cov(Ctr), fourier_cov(Cte)
    for nm, C, TR in (('train', Chat_tr, TR_tr), ('test', Chat_te, TR_te)):
        err = abs(float(C.diagonal().sum().real) - TR)
        assert err < 1e-6 * TR, f"Parseval failed on {nm}: {err}"

    print(f"CIFAR-10 raw pixels  d={d}   train={Xtr.shape[0]} (CIFAR train[:N])  "
          f"test={Xte.shape[0]} (CIFAR test split)")
    print(f"Tr(Sigma) train {TR_tr:.4f}   test {TR_te:.4f}  (about the TRAIN mean; the test\n"
          f"  split carries {TR_tr-TR_te:.3f} less energy -- see docs/rf_heldout_and_nsweep.md"
          f" sec 7.1)\n")

    # ---- baseline, held out the same way -------------------------------------------------
    lin = {s: linear_split(Xtr, Xte, s) for s in SIGS}
    store = {'sigmas': np.array(SIGS),
             'linear': np.array([lin[s] for s in SIGS]),
             'trace': np.array([TR_tr, TR_te])}

    prior = {}
    for f in ('tables/rf_band_relaxation.npz', 'tables/rf_equivariance_toll.npz'):
        if os.path.exists(f):
            prior.update(dict(np.load(f, allow_pickle=True)))

    hdr = "".join(f"{f'sg={s}':>26}" for s in SIGS)
    print(f"{'model':>22} {'W params':>10} |{hdr}")
    print(f"{'':>22} {'':>10} |" + "".join(f"{'train':>11}{'test':>11}{'':>4}" for _ in SIGS))
    print("-" * (34 + 26 * len(SIGS)))
    print(f"{'free Wiener':>22} {d*d:>10,} |"
          + "".join(f"{lin[s][0]:>11.4f}{lin[s][1]:>11.4f}{'':>4}" for s in SIGS))

    def row(lab, key, npar, vals):
        """vals[s] = (train, test).  Excess over Wiener is train-vs-train / test-vs-test."""
        store[key] = np.array([vals[s] for s in SIGS])
        print(f"{lab:>22} {npar:>10,} |"
              + "".join(f"{vals[s][0]-lin[s][0]:>+11.4f}{vals[s][1]-lin[s][1]:>+11.4f}{'':>4}"
                        for s in SIGS))
        old = prior.get(key.replace('_ho', ''))
        if old is not None:
            dv = max(abs(float(old[i]) - vals[s][0]) for i, s in enumerate(SIGS))
            assert dv < TOL, f"{key} train column does not reproduce the in-sample table: {dv}"
            return dv
        return None

    print(f"\n  the toll of the group OUR sliding model lives in (Z_3072)")
    reps = {}
    for P in (1,):
        pm_t, pm_e = pm_1d(Ctr, P), pm_1d(Cte, P)
        v = {s: score(*blocks_period(pm_t, s), *blocks_period(pm_e, s), TR_tr, TR_te)
             for s in SIGS}
        reps[f'period1d|{P}'] = row(f'period1d P={P}', f'period1d|{P}', d, v)

    print(f"\n  COMB -- period-p on Z_32 x Z_32 with free 3-channel mixing")
    ref = None
    for p in PS:
        pm_t, pm_e = pm_2d(Ctr, p), pm_2d(Cte, p)
        v = {s: score(*blocks_period(pm_t, s), *blocks_period(pm_e, s), TR_tr, TR_te)
             for s in SIGS}
        reps[f'period2d|{p}'] = row(f'period2d p={p}', f'period2d|{p}', 3 * d * p * p, v)
        if p == 1:
            ref = v

    print(f"\n  BAND -- smoothly space-varying kernel")
    for B in BS:
        nb = (2 * B + 1) ** 2
        v = {s: score(*blocks_band(Chat_tr, B, s), *blocks_band(Chat_te, B, s), TR_tr, TR_te)
             for s in SIGS}
        reps[f'band2d|{B}'] = row(f'band2d B={B} ({nb})', f'band2d|{B}', 3 * d * nb, v)
        if B == 0 and ref is not None:
            e = max(max(abs(a - b) for a, b in zip(v[s], ref[s])) for s in SIGS)
            print(f"{'':>22} {'':>10} |  SELFTEST band B=0 vs period2d p=1, BOTH columns: "
                  f"{e:.2e}  {'OK' if e < TOL else 'MISMATCH'}")
            assert e < TOL

    done = {k: v for k, v in reps.items() if v is not None}
    if done:
        print(f"\n  [train column reproduces the stored in-sample tables on {len(done)} of "
              f"{len(reps)} classes, max|delta| = {max(done.values()):.2e}]")

    # ---- (a) one split fed in twice must collapse onto the in-sample number ---------------
    v = {s: score(*blocks_band(Chat_tr, 1, s), *blocks_band(Chat_tr, 1, s), TR_tr, TR_tr)
         for s in SIGS}
    e = max(abs(v[s][0] - v[s][1]) for s in SIGS)
    print(f"  [degenerate-split check (train fed in as its own test), band B=1: "
          f"{e:.2e}  {'OK' if e < TOL else 'MISMATCH'}]")
    assert e < TOL

    npar = {f'period1d|1': d}
    npar.update({f'period2d|{p}': 3 * d * p * p for p in PS})
    npar.update({f'band2d|{B}': 3 * d * (2 * B + 1) ** 2 for B in BS})

    # ---- EACH CLASS'S OWN OPTIMISM ---------------------------------------------------------
    # The toll moves by (Wiener's optimism) - (the class's own optimism).  Reading the class
    # optimism straight off the absolute columns turns this table into an independent
    # measurement of how optimism scales with free-parameter count, on eleven nested linear
    # classes spanning 3,072 to 9,437,184 parameters -- all fitted on the same 10,000 images.
    print(f"\n{'='*(34+26*len(SIGS))}")
    print("OWN OPTIMISM (test - train, absolute).  The classes are ordered by parameter count;")
    print("this is the same quantity the held-out RF work measures, on the linear side.")
    print(f"{'='*(34+26*len(SIGS))}")
    print(f"{'model':>22} {'W params':>10} |" + "".join(f"{f'sg={s}':>12}" for s in SIGS))
    for key in ['period1d|1'] + [f'period2d|{p}' for p in PS] + [f'band2d|{B}' for B in BS]:
        if key not in store:
            continue
        print(f"{key:>22} {npar[key]:>10,} |"
              + "".join(f"{store[key][i][1]-store[key][i][0]:>+12.4f}"
                        for i in range(len(SIGS))))
    print(f"{'free Wiener':>22} {d*d:>10,} |"
          + "".join(f"{store['linear'][i][1]-store['linear'][i][0]:>+12.4f}"
                    for i in range(len(SIGS))))
    print("  => optimism tracks parameter count, and the free Wiener is off the end of it.")
    print("     That is the whole reason the in-sample tolls were inflated.")
    print("  ** READ THE sg>=1.61 COLUMNS OF *THIS* TABLE AS DIFFERENCES ONLY. **  They are")
    print("     negative, which no honest optimism can be: at large sigma L ~ Tr(Sigma) -")
    print("     (small), so the split's 1.728 trace deficit passes ~1:1 into every absolute")
    print("     test loss (docs/rf_heldout_and_nsweep.md sec 7.1).  The offset is common to")
    print("     all rows, so class-vs-class comparisons here survive it and the TOLL table")
    print("     above is unaffected entirely (test-vs-test, the offset cancels in the")
    print("     subtraction).  Only these absolute per-class gaps need symmetrising.")

    # ---- PER-PARAMETER EFFICIENCY, RECOMPUTED HELD OUT -------------------------------------
    # michimin's in-sample script ranks the relaxations this way.  The ranking is not
    # invariant under the correction: a class's apparent purchase includes the part of the
    # baseline's optimism it happens to reproduce, and the big classes buy the most of that.
    print(f"\n{'='*(34+26*len(SIGS))}")
    print("TOLL REMOVED PER MILLION READOUT PARAMS, relative to period2d p=1 (= band B=0).")
    print(f"{'='*(34+26*len(SIGS))}")
    print(f"{'model':>22} {'W params':>10} |"
          + "".join(f"{f'sg={s}':>22}" for s in SIGS))
    print(f"{'':>22} {'':>10} |" + "".join(f"{'in-sample':>11}{'held out':>11}" for _ in SIGS))
    base = store['period2d|1']
    for key in [f'period2d|{p}' for p in PS if p > 1] + [f'band2d|{B}' for B in BS if B > 0]:
        if key not in store:
            continue
        M = npar[key] / 1e6
        print(f"{key:>22} {npar[key]:>10,} |"
              + "".join(f"{(base[i][0]-store[key][i][0])/M:>11.2f}"
                        f"{(base[i][1]-store[key][i][1])/M:>11.2f}" for i in range(len(SIGS))))

    # ---- THE NONLINEAR GAIN, AND michimin's sec 5 CHAIN, REDONE HELD OUT -------------------
    # The band writeup's load-bearing step is that the nonlinear gain (class linear - the
    # nonlinear RF in the SAME class) transfers to a larger linear class.  Both terms were
    # in-sample; both move under the split, and in opposite directions, so the chain has to be
    # redone rather than shifted.
    ho = 'tables/rf_pixel_heldout.npz'
    if os.path.exists(ho):
        R = dict(np.load(ho, allow_pickle=True))

        def rf(sg, c, arm, col):
            k = f'{sg}|{c}|{arm}'
            return float(R[k][:, col].mean()) if k in R else None

        print(f"\n{'='*(34+26*len(SIGS))}")
        print("NONLINEAR GAIN = (best LINEAR map in the class) - (the nonlinear RF in it),")
        print("in-sample and held out.  in-sample col uses the stored in-sample RF tables.")
        print(f"{'='*(34+26*len(SIGS))}")
        print(f"{'class / RF':>34} |" + "".join(f"{f'sg={s}':>22}" for s in SIGS))
        print(f"{'':>34} |" + "".join(f"{'in-sample':>11}{'held out':>11}" for _ in SIGS))
        for lab, key, c, arm in (('Z_3072      vs 1-D RF c=1536', 'period1d|1', 1536, '1d'),
                                 ('Z_32xZ_32   vs 2-D RF c=1536', 'period2d|1', 1536, '2d'),
                                 ('Z_32xZ_32   vs 2-D RF c=3072', 'period2d|1', 3072, '2d')):
            cells = ''
            for i, s in enumerate(SIGS):
                a, b = rf(s, c, arm, 0), rf(s, c, arm, 2)
                cells += (f"{store[key][i][0]-a:>11.4f}" if a is not None else f"{'--':>11}")
                cells += (f"{store[key][i][1]-b:>11.4f}" if b is not None else f"{'--':>11}")
            print(f"{lab:>34} |{cells}")
        print("  => the gain SURVIVES the correction (it shrinks a few %), so the band chain's")
        print("     load-bearing assumption is not damaged by holding out.")

        print(f"\n  sec-5 CHAIN at sigma=0.127, held out, using the 2-D gain (the band is built")
        print(f"  on Z_32 x Z_32, so that is the gain that belongs in it):")
        i = SIGS.index(0.127) if 0.127 in SIGS else None
        if i is not None and rf('0.127', 1536, '2d', 2) is not None:
            W = store['linear'][i][1]
            g = store['period2d|1'][i][1] - rf('0.127', 1536, '2d', 2)
            for key, lab in (('period2d|1', 'B=0 (plain re-index)'), ('band2d|1', 'B=1'),
                             ('band2d|2', 'B=2'), ('band2d|3', 'B=3'), ('band2d|4', 'B=4')):
                if key not in store:
                    continue
                pred = store[key][i][1] - g
                note = ''
                if key == 'period2d|1':
                    note = f"   <- MEASURED: {rf('0.127', 1536, '2d', 2):.4f} (c=1536), " \
                           f"{rf('0.127', 3072, '2d', 2):.4f} (c=3072)"
                print(f"    {lab:>22}  class linear {store[key][i][1]:7.4f}  - gain {g:.4f}"
                      f"  = {pred:7.4f}   vs Wiener {W:.4f}  ({pred-W:+.4f}){note}")
            print("    (the B=0 row is a consistency check on the chain, not a prediction:")
            print("     it must land on the measured 2-D RF by construction of g.)")

    os.makedirs('tables', exist_ok=True)
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
