"""DOES CONDITIONING SHRINK THE 2-D EQUIVARIANCE TOLL?  Held out, linear, closed form.

    michimin, 2026-09-25: "write a script to calculate the conditional 2d toll to see if
    conditioning will shrink the toll ... WRITE THE CODE SCRIPT AND DOCUMENTATION ONLY
    DON'T RUN ANYTHING. i'll run them on the clusters"

    STATUS: written 2026-09-25, NEVER EXECUTED.  Every number it prints is new.

WHAT IS MEASURED
----------------
The toll of a readout class is  toll(B) = floor(B) - Wiener :  the best LINEAR denoiser whose
readout is band-B on Z_32 x Z_32 (free 3-channel mixing, scripts/rf_band_relaxation.py), minus
the free d x d optimum.  It is an exact function of second moments -- no random features.
Here it is priced for three ways of telling a linear denoiser the class, every model FIT ON
train[:NIMG] AND SCORED ON THE CIFAR TEST SPLIT with the test images' own labels:

    U  unconditional       D = A y + b                 one mean, one covariance
                                                       (this column IS the existing ladder)
    S  class mean known    D = A y + V U + b           one shared slope A; minimising over V, b
                             = A y + beta_class        Schur-complements every moment on U,
                                                       i.e. POOLED WITHIN-CLASS moments
    C  per class           D = A_c y + b_c             one slope per class; per-class moments,
                                                       losses averaged with the class weights

For X in {U, S, C}:  band_X(B) = the best band-B slope inside family X,  W_X = the free optimum
of family X,  toll_X(B) = band_X(B) - W_X.

WHICH NUMBER ANSWERS WHICH QUESTION
-----------------------------------
  toll_S / toll_U   THE ONE FOR THE CONDITIONAL RF.  core/rf_circulant2d{,_band}.py with
                    class_centre=True has readout band-B + V U, so S is its linear class and
                    band_S(B) is the baseline its nonlinear gain is measured against -- exactly
                    as band_U(B) is for the unconditional RF.  < 1 => conditioning shrinks the
                    toll the RF has to pay.
  toll_C / toll_U   the same for per-class slopes (a per-class RF = 10 unconditional fits on
                    ~1000 images each; not implemented).
  eff               EFFECTIVE toll = [min(band_S, band_C) - min(W_S, W_C)] / toll_U: what an
                    equivariant conditional model still owes against the BEST conditional
                    linear denoiser, whichever style that is.
  band_S - W_U      < 0 => the conditional band class beats the UNCONDITIONAL optimum before
                    any nonlinearity: headroom the label hands the RF for free.
  A_S, A_C          W_U - W_S, W_U - W_C: the linear headroom conditioning creates at all.

HELD-OUT CONVENTIONS -- those of scripts/rf_band_relaxation_heldout.py, extended to classes.
Every mean is a model parameter, so it comes from TRAIN and centres BOTH splits: U by mu_train,
S by the train mean of each image's OWN class (test images by their own labels), C by the
class's train mean.  The noise is split-independent, so P = (signal block) + sigma^2 I on both
sides and there is no ridge.  `fourier_cov`, `blocks_band` and `score` are imported from that
script unchanged, and the Wiener arithmetic is linear_split's, so U is the ladder by
construction and S / C differ from it ONLY in which rows go in and which means centre them.

VALIDATION (asserted -- the run stops rather than print a plausible wrong table)
  (a) U reproduces tables/rf_band_relaxation_heldout{,_sigmid,_sig2212}.npz -- band2d|B and
      the Wiener, BOTH columns, matched on the SIGMA VALUE -- wherever a cell exists
      (only at the ladder's own NIMG = NTEST = 10000).
  (b) the factored Wiener reproduces linear_split() on U.
  (c) ONE class (every label 0): S and C collapse onto U, band and Wiener, both columns.
      This exercises the class-mean / label-indexing / class-weight plumbing end to end.
  (d) Parseval on every Fourier covariance, and image<->label alignment on three images per
      split against the raw dataset.

COST.  One GPU, minutes.  12 d x d eigh (U, S, 10 classes), 12 band solves per (sigma, B), the
largest being 1024 blocks of 243 x 243 at B=4; peak ~6 GB.

    python scripts/rf_cond_toll2d_heldout.py
    SIGS=0.621,0.853,1.172 BS=0,1,2,3,4 python scripts/rf_cond_toll2d_heldout.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torchvision

from scripts.rf_pixel_heldout import load, linear_split, DEV, DT, d
from scripts.rf_band_relaxation_heldout import fourier_cov, score, blocks_band

SIGS = [float(x) for x in
        os.environ.get('SIGS', '0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0').split(',')]
BS = [int(x) for x in os.environ.get('BS', '0,1,2,3,4').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
OUT = os.environ.get('OUT', 'tables/rf_cond_toll2d_heldout.npz')
ROOT = '/n/home12/binxuwang/.keras/datasets'      # the root scripts/rf_pixel_heldout.load reads
LADDER = ('tables/rf_band_relaxation_heldout.npz',
          'tables/rf_band_relaxation_heldout_sigmid.npz',
          'tables/rf_band_relaxation_heldout_sig2212.npz')
NCLS = 10
TOL = 1e-8


# ------------------------------------------------------------------------------ data + labels
def load_labels(train, n, X=None):
    """CIFAR-10 labels in the order scripts/rf_pixel_heldout.load returns images (a sequential
    DataLoader over the same dataset), as a LongTensor on DEV.  Given load()'s images X, three
    of them are checked against the raw dataset, so an image/label misalignment -- which would
    silently shuffle every class mean -- fails loudly instead."""
    ds = torchvision.datasets.CIFAR10(ROOT, train=train, download=False)
    y = torch.as_tensor(np.asarray(ds.targets[:n], dtype=np.int64), device=DEV)
    assert y.shape[0] == n, f"only {y.shape[0]} labels for {n} images"
    if X is not None:
        for i in (0, n // 2, n - 1):
            ref = torch.as_tensor(ds.data[i], device=DEV).permute(2, 0, 1).to(DT) / 255.0
            err = float((X[i].reshape(ref.shape) - ref).abs().max())
            assert err < 1e-6, f"{'train' if train else 'test'} image {i} does not match " \
                               f"the dataset entry its label comes from (max|d| = {err:.2e})"
    return y


def class_means(X, y, ncls):
    """(ncls, d) class means of the rows of X and the class counts, via a one-hot matmul."""
    oh = torch.nn.functional.one_hot(y, ncls).to(X.dtype)
    cnt = oh.sum(0)
    assert bool((cnt > 0).all()), f"class(es) with no TRAIN image: " \
                                  f"{torch.nonzero(cnt == 0).flatten().tolist()}"
    return (oh.T @ X) / cnt[:, None], cnt


# --------------------------------------------------------------------------------- one family
class Wiener:
    """The free linear optimum of one family, fitted on TRAIN rows already centred by the
    family's train means and scored on test rows centred by the SAME means.  The arithmetic
    of scripts/rf_pixel_heldout.py::linear_split (asserted in main) with the eigh factored out
    of the sigma loop -- it is the only expensive step, and 12 of them are needed."""

    def __init__(self, Ctr):
        self.ev, self.V = torch.linalg.eigh((Ctr.T @ Ctr) / Ctr.shape[0])

    def split(self, Cte, sg):
        shr = self.ev / (self.ev + sg ** 2)
        train = float((sg ** 2 * shr).sum())
        Z = Cte @ self.V
        resid = float(((Z * (1.0 - shr)) ** 2).sum() / Cte.shape[0])
        noise = float((sg ** 2) * (shr ** 2).sum())
        return train, resid + noise


def fit_family(Ctr, Cte, sigs, bs):
    """Held-out band floors and free optimum of ONE family from its centred (n, d) rows.
    Returns ({('W', s): (train, test), (B, s): (train, test)}, (Tr_train, Tr_test)).
    With bs empty only the Wiener is computed (no Fourier covariance is formed)."""
    TR_tr = float((Ctr ** 2).sum() / Ctr.shape[0])
    TR_te = float((Cte ** 2).sum() / Cte.shape[0])
    wie = Wiener(Ctr)
    res = {('W', s): wie.split(Cte, s) for s in sigs}
    del wie
    if bs:
        Ch_tr, Ch_te = fourier_cov(Ctr), fourier_cov(Cte)
        for nm, Ch, TR in (('train', Ch_tr, TR_tr), ('test', Ch_te, TR_te)):
            err = abs(float(Ch.diagonal().sum().real) - TR)
            assert err < 1e-6 * TR, f"Parseval failed on {nm}: {err}"
        for s in sigs:
            for B in bs:
                res[(B, s)] = tuple(score(*blocks_band(Ch_tr, B, s),
                                          *blocks_band(Ch_te, B, s), TR_tr, TR_te))
        del Ch_tr, Ch_te
    torch.cuda.empty_cache()
    return res, (TR_tr, TR_te)


def conditional_families(Xtr, ytr, Xte, yte, ncls, sigs, bs, verbose=True):
    """Families S (pooled within-class, class mean known) and C (per class), held out.

    Xtr, Xte : (n, d) images;  ytr, yte : (n,) class ids, each split its OWN labels.
    Returns {'S': res, 'C': res (class-weighted: pi_train on the train column, pi_test on the
    test column, i.e. the average over each split's images), 'Ck': [res per class],
    'trace_S', 'trace_C', 'pi_train', 'pi_test'} with res keyed as in fit_family."""
    mu_c, ntr = class_means(Xtr, ytr, ncls)
    nte = torch.bincount(yte, minlength=ncls)
    assert bool((nte > 0).all()), "a class has no TEST image: its per-class loss is undefined"
    pi_tr = (ntr / ntr.sum()).tolist()
    pi_te = (nte.to(DT) / nte.sum()).tolist()
    t0 = time.time()
    RS, trS = fit_family(Xtr - mu_c[ytr], Xte - mu_c[yte], sigs, bs)
    if verbose:
        print(f"  S (pooled within-class) done  {time.time() - t0:.0f}s", flush=True)
    RC, RCk, trC = {}, [], [0.0, 0.0]
    for k in range(ncls):
        t0 = time.time()
        r, tr = fit_family(Xtr[ytr == k] - mu_c[k], Xte[yte == k] - mu_c[k], sigs, bs)
        RCk.append(r)
        for key, (a, b) in r.items():
            ra, rb = RC.get(key, (0.0, 0.0))
            RC[key] = (ra + pi_tr[k] * a, rb + pi_te[k] * b)
        trC[0] += pi_tr[k] * tr[0]
        trC[1] += pi_te[k] * tr[1]
        if verbose:
            print(f"  C class {k}: {int(ntr[k])} train / {int(nte[k])} test  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {'S': RS, 'C': RC, 'Ck': RCk, 'trace_S': trS, 'trace_C': tuple(trC),
            'pi_train': pi_tr, 'pi_test': pi_te}


# ------------------------------------------------------------------------------------ report
def report(R, j, title):
    """j = 1: the held-out (test) column, j = 0: in-sample."""
    U, S, C = R['U'], R['S'], R['C']
    bar = '=' * 104
    print(f"\n{bar}\n{title}\n{bar}")
    print(f"{'sigma':>7} | {'W_U':>9} {'W_S':>9} {'W_C':>9} | {'A_S = W_U-W_S':>14} "
          f"{'A_C = W_U-W_C':>14}")
    for s in SIGS:
        wu, ws, wc = U[('W', s)][j], S[('W', s)][j], C[('W', s)][j]
        print(f"{s:>7.3f} | {wu:9.4f} {ws:9.4f} {wc:9.4f} | {wu - ws:+14.4f} {wu - wc:+14.4f}")
    print(f"\n  toll_X(B) = band_X(B) - W_X, same family both sides; ratios are over toll_U(B)")
    print(f"{'sigma':>7} {'B':>2} | {'toll_U':>8} | {'toll_S':>8} {'S/U':>6} | "
          f"{'toll_C':>8} {'C/U':>6} | {'eff':>6} | {'band_S-W_U':>10} {'band_C-W_U':>10}")

    def ratio(v, t):
        return f"{v / t:6.2f}" if abs(t) > 1e-12 else f"{'n/a':>6}"

    for s in SIGS:
        for B in BS:
            tU = U[(B, s)][j] - U[('W', s)][j]
            tS = S[(B, s)][j] - S[('W', s)][j]
            tC = C[(B, s)][j] - C[('W', s)][j]
            eff = min(S[(B, s)][j], C[(B, s)][j]) - min(S[('W', s)][j], C[('W', s)][j])
            print(f"{s:>7.3f} {B:>2} | {tU:8.4f} | {tS:8.4f} {ratio(tS, tU)} | "
                  f"{tC:8.4f} {ratio(tC, tU)} | {ratio(eff, tU)} | "
                  f"{S[(B, s)][j] - U[('W', s)][j]:+10.4f} {C[(B, s)][j] - U[('W', s)][j]:+10.4f}")
        print()
    print("  S/U, C/U < 1      => conditioning SHRINKS the toll of the band-B readout")
    print("  eff               => toll still owed against the BEST conditional linear (S or C)")
    print("  band_X - W_U < 0  => the conditional band class beats the UNCONDITIONAL optimum")
    print("                       before any nonlinearity")


# -------------------------------------------------------------------------------------- main
def main():
    t00 = time.time()
    Xtr = load(True, NIMG).reshape(NIMG, d)
    Xte = load(False, NTEST).reshape(NTEST, d)
    ytr = load_labels(True, NIMG, Xtr)
    yte = load_labels(False, NTEST, Xte)
    print(f"CIFAR-10 raw pixels  d={d}   train={NIMG} (CIFAR train[:N])  test={NTEST} "
          f"(CIFAR test split), labels aligned (checked)")
    print(f"  train class counts {torch.bincount(ytr, minlength=NCLS).tolist()}")
    print(f"  test  class counts {torch.bincount(yte, minlength=NCLS).tolist()}")
    print(f"  sigmas {SIGS}   band orders {BS}\n", flush=True)

    # ---- U: the unconditional ladder, recomputed through the same code path ----------------
    t0 = time.time()
    mu = Xtr.mean(0)                     # the ONLY mean U is allowed to use, both splits
    RU, trU = fit_family(Xtr - mu, Xte - mu, SIGS, BS)
    print(f"  U (unconditional) done  {time.time() - t0:.0f}s", flush=True)

    # (b) the factored Wiener is linear_split
    lt = linear_split(Xtr, Xte, SIGS[0])
    e = max(abs(a - b) for a, b in zip(lt, RU[('W', SIGS[0])]))
    print(f"  [check b] factored Wiener vs linear_split at sigma={SIGS[0]}: {e:.1e}", flush=True)
    assert e < 1e-9, f"Wiener helper does not reproduce linear_split: {e}"

    # (a) U reproduces the stored held-out ladder, cell for cell, matched on the sigma VALUE
    if NIMG == 10000 and NTEST == 10000:
        ladder = {}
        for f in LADDER:
            if not os.path.exists(f):
                continue
            z = dict(np.load(f, allow_pickle=True))
            for i, sv in enumerate(np.asarray(z['sigmas']).ravel()):
                ladder[('W', float(sv))] = np.asarray(z['linear'][i], float)
                for B in BS:
                    if f'band2d|{B}' in z:
                        ladder[(B, float(sv))] = np.asarray(z[f'band2d|{B}'][i], float)
        hit = [k for k in RU if k in ladder]
        if hit:
            dv = max(float(np.abs(np.asarray(RU[k]) - ladder[k]).max()) for k in hit)
            print(f"  [check a] U reproduces the stored held-out ladder on {len(hit)} of "
                  f"{len(RU)} cells: max|delta| = {dv:.1e}", flush=True)
            assert dv < TOL, f"U does not reproduce the held-out ladder: {dv}"
        else:
            print("  [check a] no stored ladder cell overlaps this sigma/B grid: skipped")
    else:
        print("  [check a] skipped: the stored ladder is at NIMG = NTEST = 10000")

    # ---- S and C ------------------------------------------------------------------------------
    RSC = conditional_families(Xtr, ytr, Xte, yte, NCLS, SIGS, BS)

    # (c) ONE class: S and C must collapse onto U (exercises all the class plumbing)
    s0, b0 = SIGS[0], BS[0]
    one = conditional_families(Xtr, torch.zeros_like(ytr), Xte, torch.zeros_like(yte), 1,
                               [s0], [b0], verbose=False)
    e = max(abs(one[f][k][j] - RU[k][j]) for f in ('S', 'C') for k in (('W', s0), (b0, s0))
            for j in (0, 1))
    print(f"  [check c] one class: S and C == U at sigma={s0} B={b0}, both columns: {e:.1e}",
          flush=True)
    assert e < TOL, f"one-class S/C do not collapse onto U: {e}"

    R = {'U': RU, 'S': RSC['S'], 'C': RSC['C']}
    report(R, 1, "HELD OUT -- fitted on train, scored on the CIFAR test split (THE number)")
    report(R, 0, "IN-SAMPLE -- approximation only; the per-class C column is badly optimistic "
                 "(~1000 images per class against d = 3072)")

    store = {'sigmas': np.array(SIGS), 'bs': np.array(BS),
             'trace_U': np.array(trU), 'trace_S': np.array(RSC['trace_S']),
             'trace_C': np.array(RSC['trace_C']),
             'pi_train': np.array(RSC['pi_train']), 'pi_test': np.array(RSC['pi_test'])}
    for fam, res in (('U', RU), ('S', RSC['S']), ('C', RSC['C'])):
        store[f'{fam}|W'] = np.array([res[('W', s)] for s in SIGS])
        for B in BS:
            store[f'{fam}|band2d|{B}'] = np.array([res[(B, s)] for s in SIGS])
    for k, res in enumerate(RSC['Ck']):
        store[f'C{k}|W'] = np.array([res[('W', s)] for s in SIGS])
        for B in BS:
            store[f'C{k}|band2d|{B}'] = np.array([res[(B, s)] for s in SIGS])
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}  (every value is a [train, test] pair per sigma)   "
          f"{time.time() - t00:.0f}s total")


if __name__ == '__main__':
    main()
