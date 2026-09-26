"""DENSE RF, HELD OUT ON THE SAME 10,000 CIFAR TEST IMAGES (michimin, 2026-09-23 05:44).

WHY THIS ARM MATTERS MORE THAN THE OTHER TWO
--------------------------------------------
The circulant arms were held out earlier today and both PASSED: their own generalisation
gaps are +0.02 .. +0.10, six to twenty times below the Wiener baseline's +0.459.  The dense
arm is the one with a standing reason to FAIL.  On 2026-09-21 the N-sweep caught it scoring
by memorisation -- its advantage over linear collapsed

    k/d = 4, sigma = 0.127:   0.825 -> 0.336 -> 0.039    as N went 10k -> 20k -> 40k

with lam = 1e-6 doing nothing to stop it.  k/d = 8 puts 24,576 features and 75.5M trained
parameters against 10^4 images.  Every dense number this project has published
(tables/rf_pixel_dense_sweep.npz, tables/rf_pixel_rowmult.npz) is in-sample, and the
dense-beats-circulant crossing at k/d ~ 1.6-2.7 is measured entirely on that side.  So this
is the run that decides whether that crossing is a property of the model class or of N=10^4.

THE ESTIMATOR
-------------
The dense readout is solved on train moments and then scored, unchanged, against test
moments:

    W    = Cov_tr (Sig_tr + lam I)^{-1}                                (d, k)
    beta = mu_tr - W gmean_tr
    L_test = E_test || x0 - W phi(y) - beta ||^2
           = Tr_te - 2 Tr(W Cov_te^T) + Tr(W Sig_te W^T)

with EVERY test moment centred by the TRAIN means:

    Tr_te  = E_te || x0 - mu_tr ||^2
    Cov_te = E_te[(x0 - mu_tr)(phi - gmean_tr)^T]
    Sig_te = E_te[(phi - gmean_tr)(phi - gmean_tr)^T]

Re-centring on the test split would silently refit d + k parameters at evaluation time; beta
is part of the model, so it crosses the split frozen, exactly as in the circulant estimator.
The expectation over the noise z stays analytic (Stein / Hermite) as everywhere else in this
project -- the split is over IMAGES, not over noise draws.

TWO IN-SAMPLE LOSSES, AS EVER.  The stored dense tables hold

    L_stored = Tr - Tr(Cov (Sig + lam I)^{-1} Cov^T),

which is NOT the achieved residual of W: it is shifted by exactly lam ||W||^2.  So three
columns come back -- `train` (= L_stored, the one the existing tables must be differenced
against, and which is asserted to reproduce them), `train_resid` (the residual form of the
same W), and `test`.  Only `test - train_resid` is an honest gap.

THE BASELINE IS HELD OUT TOO (linear_split, imported from rf_pixel_heldout).  Sigma at
d = 3072 is 4.7M free parameters fitted on the same 10^4 images; scoring a held-out RF
against an in-sample Wiener invents a gap 19x the real one, in the direction that flatters
linear.

VALIDATION -- `SELFTEST=1 python scripts/rf_pixel_dense_heldout.py`:
  (a) DEGENERATE SPLIT at lam = 0: feed the train split in as its own test.  L_test must
      collapse exactly onto L_train_resid and onto L_stored.  Exact to machine precision,
      and it pins the sign and normalisation of every term with no reference code at all.
  (b) BRUTE FORCE by Monte Carlo over explicit noise draws with the explicit (W, beta), on a
      test split drawn from a SHIFTED and RESCALED distribution so the two moment sets
      genuinely differ.  This is the check that the train-mean centring is right -- a
      re-centring bug is invisible to (a) and large here.
  (c) EXACT-DIAGONAL AGREEMENT with core.rf_gmm_estimators_torch.stein_covariances_t: the
      refactored moment builder must reproduce the shipped one bit for bit on the train side.
  (d) the driver ASSERTS its train column reproduces tables/rf_pixel_dense_sweep.npz.

    SELFTEST=1 python scripts/rf_pixel_dense_heldout.py
    JS=2,4,8 SIGS=0.127,0.452 python scripts/rf_pixel_dense_heldout.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from core.rf_gmm_estimators_torch import _c0_t, _ndtr, _npdf
from scripts.rf_pixel_heldout import load, linear_split, DEV, DT, d

LAM = float(os.environ.get('LAM', '1e-6'))
JS = [float(x) for x in os.environ.get('JS', '2,3,4,6,8').split(',')]
SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610,5.0').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
NSEED = int(os.environ.get('NSEED', '2'))
OUT = os.environ.get('OUT', 'tables/rf_pixel_dense_heldout.npz')


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------
def stein_moments(X0, Theta, sigma, mu=None, gmean=None, *, lab=None, gam=None,
                  class_centre=False):
    """Noise-analytic moments of the relu features about GIVEN centres.

    Returns (Cov, Sig, tr, mu, gmean) with, writing phi = relu(Theta y), y = x0 + sigma z,

        Cov = E[(x0 - mu)(phi - gmean)^T]          (d, k)
        Sig = E[(phi - gmean)(phi - gmean)^T]      (k, k)   -- NO ridge
        tr  = E||x0 - mu||^2

    mu / gmean default to this sample's own means, which is the in-sample convention and
    reproduces stein_covariances_t exactly.  Passing the TRAIN means is what makes the test
    column a held-out score rather than a refit.

    The z-expectation is exact.  Split phi = c0(M, s) + eta with M = Theta x0,
    s = sigma ||theta||, E_z[eta] = 0.  Then

        Cov = E_n[(x0 - mu) (c0 - gmean)^T]
        Sig = E_n[(c0 - gmean)(c0 - gmean)^T]  +  E_n[Cov_z(phi | x0_n)]

    and the SECOND term does not depend on the centre at all -- which is exactly why a
    train-centred Sig_te is still exact rather than an approximation.  Cov_z is the Mehler
    sum rho c1c1 + 2 rho^2 c2c2 + 6 rho^3 c3c3 off the diagonal, replaced on the diagonal by
    its exact value E[relu^2] - c0^2.
    """
    N, k = X0.shape[0], Theta.shape[0]
    cid = None
    if lab is not None:
        cid = lab if lab.ndim == 1 else torch.argmax(lab, dim=1)
        cid = cid.to(device=X0.device, dtype=torch.long)
    if gam is not None and cid is None:
        raise ValueError("gam needs lab")
    if class_centre and cid is None:
        raise ValueError("class_centre needs lab")
    if mu is None:
        if class_centre:
            nc = int(gam.shape[1]) if gam is not None else int(cid.max()) + 1
            cnt = torch.bincount(cid, minlength=nc)
            if bool((cnt == 0).any()):
                raise ValueError("every class must occur in the training split")
            mu = torch.stack([X0[cid == q].mean(0) for q in range(nc)])
        else:
            mu = X0.mean(0)
    s = sigma * torch.linalg.norm(Theta, dim=1)                       # (k,)
    M = X0 @ Theta.T                                                  # (N, k)
    if gam is not None:
        M = M + gam[:, cid].T
    G = _c0_t(M, s[None, :])                                          # E_z[phi]
    if gmean is None:
        if class_centre:
            gmean = torch.stack([G[cid == q].mean(0) for q in range(mu.shape[0])])
        else:
            gmean = G.mean(0)
    Xc = X0 - (mu[cid] if class_centre else mu)
    Gc = G - (gmean[cid] if class_centre else gmean)
    Cov = Xc.T @ Gc / N
    tr = float((Xc ** 2).sum() / N)

    z = M / torch.clamp(s[None, :], min=1e-12)
    Phi_z = _ndtr(z); phi_z = _npdf(z)
    del z
    C1 = s[None, :] * Phi_z
    C2 = s[None, :] * phi_z / 2.0
    C3 = -M * phi_z / 6.0
    E_phi_sq = (M ** 2 + s[None, :] ** 2) * Phi_z + M * s[None, :] * phi_z
    diag_noise = (E_phi_sq - G ** 2).mean(0)                          # exact Var_z, mean_n
    del E_phi_sq, Phi_z, phi_z, M

    del G
    Sig = Gc.T @ Gc / N                                               # data block
    del Gc
    diag_data = Sig.diagonal().clone()                                # survives the overwrite

    Tn = Theta / torch.linalg.norm(Theta, dim=1)[:, None]
    rho = torch.clamp(Tn @ Tn.T, -1 + 1e-6, 1 - 1e-6)
    del Tn
    R = rho.clone()                                                   # rho^1
    A = C1.T @ C1 / N; del C1
    A *= R; Sig += A; del A
    R *= rho                                                          # rho^2
    A = C2.T @ C2 / N; del C2
    A *= R; A *= 2.0; Sig += A; del A
    R *= rho                                                          # rho^3
    A = C3.T @ C3 / N; del C3
    A *= R; A *= 6.0; Sig += A; del A
    del R, rho

    # the exact diagonal replaces the Hermite truncation of the NOISE block only, so the
    # data block's diagonal has to be put back alongside it.
    Sig.diagonal().copy_(diag_data + diag_noise)
    return Cov, Sig, tr, mu, gmean


def split_losses(Xtr, Xte, Theta, sigma, lam=LAM, want_model=False, *, lab=None,
                 gam=None, lab_test=None, class_centre=False):
    """Solve the dense readout on train, score it on train and on the held-out split.

    Returns (L_stored, L_train_resid, L_test).  L_stored is what the existing in-sample
    tables hold (ridge inside the explained term); L_train_resid is the achieved residual of
    the same W.  They differ by exactly lam ||W||^2.
    """
    k = Theta.shape[0]
    Cov, Sig, tr, mu, gmean = stein_moments(
        Xtr, Theta, sigma, lab=lab, gam=gam, class_centre=class_centre)
    Sl = Sig + lam * torch.eye(k, device=DEV, dtype=DT)
    W = torch.linalg.solve(Sl, Cov.T).T                               # (d, k) = Cov Sl^-1
    del Sl
    cross = float((W * Cov).sum())
    L_stored = tr - cross                                             # = tr - Cov Sl^-1 Cov^T
    L_res = tr - 2.0 * cross + float(((W @ Sig) * W).sum())
    del Cov, Sig
    torch.cuda.empty_cache()

    Cov_te, Sig_te, tr_te, _, _ = stein_moments(
        Xte, Theta, sigma, mu, gmean, lab=lab_test, gam=gam,
        class_centre=class_centre)
    L_test = (tr_te - 2.0 * float((W * Cov_te).sum())
              + float(((W @ Sig_te) * W).sum()))
    del Cov_te, Sig_te
    torch.cuda.empty_cache()
    if want_model:
        return L_stored, L_res, L_test, W, mu, gmean
    del W
    return L_stored, L_res, L_test


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def selftest():
    from core.rf_gmm_estimators_torch import stein_covariances_t
    torch.manual_seed(0)
    ok = True

    print("(c) refactored moment builder vs core.stein_covariances_t, in-sample centring")
    for (dd, kk, NN, sg) in [(10, 7, 400, 0.6), (16, 11, 250, 1.4), (8, 13, 300, 0.3)]:
        g = torch.Generator(device='cpu').manual_seed(dd * 31 + kk)
        X = (torch.randn(NN, dd, generator=g) ** 3 * 0.5 + 0.7).to(DEV, DT)
        Th = (torch.randn(kk, dd, generator=g) / np.sqrt(dd)).to(DEV, DT)
        U = torch.zeros(NN, 1, dtype=DT, device=DEV)
        Ga = torch.zeros(kk, 1, dtype=DT, device=DEV)
        Cr, Sr, trr = stein_covariances_t(X, U, Th, Ga, sg, 0.0, conditional=False,
                                          device=DEV, dtype=DT)
        Cm, Sm, trm, _, _ = stein_moments(X, Th, sg)
        e = max(float((Cr - Cm).abs().max() / Cr.abs().max()),
                float((Sr - Sm).abs().max() / Sr.abs().max()),
                abs(trr - trm) / abs(trr))
        print(f"    d={dd:3d} k={kk:3d} N={NN:4d} sigma={sg}   rel {e:.3e}")
        ok &= e < 1e-13

    print("(a) degenerate split at lam=0: L_test must collapse onto L_train_resid")
    for (dd, kk, NN, sg) in [(10, 7, 400, 0.6), (16, 11, 250, 1.4), (8, 13, 300, 0.3)]:
        g = torch.Generator(device='cpu').manual_seed(dd + kk)
        X = (torch.randn(NN, dd, generator=g) ** 3 * 0.5 + 0.7).to(DEV, DT)
        Th = (torch.randn(kk, dd, generator=g) / np.sqrt(dd)).to(DEV, DT)
        a, b, c = split_losses(X, X, Th, sg, lam=0.0)
        print(f"    d={dd:3d} k={kk:3d} N={NN:4d} sigma={sg}   "
              f"stored {a:.10f}  resid {b:.10f}  test {c:.10f}   "
              f"|test-resid| {abs(c-b):.2e}  |stored-resid| {abs(a-b):.2e}")
        ok &= abs(c - b) < 1e-9 * max(1.0, abs(b)) and abs(a - b) < 1e-9 * max(1.0, abs(b))

    print("(b) brute force: Monte Carlo over explicit noise, SHIFTED+RESCALED test split")
    # Every replicate sweeps ALL the test images, so the only randomness left is the noise
    # draw and the replicate spread is a genuine standard error -- the check is then
    # |closed - MC| < 3 SE rather than an invented tolerance.  Sampling images WITH
    # replacement instead would add an O(1/sqrt(n)) image-sampling term that the closed form
    # does not have, and would make the comparison noise-limited for no reason.
    for (dd, kk, NN, sg, nrep) in [(10, 6, 500, 0.7, 400), (12, 9, 400, 1.2, 400)]:
        g = torch.Generator(device='cpu').manual_seed(7 * dd + kk)
        Xtr = (torch.randn(NN, dd, generator=g) ** 3 * 0.4 + 0.6).to(DEV, DT)
        # genuinely different moments: shift, rescale and stretch the coordinates
        Xte = (torch.randn(NN, dd, generator=g) ** 3 * 0.9 - 0.35).to(DEV, DT)
        Xte = Xte * torch.linspace(0.5, 1.8, dd, device=DEV, dtype=DT)[None, :]
        Th = (torch.randn(kk, dd, generator=g) / np.sqrt(dd)).to(DEV, DT)
        _, _, L_test, W, mu, gmean = split_losses(Xtr, Xte, Th, sg, lam=1e-6,
                                                  want_model=True)
        beta = mu - W @ gmean
        gz = torch.Generator(device=DEV).manual_seed(5)
        reps = []
        for _ in range(nrep):
            y = Xte + sg * torch.randn(Xte.shape, device=DEV, dtype=DT, generator=gz)
            pred = torch.clamp(y @ Th.T, min=0.0) @ W.T + beta
            reps.append(float(((Xte - pred) ** 2).sum()) / NN)
        mc = float(np.mean(reps)); se = float(np.std(reps, ddof=1) / np.sqrt(nrep))
        print(f"    d={dd:3d} k={kk:3d} sigma={sg}   closed {L_test:.6f}   "
              f"MC({nrep} sweeps) {mc:.6f} +- {se:.6f}   "
              f"off by {abs(mc-L_test)/se:.2f} SE")
        ok &= abs(mc - L_test) < 3.0 * se
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
def main():
    Xtr = load(True, NIMG).reshape(NIMG, d)
    Xte = load(False, NTEST).reshape(NTEST, d)
    print(f"DENSE HELD-OUT  d={d}  train={NIMG}  test={NTEST}  lam={LAM}  seeds={NSEED}",
          flush=True)
    print(f"  Tr(Sigma_train) = {float(((Xtr-Xtr.mean(0))**2).sum()/NIMG):.4f}   "
          f"Tr_test about the TRAIN mean = "
          f"{float(((Xte-Xtr.mean(0))**2).sum()/NTEST):.4f}", flush=True)

    store = {}
    if os.path.exists(OUT):
        store = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}

    # the stored in-sample dense tables, for the reproduction assert
    prior = {}
    for f, tag in (('tables/rf_pixel_dense_sweep.npz', 'sweep'),
                   ('tables/rf_pixel_rowmult.npz', 'rowmult')):
        if os.path.exists(f):
            prior[tag] = {k: v for k, v in np.load(f, allow_pickle=True).items()}

    for sg in SIGS:
        ltr, lte = linear_split(Xtr, Xte, sg)
        store[f'linear|{sg}'] = np.array([ltr, lte])
        print(f"\n=== sigma={sg}   WIENER  train {ltr:.4f}   test {lte:.4f}   "
              f"gap {lte-ltr:+.4f} ===", flush=True)
        for j in sorted(JS):
            key = f'{sg}|{j}|dense'
            if key in store:
                v = store[key]
                print(f"  j={j}: already done  "
                      f"train {v[:,0].mean():.4f}  test {v[:,2].mean():.4f}", flush=True)
                continue
            k = int(round(j * d))
            t0 = time.time(); rows = []
            try:
                for s in range(NSEED):
                    # SAME seed scheme as scripts/rf_pixel_dense_sweep.py, so the train
                    # column is comparable cell for cell with the stored in-sample table.
                    Th = torch.as_tensor(
                        np.random.default_rng(3 + s).standard_normal((k, d)) / np.sqrt(d),
                        device=DEV, dtype=DT)
                    rows.append(split_losses(Xtr, Xte, Th, sg))
                    del Th
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  j={j} k={k}: OOM, stopping this sigma", flush=True)
                torch.cuda.empty_cache()
                break
            a = np.array(rows)
            store[key] = a
            np.savez(OUT, **store)
            tr_m, rs_m, te_m = a.mean(0)
            print(f"  j={j} k={k} ({k*d:,} params): train {tr_m:.4f}  resid {rs_m:.4f}  "
                  f"test {te_m:.4f}   own gap {te_m-rs_m:+.4f}   "
                  f"vs Wiener: train {tr_m-ltr:+.4f}  test {te_m-lte:+.4f}   "
                  f"[{time.time()-t0:.0f}s, peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB]", flush=True)
            torch.cuda.reset_peak_memory_stats()

            # (d) the train column must reproduce the stored in-sample dense tables
            for tag, tab in prior.items():
                ref = tab.get(f'{sg}|{j}|dense') if tag == 'sweep' else tab.get(f'{sg}|{j}|0')
                if ref is None:
                    continue
                n = min(len(ref), len(a))
                e = float(np.abs(np.asarray(ref, float)[:n] - a[:n, 0]).max())
                print(f"      reproduces {tag}: max |diff| = {e:.2e}", flush=True)
                assert e < 5e-6, f"{key} does not reproduce {tag}: {e}"
    print("\ndone", flush=True)


if __name__ == '__main__':
    if os.environ.get('SELFTEST'):
        sys.exit(0 if selftest() else 1)
    main()
