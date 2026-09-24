"""THE BAYES ORACLE FOR THE EMPIRICAL TRAINING PRIOR, HELD OUT (michimin, 2026-09-23 21:28
"add bayes oracle to the graph").

*** THE IN-SAMPLE VERSION OF THIS CURVE WAS RETRACTED ON 2026-08-10 AND MUST NOT GO ON THE
PLOT.  THE HELD-OUT VERSION IS A DIFFERENT AND WELL-POSED OBJECT.  READ THIS BEFORE USING
EITHER. ***

The `bayes_uncond` curve in figures/dnn_feature_mmse_*.png is the posterior mean under the
EMPIRICAL measure on N = 10^4 CIFAR atoms, evaluated on noised versions of THOSE SAME ATOMS.
Its reported MMSE falls to ~0 below sigma ~ 0.5, and that is not a property of CIFAR: the
posterior effective support N_eff = exp(H(w)) measured 1.00 at every sigma <= 1.61, i.e. the
posterior is a point mass on the very image that generated y.  It is a memoriser being
scored on its own training set, so it is a LOWER bound on nothing and an oracle for nothing.

Held out, the same estimator becomes meaningful and the pathology disappears by construction:

    x_hat(y) = sum_j w_j x_j,   w_j prop exp(-||y - x_j||^2 / 2 sigma^2),   x_j in TRAIN,
    L = E_test || x0 - x_hat(x0 + sigma z) ||^2,                            x0 in TEST.

The atoms are the 10,000 TRAIN images and the thing being denoised is a TEST image the atom
set does not contain, so no weight can ever collapse onto the answer.  What this measures is
the exact Bayes risk of the EMPIRICAL TRAINING PRIOR on held-out data -- i.e. the loss of a
model that has learned the training distribution PERFECTLY and learned nothing beyond it.
That is exactly the object a diffusion model is fitted to approximate, which makes it the
right reference to put next to EDM: EDM is an approximation to this, and anywhere EDM BEATS
it, the approximation is generalising rather than reproducing its prior.

⚠ SO IT IS NOT A LOWER BOUND ON THE PLOT AND MUST NOT BE DESCRIBED AS ONE.  The true Bayes
risk is under the POPULATION prior, which is unestimable at d = 3072 with 10^4 atoms (the
2026-08-10 N_eff finding: you would need N growing like exp(d)).  The empirical prior is a
mis-specified prior, and at small sigma it is a very bad one -- it puts all its mass on 10^4
points and the test image is not one of them.  Expect this curve to be WORSE than linear at
low sigma, which is the whole point of showing it.

Same contract as every other curve on figures/rf_heldout_band_vs_sigma.png: fitted (here:
its atoms) on CIFAR train[:10000], scored on the 10,000 CIFAR TEST images.

    NREP=4 python scripts/bayes_oracle_heldout.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scripts.rf_pixel_heldout import load, linear_split, DEV, DT, d

SIGS = [float(x) for x in
        os.environ.get('SIGS', '0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
NREP = int(os.environ.get('NREP', '4'))
CHUNK = int(os.environ.get('CHUNK', '500'))
OUT = os.environ.get('OUT', 'tables/bayes_oracle_heldout.npz')


def oracle(atoms, targets, sg, nrep, seed0=0, chunk=CHUNK, want_neff=True):
    """Posterior mean under the empirical measure on `atoms`, scored on `targets`.

    Returns (mean loss, standard error over the nrep sweeps, median N_eff).

    `atoms` and `targets` are the SAME tensor for the in-sample (memorising) variant and
    different splits for the held-out one -- that single substitution is the entire
    difference between the retracted curve and the usable one.
    """
    # dimension is inferred, NOT taken from the module-level d -- the validation below runs
    # 3x4x4 toys through this same function.
    A = atoms.reshape(atoms.shape[0], -1)
    T = targets.reshape(targets.shape[0], -1)
    a2 = (A * A).sum(1)                                   # ||x_j||^2, once
    per_rep, neffs = [], []
    for r in range(nrep):
        g = torch.Generator(device=DEV).manual_seed(4200 + 17 * r)
        tot, cnt = 0.0, 0
        for i in range(0, T.shape[0], chunk):
            x0 = T[i:i + chunk]
            y = x0 + sg * torch.randn(x0.shape, generator=g, device=DEV, dtype=DT)
            # ||y - x_j||^2 = ||y||^2 - 2 y.x_j + ||x_j||^2; the ||y||^2 term is constant
            # across j inside the softmax and is dropped.
            logw = -(a2.view(1, -1) - 2.0 * (y @ A.T)) / (2.0 * sg ** 2)
            w = torch.softmax(logw, dim=1)
            xh = w @ A
            tot += float(((x0 - xh) ** 2).sum())
            cnt += x0.shape[0]
            if want_neff and r == 0:
                lw = torch.log(w.clamp_min(1e-300))
                neffs.append(torch.exp(-(w * lw).sum(1)))
        per_rep.append(tot / cnt)
    v = np.array(per_rep)
    ne = float(torch.cat(neffs).median()) if neffs else float('nan')
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, ne


def _validate():
    """(a) brute force, (b) the sigma -> infinity limit, (c) the sigma -> 0 limits.

    (a) is the one that would catch an algebra error; (b) and (c) pin the two ends against
    quantities computed by completely different code (a mean, and a nearest-neighbour search).
    """
    g = torch.Generator(device=DEV).manual_seed(7)
    A = torch.randn(40, 3, 4, 4, generator=g, device=DEV, dtype=DT) * 1.3 + 0.2
    T = torch.randn(11, 3, 4, 4, generator=g, device=DEV, dtype=DT) * 0.9
    dd = 3 * 4 * 4
    for sg in (0.4, 1.3):
        got, _, _ = oracle(A, T, sg, 1, chunk=3)
        # explicit python loop over the same definition
        gen = torch.Generator(device=DEV).manual_seed(4200)
        Af, Tf = A.reshape(40, dd), T.reshape(11, dd)
        tot = 0.0
        for i in range(0, 11, 3):
            x0 = Tf[i:i + 3]
            y = x0 + sg * torch.randn(x0.shape, generator=gen, device=DEV, dtype=DT)
            for k in range(x0.shape[0]):
                lg = torch.tensor([-float(((y[k] - Af[j]) ** 2).sum()) / (2 * sg ** 2)
                                   for j in range(40)], device=DEV, dtype=DT)
                ww = torch.exp(lg - lg.max()); ww = ww / ww.sum()
                tot += float(((x0[k] - ww @ Af) ** 2).sum())
        ref = tot / 11
        rel = abs(got - ref) / abs(ref)
        print(f"  brute force sigma={sg}: {got:.10f} vs {ref:.10f}  rel={rel:.1e} "
              f"{'OK' if rel < 1e-12 else 'FAIL'}")
        assert rel < 1e-12
    # (b) sigma -> infinity: weights flatten, x_hat -> mean(atoms), so the loss must approach
    # E||x0_test - mu_train||^2 -- computed here without any softmax at all.
    big = 4000.0
    got, _, ne = oracle(A, T, big, 1, chunk=5)
    mu = A.reshape(40, dd).mean(0)
    ref = float(((T.reshape(11, dd) - mu) ** 2).sum(1).mean())
    print(f"  sigma->inf: {got:.6f} vs E||x0-mu_train||^2 = {ref:.6f}  "
          f"rel={abs(got-ref)/ref:.1e}  N_eff={ne:.2f}/40 {'OK' if abs(got-ref)/ref < 2e-3 else 'FAIL'}")
    assert abs(got - ref) / ref < 2e-3
    # (c) sigma -> 0, held out: the posterior collapses onto the NEAREST atom, so the loss
    # must approach the mean squared nearest-neighbour distance from test to train.
    small = 1e-3
    got, _, ne = oracle(A, T, small, 1, chunk=5)
    D = torch.cdist(T.reshape(11, dd), A.reshape(40, dd)) ** 2
    ref = float(D.min(1).values.mean())
    print(f"  sigma->0 held out: {got:.6f} vs mean sq NN distance = {ref:.6f}  "
          f"rel={abs(got-ref)/ref:.1e}  N_eff={ne:.3f} "
          f"{'OK' if abs(got-ref)/ref < 1e-6 else 'FAIL'}")
    assert abs(got - ref) / ref < 1e-6
    # (c') sigma -> 0, IN SAMPLE: the atom set contains the answer, so the loss collapses to
    # ZERO.  This is the memorisation the 2026-08-10 retraction is about, reproduced on demand.
    got, _, ne = oracle(A, A, small, 1, chunk=5)
    print(f"  sigma->0 IN SAMPLE: {got:.3e} (must be ~0 = the memorisation artefact)  "
          f"N_eff={ne:.3f} {'OK' if got < 1e-9 else 'FAIL'}")
    assert got < 1e-9
    return True


def main():
    print("validating:", flush=True)
    _validate()
    Xtr, Xte = load(True, NIMG), load(False, NTEST)
    Xtr1, Xte1 = Xtr.reshape(NIMG, d), Xte.reshape(NTEST, d)
    print(f"\nEMPIRICAL-PRIOR BAYES ORACLE.  atoms = {NIMG} train, targets = {NTEST} test, "
          f"{NREP} noise sweeps each\n", flush=True)
    # THE sigma -> 0 ASYMPTOTE OF THE HELD-OUT CURVE, computed WITHOUT any softmax: as sigma
    # falls the posterior collapses onto the nearest train atom, so the loss must flatten onto
    # the mean squared nearest-neighbour distance from test to train.  Storing it turns the
    # low-sigma plateau into a checkable prediction rather than a shape on a plot.
    nn = 0.0
    for i in range(0, NTEST, 500):
        nn += float((torch.cdist(Xte1[i:i + 500], Xtr1) ** 2).min(1).values.sum())
    nn /= NTEST
    print(f' mean squared NN distance, test -> train = {nn:.4f}   '
          f'(= the sigma->0 limit of the held-out column below)\n', flush=True)
    # *** THE 50,000-ATOM ORACLE IS THE MATCHED-PRIOR REFERENCE FOR EDM, AND WITHOUT IT THE
    # EDM COMPARISON IS RIGGED. ***  EDM saw all 50k CIFAR train images, so scoring it against
    # a 10k-atom empirical prior charges it for a handicap it does not have: a denser atom set
    # has a smaller nearest-neighbour distance and therefore a genuinely lower Bayes risk under
    # its own prior.  Both oracles are scored on the SAME 10,000 test images as every other
    # curve, so the 10k one stays matched to the RF arms and the 50k one is matched to EDM.
    X50 = load(True, 50000)
    X501 = X50.reshape(50000, d)
    nn50 = 0.0
    for i in range(0, NTEST, 500):
        nn50 += float((torch.cdist(Xte1[i:i + 500], X501) ** 2).min(1).values.sum())
    nn50 /= NTEST
    print(f' mean squared NN distance, test -> train[:50000] = {nn50:.4f}   '
          f'(only {100*(nn-nn50)/nn:.1f}% below the 10k figure -- 5x the atoms barely moves it '
          f'at d=3072)\n', flush=True)
    print(' sigma   Wiener_te  ORACLE_te     SE   vs Wiener |  in-sample   N_eff_te  N_eff_tr'
          ' | 50k-atom  vs EDM')
    store = {'nn_sq_test_to_train': np.array([nn]),
             'nn_sq_test_to_train50': np.array([nn50]),
             'trace_test': np.array([
                 float((Xte1 ** 2).sum(1).mean()) - float((Xte1.mean(0) ** 2).sum())])}
    for sg in SIGS:
        t0 = time.time()
        wtr, wte = linear_split(Xtr1, Xte1, sg)
        te, se, ne_te = oracle(Xtr, Xte, sg, NREP)
        ins, _, ne_tr = oracle(Xtr, Xtr, sg, 1)
        te50, se50, ne50 = oracle(X50, Xte, sg, NREP)
        store[f'oracle|{sg}'] = np.array([te, se, ins, ne_te, ne_tr])
        store[f'oracle50|{sg}'] = np.array([te50, se50, ne50])
        # [train, test], the SAME convention as every other held-out table in the project --
        # rf_heldout_band_plot.py cross-checks this row against all of them and indexes [1].
        store[f'linear|{sg}'] = np.array([wtr, wte])
        print(f' {sg:5.3f} {wte:10.4f} {te:10.4f} {se:6.4f} {te-wte:+10.4f} | '
              f'{ins:10.4f} {ne_te:10.2f} {ne_tr:9.2f} | {te50:9.4f}'
              f'   [{time.time()-t0:.0f}s]', flush=True)
        np.savez(OUT, **store)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == '__main__':
    main()
