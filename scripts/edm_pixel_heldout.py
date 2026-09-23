"""EDM U-NET ON THE SAME 10,000 CIFAR TEST IMAGES (michimin, 2026-09-23 05:44).

WHAT THIS ADDS THAT NOTHING ELSE IN THE PROJECT HAS
---------------------------------------------------
Every denoiser in tables/rf_pixel_heldout.npz and tables/rf_pixel_dense_heldout.npz is a
random-feature model with a linear readout, and the only nonlinear reference we have ever
had is the "oracle Bayes" curve in figures/dnn_feature_mmse_*.png -- which is Bayes for the
ATOMIC empirical measure on 10^4 images and was retracted on 2026-08-10 as a memoriser
(posterior effective support N_eff = 1.00 for every sigma <= 1.61).  A pretrained EDM
denoiser is the one thing available that is genuinely nonlinear, genuinely trained for this
exact task, and NOT a nearest-neighbour lookup.  On the held-out split it is the closest
thing to an achievable upper end of the scale our RF numbers live on.

⚠⚠ EDM IS NOT A MATCHED-10k COMPARISON, AND MUST NOT BE TABULATED AS ONE.
EDM was trained on the CIFAR-10 TRAINING SET -- all 50,000 images -- while every RF arm here
was fitted on 10,000.  So the EDM row is the analogue of the 50k-Wiener row in
docs/rf_heldout_and_nsweep.md section 4.3: same evaluation set, five times the training
data, and therefore a HANDICAP IN EDM'S FAVOUR.  It bounds what is achievable on these test
images; it does not say what a model trained on 10k could do.

WHAT IS AND IS NOT HELD OUT HERE.  There is nothing to solve -- the network is frozen -- so
"held out" means only that the 10,000 CIFAR TEST images were never seen in training.  The
TRAIN column evaluates the same network on the first 10,000 CIFAR TRAIN images, which ARE in
its training set.  The difference between the two columns is therefore a direct measurement
of how much a real diffusion model memorises its training set at each noise level, on
exactly the same footing as the +0.459 we measured for Wiener and the +0.02..+0.10 for the
circulant arms.  That number is worth having on its own.

UNITS AND SCALE.  EDM works in [-1,1]:  x_edm = 2 x_pixel - 1  =>  sigma_edm = 2 sigma_pixel,
and the returned denoised image is mapped back before the error is taken, so the loss below
is the per-image SUM over all 3072 coordinates of the squared error in [0,1] pixel units --
identical units to every L in rf_pixel_heldout.npz.  (The 2026-08-17 warning about not
overlaying the EDM curve applies to the d=512 avgpool representation, NOT here: both sides
of this comparison are raw pixels.)

ERROR BARS.  The noise z is sampled, not integrated analytically, so each number carries MC
error.  Each replicate is a full sweep over all 10,000 images with a fresh z, so the spread
across replicates is a genuine standard error on the mean and the train-vs-test gap can be
read against it instead of being taken on faith.

    python scripts/edm_pixel_heldout.py
    SIGS=0.127,0.452 NETS=uncond-ve NREP=2 python scripts/edm_pixel_heldout.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/n/home12/binxuwang/Github/edm')
import pickle
import numpy as np, torch
from scripts.rf_pixel_heldout import load, linear_split, DEV, DT, d

STORE = os.environ.get('STORE_DIR',
                       '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
CKPT_DIR = os.path.join(STORE, 'Datasets/EDM_datasets/edm_ckpts')
SIGS = [float(x) for x in
        os.environ.get('SIGS', '0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0').split(',')]
NETS = os.environ.get('NETS', 'uncond-ve,uncond-vp').split(',')
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
NREP = int(os.environ.get('NREP', '6'))
BATCH = int(os.environ.get('BATCH', '500'))
OUT = os.environ.get('OUT', 'tables/edm_pixel_heldout.npz')


@torch.no_grad()
def edm_loss(net, X, sigma_pixel, nrep, seed, labels=None):
    """Per-image summed squared error in [0,1] pixel units, and its MC standard error.

    X is (N, 3072) in [0,1].  Each of the `nrep` replicates sweeps ALL N images with a fresh
    noise draw, so np.std(reps)/sqrt(nrep) is an honest SE on the mean.
    """
    N = X.shape[0]
    sigma_edm = 2.0 * sigma_pixel
    Xe = (X.reshape(N, 3, 32, 32).to(torch.float32) * 2.0 - 1.0)
    g = torch.Generator(device=DEV).manual_seed(seed)
    reps = []
    for _ in range(nrep):
        tot = 0.0
        for a in range(0, N, BATCH):
            xb = Xe[a:a + BATCH]
            y = xb + sigma_edm * torch.randn(xb.shape, device=DEV, dtype=torch.float32,
                                             generator=g)
            sv = torch.full([len(xb)], sigma_edm, device=DEV, dtype=torch.float32)
            lb = labels[a:a + BATCH] if labels is not None else None
            D = net(y, sv, class_labels=lb)
            pred = (D + 1.0) / 2.0                                  # back to [0,1]
            tot += float(((pred.reshape(len(xb), -1).to(torch.float64)
                           - X[a:a + BATCH]) ** 2).sum())
        reps.append(tot / N)
    return float(np.mean(reps)), float(np.std(reps, ddof=1) / np.sqrt(nrep)) if nrep > 1 \
        else 0.0


def main():
    Xtr = load(True, NIMG).reshape(NIMG, d)
    Xte = load(False, NTEST).reshape(NTEST, d)
    print(f"EDM HELD-OUT  train={NIMG} (IN EDM'S TRAINING SET)  test={NTEST} (unseen)  "
          f"reps={NREP}", flush=True)

    store = {}
    if os.path.exists(OUT):
        store = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}

    for sg in SIGS:                                    # the linear reference, same splits
        ltr, lte = linear_split(Xtr, Xte, sg)
        store[f'linear|{sg}'] = np.array([ltr, lte])

    for tag in NETS:
        path = os.path.join(CKPT_DIR, f'edm-cifar10-32x32-{tag}.pkl')
        with open(path, 'rb') as f:
            net = pickle.load(f)['ema'].to(DEV).eval()
        smin, smax = float(net.sigma_min), float(net.sigma_max)
        print(f"\n=== {tag}   label_dim={net.label_dim}   "
              f"sigma_edm valid in [{smin:g}, {smax:g}] ===", flush=True)
        print(f"{'sigma':>7} {'sig_edm':>8} | {'train(seen)':>12} {'test(unseen)':>13} "
              f"{'gap':>9} | {'Wiener_test':>11} {'vs Wiener':>10}", flush=True)
        for sg in SIGS:
            se_edm = 2.0 * sg
            if not (smin <= se_edm <= smax):
                print(f"  sigma={sg}: sigma_edm={se_edm:g} OUTSIDE the model's range, "
                      f"skipping", flush=True)
                continue
            t0 = time.time()
            a, a_se = edm_loss(net, Xtr, sg, NREP, seed=1000 + int(sg * 1000))
            b, b_se = edm_loss(net, Xte, sg, NREP, seed=1000 + int(sg * 1000))
            store[f'{tag}|{sg}'] = np.array([a, a_se, b, b_se])
            np.savez(OUT, **store)
            lte = float(store[f'linear|{sg}'][1])
            print(f"{sg:7.3f} {se_edm:8.3f} | {a:8.4f}+-{a_se:.4f} "
                  f"{b:9.4f}+-{b_se:.4f} {b-a:+9.4f} | {lte:11.4f} {b-lte:+10.4f}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
        del net
        torch.cuda.empty_cache()
    print("\ndone", flush=True)


if __name__ == '__main__':
    main()
