"""IS THE Z_32 x Z_32 WIN OVER WIENER IN-SAMPLE ONLY?  Train/test split on raw CIFAR pixels.

Every L^circ this project has reported is an IN-SAMPLE number: the per-frequency readout
w_f = (P_f + lam I)^{-1} q_f is solved on the same moments it is scored against.  At c = 1536
the 2-D model carries 4.7M trained parameters and 1.57M features against N = 10^4 images, and
on 2026-09-21 a model in that regime (the dense RF at k/d = 8) turned out to be scoring well
by memorising -- its advantage over linear collapsed 0.825 -> 0.336 -> 0.039 as N went
10k -> 20k -> 40k, with lam = 1e-6 doing nothing to stop it.

The N-sweep (job 47894575) attacks that indirectly.  This attacks it directly: keep w_f, build
P_f^test and q_f^test from the CIFAR-10 TEST split (10,000 images the solve never saw), and
evaluate

    L_test = Tr(Sigma_test) + sum_f wgt_f [ -2 Re <q_f^test, w_f> + w_f^H P_f^test w_f ].

TRAIN = the first 10,000 CIFAR TRAIN images -- the exact set every existing table was built
on -- so the 'train' column here must reproduce tables/rf_pixel_circ2d.npz and
tables/rf_pixel_circ_csweep.npz cell for cell.  That reproduction is a free correctness check
on the refactor and is asserted below, not merely eyeballed.

THE BASELINE HAS TO MOVE TOO.  The Wiener optimum is itself fitted: Sigma is estimated on the
train split.  Comparing a held-out RF loss against an in-sample Wiener loss would manufacture
a fake generalisation gap.  So the linear baseline is also solved on train and scored on test:
    A = Sigma_tr (Sigma_tr + sigma^2 I)^{-1},  b = mu_tr - A mu_tr,
    L^lin_test = E_test || x0 - A y - b ||^2 = Tr((I-A) Sigma_te (I-A)^T) + sigma^2 Tr(A A^T)
                                              + || (I-A)(mu_te - mu_tr) ||^2,
which is the closed form of the same held-out criterion applied to the linear denoiser (the
noise is independent of the split, hence the bare sigma^2 Tr(A A^T)).  Every "excess over
Wiener" below is train-vs-train or test-vs-test, never mixed.

BOTH ARMS.  The 1-D Z_3072 control is run at the same c: the 1-D circulant's flatness in c has
been quoted as evidence that its function class is too small to overfit, but that has never
been checked against a held-out split either.  If the 2-D arm generalises and the 1-D arm does
too, the re-index win is real; if only the 2-D arm degrades, the extra freedom (3-channel
mixing, 27 taps) is being spent on memorisation.

    CS=512,1536 SIGS=0.127,0.452 python scripts/rf_pixel_heldout.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torchvision, torchvision.transforms as T
from core.rf_circulant2d import circulant2d_rf_mmse
from core.rf_circulant_struct import circulant_rf_mmse_lag2

DEV = 'cuda'; DT = torch.float64; LAM = 1e-6
H = W = 32; CIN = 3; d = CIN * H * W
T2 = int(os.environ.get('T2', '3'))                 # 2-D taps per side
TB = int(os.environ.get('T_BAND', '8'))             # 1-D taps
CS = [int(x) for x in os.environ.get('CS', '32,96,256,512,1536').split(',')]
SIGS = [float(x) for x in os.environ.get('SIGS', '0.127,0.452,1.610,5.0').split(',')]
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '10000'))
NSEED = int(os.environ.get('NSEED', '2'))
ARMS = os.environ.get('ARMS', '2d,1d').split(',')
OUT = os.environ.get('OUT', 'tables/rf_pixel_heldout.npz')


def load(train, n):
    ds = torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets', train=train,
                                      download=False, transform=T.ToTensor())
    dl = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=4)
    im = []
    for xb, _ in dl:
        im.append(xb)
        if sum(o.shape[0] for o in im) >= n:
            break
    return torch.cat(im)[:n].to(DEV, DT)


def linear_split(Xtr, Xte, sg):
    """Wiener solved on train, scored in-sample and on the held-out split."""
    n_tr, n_te = Xtr.shape[0], Xte.shape[0]
    mu_tr = Xtr.mean(0)
    Ctr = Xtr - mu_tr
    S_tr = (Ctr.T @ Ctr) / n_tr
    ev, U = torch.linalg.eigh(S_tr)
    shr = ev / (ev + sg ** 2)                          # A = U diag(shr) U^T
    train = float((sg ** 2 * shr).sum())               # = sum lam_i sigma^2/(lam_i+sigma^2)
    Cte = Xte - mu_tr                                  # TRAIN-centred: b is part of the model
    Z = Cte @ U                                        # (n_te, d) in the train eigenbasis
    resid = float(((Z * (1.0 - shr)) ** 2).sum() / n_te)   # Tr((I-A) Sigma_te^raw (I-A)^T)
    noise = float((sg ** 2) * (shr ** 2).sum())            # sigma^2 Tr(A A^T)
    return train, resid + noise


def sizing2d(c):
    """Held-out holds TWO lag tensors (train and test) live at once, and pass 2 holds two
    (nf, c, c) moment blocks, so both budgets double against scripts/rf_pixel_circ2d.py."""
    nlag = 3 * (2 * T2 - 1) ** 2
    nf = int(min(64, max(4, (30e9 - 2 * c * c * nlag * 8) / (2 * c * c * 24))))
    ns = int(min(max(NIMG, NTEST), max(256, 4e9 / (nf * c * 16))))
    return nf, ns


def sizing1d(c):
    nlag = 3 * (2 * TB - 1)
    nf = int(min(48, max(4, (30e9 - 2 * c * c * nlag * 8) / (2 * c * c * 24))))
    ns = int(min(max(NIMG, NTEST), max(256, 4e9 / (nf * c * 16))))
    return nf, ns


# The two in-sample drivers use DIFFERENT seed bases (800 in rf_pixel_featmatch2.py, 900 in
# rf_pixel_circ2d.py).  Both are mirrored exactly so that the train column reproduces the
# stored tables bit for bit -- that reproduction is the correctness check asserted in main().
def filt2d(c, s):
    g = torch.Generator(device=DEV); g.manual_seed(900 + 11 * s + c)
    h = torch.zeros(c, CIN, H, W, device=DEV, dtype=DT)
    h[:, :, :T2, :T2] = torch.randn(c, CIN, T2, T2, generator=g, device=DEV,
                                    dtype=DT) / np.sqrt(CIN * T2 * T2)
    return h


def filt1d(c, s):
    g = torch.Generator(device=DEV); g.manual_seed(800 + 11 * s + c)
    h = torch.zeros(c, d, device=DEV, dtype=DT)
    h[:, :TB] = torch.randn(c, TB, generator=g, device=DEV, dtype=DT) / np.sqrt(TB)
    return h


def main():
    Xtr = load(True, NIMG)
    Xte = load(False, NTEST)
    print(f"held-out L^circ, raw CIFAR pixels   train={Xtr.shape[0]} (CIFAR train[:N])  "
          f"test={Xte.shape[0]} (CIFAR test split)  arms={ARMS}  seeds={NSEED}", flush=True)
    Xtr1 = Xtr.reshape(Xtr.shape[0], d)
    Xte1 = Xte.reshape(Xte.shape[0], d)

    store = {}
    if os.path.exists(OUT):
        store = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}
        print(f"resuming from {OUT}: {len(store)} cells already present", flush=True)

    # the in-sample cells every existing table was built on, for the reproduction assert
    prior = {}
    for f, tag in (('tables/rf_pixel_circ2d.npz', '2d'),
                   ('tables/rf_pixel_circ_csweep.npz', '1d')):
        if os.path.exists(f):
            prior[tag] = dict(np.load(f, allow_pickle=True))

    for sg in SIGS:
        lin_tr, lin_te = linear_split(Xtr1, Xte1, sg)
        store[f'linear|{sg}'] = np.array([lin_tr, lin_te])
        print(f"\n=== sigma={sg}   Wiener train={lin_tr:.4f}  test={lin_te:.4f} ===",
              flush=True)
        for c in CS:
            for arm in ARMS:
                key = f'{sg}|{c}|{arm}'
                if key in store:
                    v = store[key]
                    print(f"  {arm} c={c}: already done "
                          f"(train {np.mean(v[:, 0]):.4f} test {np.mean(v[:, 2]):.4f})",
                          flush=True)
                    continue
                nf, ns = (sizing2d(c) if arm == '2d' else sizing1d(c))
                rows = []
                for s in range(NSEED):
                    t1 = time.time()
                    if arm == '2d':
                        r = circulant2d_rf_mmse(Xtr, filt2d(c, s), sg, T2, lam=LAM,
                                                device=DEV, freq_chunk=nf, super_chunk=ns,
                                                x0_test=Xte)
                    else:
                        r = circulant_rf_mmse_lag2(Xtr1, filt1d(c, s), sg, TB, lam=LAM,
                                                   device=DEV, freq_chunk=nf, super_chunk=ns,
                                                   x0_test=Xte1)
                    rows.append([r['train'], r['train_resid'], r['test']])
                    print(f"      [{arm} seed {s}: train {r['train']:.4f}  "
                          f"resid {r['train_resid']:.4f}  test {r['test']:.4f}  "
                          f"{time.time()-t1:.0f}s  nf={nf} NS={ns}  "
                          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB]",
                          flush=True)
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                a = np.array(rows)
                store[key] = a
                np.savez(OUT, **store)
                tr_m, te_m = float(a[:, 0].mean()), float(a[:, 2].mean())
                print(f"  >> {arm} c={c}: train {tr_m:.4f} ({tr_m-lin_tr:+.4f} vs Wiener)   "
                      f"test {te_m:.4f} ({te_m-lin_te:+.4f} vs Wiener)   "
                      f"gap {te_m-float(a[:, 1].mean()):+.4f}", flush=True)

                # REPRODUCTION CHECK: same data, same seeds, same lam -- the train column
                # must land on the in-sample table.  A mismatch means the held-out refactor
                # moved the estimator, which would invalidate every comparison above.
                # the 1-D table is keyed by j = c/d, the 2-D one by c itself
                pk = (f'{sg}|{c}|circ2d' if arm == '2d' else f'{sg}|{c / d}|circ')
                old = prior.get(arm, {}).get(pk)
                if old is not None and len(old) >= len(a):
                    dv = float(np.abs(np.asarray(old)[:len(a)] - a[:, 0]).max())
                    print(f"     [reproduces {'circ2d' if arm=='2d' else 'csweep'} table: "
                          f"max|delta| = {dv:.2e}]", flush=True)
                    assert dv < 1e-8, f"{key} does not reproduce the in-sample table: {dv}"
    print("\ndone", flush=True)


if __name__ == '__main__':
    main()
