"""LOCAL SOFTMAX DENOISERS HELD OUT: Kamb & Ganguli LS / ELS and Lukoianov et al., on the
footing of figures/rf_heldout_band_vs_sigma.png  (2026-09-28).

WHERE THESE SIT.  scripts/bayes_oracle_heldout.py is the empirical-Bayes posterior mean over
the whole image: held out it is ~100 per image for every sigma <= 1.6 (worse than the
test-to-train nearest-neighbour distance would suggest is useful, and 12x the Wiener at
sigma=0.127), because a test image is never one of the 10^4 atoms.  LS / ELS / LUK are the
same estimator with the distance restricted to a LOCAL window (core/local_softmax.py), which
is the mechanism Kamb & Ganguli and Lukoianov et al. propose for how a CNN denoiser
generalises past its training set.  Held out they are the natural bridge between that
failing global oracle and EDM, and the non-parametric counterparts of the circulant RFs.

THE CONTRACT -- identical to every curve on that figure:
  pool    = CIFAR train[:NIMG] (NIMG=10000): the softmax atoms, and the covariance behind
            both the Wiener arm and the Lukoianov masks;
  scored  = the first NTEST images of the CIFAR TEST split (never in the pool);
  loss    = per-image SUM over 3072 coordinates of squared error in [0,1] PIXEL units;
  sigma   = PIXEL units on the project grid.  Internally everything runs in the EDM
            convention x_edm = 2x - 1, sigma_edm = 2 sigma (the softmax and the Wiener filter
            are invariant to that affine change; the reference code's VP form maps to it
            exactly, see core/local_softmax.py), and errors are divided by 4 on the way out.

PAIRED.  At each sigma ONE noise draw z is made for the test (and val) images and EVERY arm --
wiener, ls, els, luk, edm -- denoises the SAME y.  Per-image losses are stored, so any
difference between two arms has a paired standard error.  NTEST < 10000 is a subset: compare
to the full-test EDM / RF tables through the `wiener` and `edm-*` arms run here on the same
subset, not by mixing subsets.  (The subset Wiener mean is also printed next to the full-test
closed form from linear_split as a check on how representative the subset is.)

HYPERPARAMETERS ARE SELECTED ON VALIDATION, NEVER ON TEST.  Kamb & Ganguli calibrate the patch
size per noise level (greedy, against a target network); Lukoianov et al. fix tau.  Here every
k in KS_* and tau in TAUS is run, and the report picks, per sigma, the value with the lowest
loss on NVAL images from CIFAR train[NIMG:NIMG+NVAL] -- disjoint from both the pool and the
test set.  The full k / tau curves are stored, so an oracle-on-test envelope or an
EDM-matched calibration can be read off later without re-running.

CONDITIONAL (COND=1).  The softmax pool of every test/val image is restricted to train images
of ITS OWN class (Kamb & Ganguli's `label=`; the class-conditional oracle of
scripts/dnn_feature_mmse.py).  The Lukoianov mask stays the unconditional one (it is a
locality prior, not a class statistic); the linear references are W_U and the per-class W_C;
the EDM arm is the class-conditional network.

KEYS in OUT (per-image loss vectors, float64):
    test:{arm}|{sigma}[|{param}]    (NTEST,)      val:...   (NVAL,)
    arm in wiener, wienerC, ls|k, els|k, luk|tau|mode, edm-<net>
    neff:ls|{sigma}|{k}, neff:luk|...   median posterior N_eff per image (test)
    meta_* : NIMG, NTEST, NVAL, COND, seeds

    VALIDATE=1 python scripts/local_softmax_heldout.py          # asserts only, tiny toys
    ARMS=wiener,ls,luk SIGS=0.452 NTEST=200 python scripts/local_softmax_heldout.py
    ARMS=els KS_ELS=3,5,7,9 NTEST=100 NVAL=50 python scripts/local_softmax_heldout.py
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torchvision, torchvision.transforms as T
from core.local_softmax import (ls_window_mask, wiener_locality_mask, masked_softmax_denoise,
                                els_denoise, wiener_denoise)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
torch.backends.cuda.matmul.allow_tf32 = False
C, H, W = 3, 32, 32
d = C * H * W
ROOT = '/n/home12/binxuwang/.keras/datasets'
STORE = os.environ.get('STORE_DIR',
                       '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
CKPT_DIR = os.path.join(STORE, 'Datasets/EDM_datasets/edm_ckpts')

SIGS = [float(x) for x in
        os.environ.get('SIGS', '0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0').split(',')]
ARMS = os.environ.get('ARMS', 'wiener,ls,luk,els,edm').split(',')
KS_LS = [int(x) for x in os.environ.get('KS_LS', '3,5,7,9,11,13,15,19,23,31,63').split(',')]
KS_ELS = [int(x) for x in os.environ.get('KS_ELS', '3,5,7,9,11,13,15,19,23,27,31').split(',')]
TAUS = [float(x) for x in os.environ.get('TAUS', '0.005,0.01,0.02,0.05,0.1,0.2').split(',')]
TAU_MODE = os.environ.get('TAU_MODE', 'global')
NIMG = int(os.environ.get('NIMG', '10000'))
NTEST = int(os.environ.get('NTEST', '1000'))
NVAL = int(os.environ.get('NVAL', '200'))
assert NTEST <= 10000 and NVAL <= 1000, 'the canonical noise tensors are 10000 / 1000 rows'
COND = int(os.environ.get('COND', '0'))
ELS_CHUNK = int(os.environ.get('ELS_CHUNK', '32'))
# float32 reproduces float64 to 3 decimals of the per-image loss on the pilot (sigma=0.853,
# k=3..31) at 2/3 of the time; the toy brute force passes at 1e-4 in float32.
ELS_DT = {'64': torch.float64, '32': torch.float32}[os.environ.get('ELS_DT', '32')]
EDM_NET = os.environ.get('EDM_NET', 'cond-vp' if COND else 'uncond-vp')
# DATASET: cifar10 (default) | ffhq32 | afhq32.  For the two face/animal sets the pool is
# images [0:NIMG] of the 32x32 tensors used by ~/Github/DiffusionLearningCurve, test is
# [TEST_START:TEST_START+NTEST], val is the LAST NVAL images.  The network arms are that
# project's SongUNets (1 block/level, 128 ch, DSM, 50k steps x 256), one per UNETS entry
# '{ntrain}_{split}':  10000_split1 trained on exactly [0:10000] = the pool (matched);
# 30000_split1 on [0:30000], 30000_split2 on [30000:60000].  FFHQ (70k) therefore uses
# TEST_START=60000 so the test images are unseen by all three; AFHQ (15803) only has room
# for 10000_split1 with TEST_START=10000.
# UNETS may also name 'full' (whole dataset, 50k steps) or 'full_longtrain' (FFHQ32 only:
# whole dataset, 250k steps) -- trained ON the test images, so not held out.
#   ⚠ 2026-09-28 diagnostic: the 10k U-nets MEMORISE -- FFHQ32 at sigma=0.127 each split
#   scores 1.30 on its own training images and 21.9 on the other split's (Wiener 10.2).
DATASET = os.environ.get('DATASET', 'cifar10')
DLC = os.path.join(STORE, 'DL_Projects/DiffusionSpectralLearningCurve')
UNETS = os.environ.get('UNETS', '10000_split1').split(',')
TEST_START = int(os.environ.get('TEST_START', str(NIMG)))
assert DATASET in ('cifar10', 'ffhq32', 'afhq32'), DATASET
assert not (COND and DATASET != 'cifar10'), 'no labels for ffhq32 / afhq32'
OUT = os.environ.get('OUT', f"tables/local_softmax_heldout{'_cond' if COND else ''}.npz"
                     if DATASET == 'cifar10' else f"tables/local_softmax_heldout_{DATASET}.npz")


def load_xy(train, n):
    ds = torchvision.datasets.CIFAR10(ROOT, train=train, download=False, transform=T.ToTensor())
    dl = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=4)
    im, lb = [], []
    for xb, yb in dl:
        im.append(xb); lb.append(yb)
        if sum(o.shape[0] for o in im) >= n:
            break
    return torch.cat(im)[:n].to(DEV, DT), torch.cat(lb)[:n].to(DEV)


def load_data():
    """(pool, pool labels, val, val labels, test, test labels, description), [0,1] pixels."""
    if DATASET == 'cifar10':
        Xp, lp = load_xy(True, NIMG + NVAL)
        Xt, lt = load_xy(False, NTEST)
        return (Xp[:NIMG], lp[:NIMG], Xp[NIMG:], lp[NIMG:], Xt, lt,
                f"pool=train[:{NIMG}]  test=test[:{NTEST}]  val=train[{NIMG}:{NIMG+NVAL}]")
    X = torch.load(os.path.join(DLC, 'wordnet_render_dataset',
                                f"{DATASET[:4]}-32x32.pt"), mmap=True)
    n = X.shape[0]
    assert NIMG <= TEST_START and TEST_START + NTEST <= n - NVAL, (n, TEST_START, NTEST, NVAL)
    f = lambda a, b: X[a:b].to(DEV, DT)
    z = lambda m: torch.zeros(m, dtype=torch.long, device=DEV)
    return (f(0, NIMG), z(NIMG), f(n - NVAL, n), z(NVAL),
            f(TEST_START, TEST_START + NTEST), z(NTEST),
            f"pool=[0:{NIMG}]  test=[{TEST_START}:{TEST_START+NTEST}]  val=[{n-NVAL}:{n}] of {n}")


def load_nets():
    """{tag: (D, uses_labels)}, D(y_edm (B,C,H,W) fp32, sigma_edm (B,), labels) -> x0-hat."""
    if DATASET == 'cifar10':
        return {f'edm-{EDM_NET}': load_net()}
    return {f'unet-{u}': load_net(u) for u in UNETS}


def load_net(unet_id=None):
    """One network arm.  unet_id None (cifar10 only) = the official EDM pickle; otherwise a
    DiffusionLearningCurve SongUNet '{ntrain}_{split}' | 'full' | 'full_longtrain'.  (CIFAR's
    learning-curve runs index EDM's cifar10-32x32.zip, whose order was checked identical to
    torchvision's train split, so CIFAR 10000_split1 is trained on exactly this pool.)"""
    if unet_id is None:
        sys.path.insert(0, '/n/home12/binxuwang/Github/edm')
        with open(os.path.join(CKPT_DIR, f'edm-cifar10-32x32-{EDM_NET}.pkl'), 'rb') as f:
            net = pickle.load(f)['ema'].to(DEV).eval()
        return lambda y, s, lab: net(y, s, class_labels=lab), bool(net.label_dim)
    # The learning-curve repo also has a top-level package named `core`, so its network file
    # is loaded by path.  Construction and preconditioning copy
    # experiment/CNN_unet_learn_curve_CLI.py (create_unet_model, EDMCNNPrecondWrapper).
    import json, importlib.util
    spec = importlib.util.spec_from_file_location(
        'dlc_network_edm_lib', '/n/home12/binxuwang/Github/DiffusionLearningCurve/core/'
        'network_edm_lib.py')
    lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
    # 'full' / 'full_longtrain' = the same architecture trained on the WHOLE dataset (50k /
    # 250k steps) -- the test images are then IN its training set (no held-out images exist).
    pre = {'cifar10': 'CIFAR', 'ffhq32': 'FFHQ32', 'afhq32': 'AFHQ32'}[DATASET]
    alias = {'full': f"{pre}_UNet_CNN_EDM_4blocks_wide128_attn_saveckpt_fewsample",
             'full_longtrain': f"{pre}_UNet_CNN_EDM_4blocks_wide128_attn_"
                               f"saveckpt_fewsample_longtrain"}
    if unet_id in alias:
        run = os.path.join(DLC, alias[unet_id])
    else:
        ntr, split = unet_id.split('_')
        run = os.path.join(DLC, f"{pre}_{ntr}_UNet_CNN_EDM_DSM_{split}")
    cfg = json.load(open(os.path.join(run, 'config.json')))
    unet = lib.SongUNet(in_channels=cfg['channels'], out_channels=cfg['channels'],
                        num_blocks=cfg['layers_per_block'],
                        attn_resolutions=cfg['attn_resolutions'],
                        decoder_init_attn=cfg.get('decoder_init_attn', True),
                        model_channels=cfg['model_channels'],
                        channel_mult=cfg['channel_mult'], dropout=cfg['dropout'],
                        img_resolution=cfg['img_size'], label_dim=cfg['label_dim'],
                        embedding_type='positional', encoder_type='standard',
                        decoder_type='standard', augment_dim=cfg['augment_dim'],
                        channel_mult_noise=1, resample_filter=[1, 1])
    unet.load_state_dict(torch.load(os.path.join(run, 'model_final.pth'), map_location=DEV))
    unet = unet.to(DEV).eval()
    sd = 0.5

    def D(y, s, lab):
        sv = s.view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (sv ** 2 + sd ** 2)
        c_out = sv * sd / (sv ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sv ** 2).sqrt()
        return c_skip * y + c_out * unet(c_in * y, (s.log() / 4).view(-1), cond=None)
    print(f"  U-net: {run}", flush=True)
    return D, False


def pca(X):
    """Pool mean and covariance eigenpairs, EDM units (X already in [-1,1])."""
    mu = X.mean(0)
    Xc = X - mu
    ev, U = torch.linalg.eigh((Xc.T @ Xc) / X.shape[0])
    return mu, ev.clamp_min(0.0), U


def per_image_loss(xh_edm, x0_edm):
    """[0,1]-pixel-unit per-image summed squared error from EDM-unit tensors."""
    return (((xh_edm - x0_edm).reshape(x0_edm.shape[0], -1) ** 2).sum(1) / 4.0).cpu().numpy()


# ----------------------------------------------------------------------------------------
def by_class(fn, Y, labs, Xp, lp):
    """Apply fn(Y_sub, pool_sub) per class; COND=0 passes the whole pool once."""
    if not COND:
        return fn(Y, Xp)
    out = None
    for c in torch.unique(labs):
        sel = labs == c
        r = fn(Y[sel], Xp[lp == c])
        if out is None:
            out = ({k: torch.empty((Y.shape[0],) + v.shape[1:], device=DEV, dtype=v.dtype)
                    for k, v in r.items()} if isinstance(r, dict)
                   else torch.empty((Y.shape[0],) + r.shape[1:], device=DEV, dtype=r.dtype))
        if isinstance(r, dict):
            for k, v in r.items():
                out[k][sel] = v
        else:
            out[sel] = r
    return out


def main():
    Xp, lp, Xv, lv, Xt, lt, desc = load_data()
    E = lambda X: (2.0 * X - 1.0).reshape(X.shape[0], d)        # EDM units, flattened
    Pp, Pv, Pt = E(Xp), E(Xv), E(Xt)
    mu, ev, U = pca(Pp)
    Q = torch.cat([Pt, Pv])                                     # every arm denoises test+val
    lq = torch.cat([lt, lv])
    nt = Pt.shape[0]
    print(f"LOCAL SOFTMAX HELD OUT  {DATASET}  {desc}  COND={COND}  arms={ARMS}  dev={DEV}",
          flush=True)

    store = {}
    if os.path.exists(OUT):
        store = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}
        print(f"resuming from {OUT}: {len(store)} keys", flush=True)
    store.update(meta_NIMG=np.array(NIMG), meta_NTEST=np.array(NTEST),
                 meta_NVAL=np.array(NVAL), meta_COND=np.array(COND),
                 meta_DATASET=np.array(DATASET),
                 # E||x0 - mean||^2 on the test subset, [0,1] pixel units: the sigma -> inf level
                 meta_trace_test=np.array(float(((Pt - Pt.mean(0)) ** 2).sum(1).mean()) / 4))

    def put(key, xh, t0, neff=None):
        l = per_image_loss(xh, Q)
        store[f'test:{key}'], store[f'val:{key}'] = l[:nt], l[nt:]
        if neff is not None:
            store[f'neff:{key}'] = neff[:nt].cpu().numpy()
        np.savez(OUT, **store)
        wt = store.get(f'test:wiener|{key.split("|")[1]}')
        rel = (f"  vs W_U {l[:nt].mean() - wt.mean():+8.3f} "
               f"+- {np.std(l[:nt] - wt, ddof=1) / np.sqrt(nt):.3f}") if wt is not None else ''
        ne = f"  N_eff~{float(neff[:nt].median()):.1f}" if neff is not None else ''
        print(f"  {key:28s} test {l[:nt].mean():9.3f} +- {l[:nt].std(ddof=1)/np.sqrt(nt):.3f}"
              f"   val {l[nt:].mean():9.3f}{rel}{ne}   [{time.time()-t0:.0f}s]", flush=True)

    have = lambda key: f'test:{key}' in store
    nets = load_nets() if 'edm' in ARMS else {}
    store['meta_NETS'] = np.array(list(nets))
    store['meta_TEST_START'] = np.array(TEST_START if DATASET != 'cifar10' else 0)
    # the U-net trained on exactly the pool, for the train = eval network curve
    pool_net = nets.get('unet-10000_split1') if NIMG == 10000 else None

    for sg in SIGS:
        se = 2.0 * sg                                           # sigma in EDM units
        # noise is drawn at a CANONICAL size and sliced, so the draw seen by test image i (or
        # val image i) does not depend on NTEST / NVAL: a 200-image ELS run is exactly paired
        # with a 1000-image run of the cheap arms stored in another file.
        g = torch.Generator(device=DEV).manual_seed(5100 + int(round(sg * 1000)))
        zt = torch.randn((10000, d), generator=g, device=DEV, dtype=DT)[:nt]
        g = torch.Generator(device=DEV).manual_seed(7100 + int(round(sg * 1000)))
        zv = torch.randn((1000, d), generator=g, device=DEV, dtype=DT)[:Pv.shape[0]]
        Y = Q + se * torch.cat([zt, zv])
        # GLOBAL softmax, TRAIN = EVAL: the first nt POOL images, noised with the test draws,
        # denoised by the whole-pool posterior mean (self-inclusive -- the pool contains the
        # image), next to the in-sample Wiener on the same y.  LS with a whole-image window IS
        # the global softmax (asserted in _validate).
        if 'insample' in ARMS and f'test:global_insample|{sg}' not in store:
            t0 = time.time()
            Yi = Pp[:nt] + se * zt
            xg = masked_softmax_denoise(Yi, Pp, se, ls_window_mask(H, W, 2 * H - 1, DEV, DT),
                                        pixel_mask=True, C=C)
            store[f'test:global_insample|{sg}'] = per_image_loss(xg, Pp[:nt])
            store[f'test:wiener_insample|{sg}'] = per_image_loss(
                wiener_denoise(Yi, mu, ev, U, se), Pp[:nt])
            if pool_net is not None:        # the matched U-net on its OWN training images
                with torch.no_grad():
                    xu = torch.cat([pool_net[0](
                        Yi[a:a + 500].view(-1, C, H, W).float(),
                        torch.full([len(Yi[a:a + 500])], se, device=DEV), None)
                        .to(DT).reshape(-1, d) for a in range(0, nt, 500)])
                store[f'test:unet_insample|{sg}'] = per_image_loss(xu, Pp[:nt])
                print(f"  unet_insample|{sg}   {store[f'test:unet_insample|{sg}'].mean():9.4f}",
                      flush=True)
            np.savez(OUT, **store)
            print(f"  global_insample|{sg}          "
                  f"{store[f'test:global_insample|{sg}'].mean():9.4f}   in-sample Wiener "
                  f"{store[f'test:wiener_insample|{sg}'].mean():9.4f}   [{time.time()-t0:.0f}s]",
                  flush=True)
            del Yi, xg
        del zt, zv
        print(f"\n=== sigma={sg} (sigma_edm={se:g}) ===", flush=True)

        if 'wiener' in ARMS:
            t0 = time.time()
            if not have(f'wiener|{sg}'):
                put(f'wiener|{sg}', wiener_denoise(Y, mu, ev, U, se), t0)
            if COND and not have(f'wienerC|{sg}'):
                def wc(Yc, Xc):
                    m, e, u = pca(Xc)
                    return wiener_denoise(Yc, m, e, u, se)
                put(f'wienerC|{sg}', by_class(wc, Y, lq, Pp, lp), t0)

        if 'ls' in ARMS:
            for k in KS_LS:
                key = f'ls|{sg}|{k}'
                if have(key):
                    continue
                t0 = time.time()
                M = ls_window_mask(H, W, k, device=DEV, dtype=DT)
                r = by_class(lambda Yc, Xc: dict(zip(('x', 'n'), masked_softmax_denoise(
                    Yc, Xc, se, M, pixel_mask=True, C=C, return_neff=True))), Y, lq, Pp, lp)
                put(key, r['x'], t0, r['n'])

        if 'luk' in ARMS:
            for tau in TAUS:
                key = f'luk|{sg}|{tau:g}|{TAU_MODE}'
                if have(key):
                    continue
                t0 = time.time()
                M = wiener_locality_mask(ev, U, se, tau, TAU_MODE)
                r = by_class(lambda Yc, Xc: dict(zip(('x', 'n'), masked_softmax_denoise(
                    Yc, Xc, se, M, pixel_mask=False, C=C, pool_chunk=512,
                    return_neff=True))), Y, lq, Pp, lp)
                put(key, r['x'], t0, r['n'])
                print(f"      mask support/row: median {float(M.sum(1).median()):.0f} "
                      f"of {d}", flush=True)

        if 'els' in ARMS:
            todo = [k for k in KS_ELS if not have(f'els|{sg}|{k}')]
            if todo:
                t0 = time.time()
                r = by_class(lambda Yc, Xc: els_denoise(
                    Yc.view(-1, C, H, W), Xc.view(-1, C, H, W), se, todo,
                    pool_chunk=ELS_CHUNK, els_dtype=ELS_DT), Y, lq, Pp, lp)
                for k in todo:
                    put(f'els|{sg}|{k}', r[k].reshape(-1, d), t0)

        for tag, (net, net_lab) in nets.items():
            if have(f'{tag}|{sg}'):
                continue
            t0 = time.time()
            lab = torch.nn.functional.one_hot(lq, 10).to(torch.float32) if net_lab else None
            outs = []
            with torch.no_grad():
                for a in range(0, Y.shape[0], 500):
                    yb = Y[a:a + 500].view(-1, C, H, W).to(torch.float32)
                    sv = torch.full([yb.shape[0]], se, device=DEV, dtype=torch.float32)
                    outs.append(net(yb, sv, None if lab is None else lab[a:a + 500])
                                .to(DT).reshape(-1, d))
            put(f'{tag}|{sg}', torch.cat(outs), t0)
    print("\ndone", flush=True)


# ----------------------------------------------------------------------------------------
def _oracle(y, A, s):
    """Global empirical-Bayes posterior mean, the definition in bayes_oracle_heldout.py."""
    lw = -((y[:, None, :] - A[None]) ** 2).sum(-1) / (2 * s ** 2)
    return torch.softmax(lw, 1) @ A


def _validate():
    g = torch.Generator(device=DEV).manual_seed(3)
    c, h, w, n, b = 2, 5, 5, 13, 4
    dd = c * h * w
    A = torch.randn(n, dd, generator=g, device=DEV, dtype=DT)
    Y = torch.randn(b, dd, generator=g, device=DEV, dtype=DT) * 1.2
    ok = lambda a, r, tol=1e-10: float((a - r).abs().max() / r.abs().max()) < tol

    for s in (0.3, 1.1):
        ref = _oracle(Y, A, s)
        # (a) LS with a window covering the whole image (zero padding => k >= 2h-1) = oracle
        got = masked_softmax_denoise(Y, A, s, ls_window_mask(h, w, 2 * h - 1, DEV), True, C=c,
                                     pool_chunk=5, query_chunk=3)
        print(f"  LS global window == oracle  s={s}: {ok(got, ref)}"); assert ok(got, ref)
        # (b) LUK with tau = 0 (all-ones mask) = oracle
        mu, ev, U = pca(A)
        got = masked_softmax_denoise(Y, A, s, wiener_locality_mask(ev, U, s, 0.0), False, C=c,
                                     pool_chunk=4)
        print(f"  LUK tau=0 == oracle         s={s}: {ok(got, ref)}"); assert ok(got, ref)
        # (c) ELS with k = h on the torus = oracle over every circular shift of every atom
        As = torch.stack([torch.roll(A.view(n, c, h, w), (i, j), (2, 3))
                          for i in range(h) for j in range(w)], 1).reshape(-1, dd)
        got = els_denoise(Y.view(b, c, h, w), A.view(n, c, h, w), s, [h], pool_chunk=4)[h]
        rs = _oracle(Y, As, s)
        print(f"  ELS k=h == shift-aug oracle s={s}: {ok(got.reshape(b, dd), rs)}")
        assert ok(got.reshape(b, dd), rs)

    # (d) brute force, explicit loops, at a genuinely local k
    s, k = 0.5, 3
    r = k // 2
    Yi, Ai = Y.view(b, c, h, w), A.view(n, c, h, w)
    ls_bf = torch.zeros_like(Yi); els_bf = torch.zeros_like(Yi)
    for q in range(b):
        for i in range(h):
            for j in range(w):
                # LS: zero padding = window truncated at the border, same location
                lg = []
                for m in range(n):
                    t = 0.0
                    for di in range(-r, r + 1):
                        for dj in range(-r, r + 1):
                            if 0 <= i + di < h and 0 <= j + dj < w:
                                t += float(((Yi[q, :, i + di, j + dj]
                                             - Ai[m, :, i + di, j + dj]) ** 2).sum())
                    lg.append(-t / (2 * s ** 2))
                wv = torch.softmax(torch.tensor(lg, dtype=DT, device=DEV), 0)
                ls_bf[q, :, i, j] = (wv[:, None] * Ai[:, :, i, j]).sum(0)
                # ELS: circular, every train location
                lg, val = [], []
                for m in range(n):
                    for vi in range(h):
                        for vj in range(w):
                            t = 0.0
                            for di in range(-r, r + 1):
                                for dj in range(-r, r + 1):
                                    t += float(((Yi[q, :, (i + di) % h, (j + dj) % w]
                                                 - Ai[m, :, (vi + di) % h, (vj + dj) % w])
                                                ** 2).sum())
                            lg.append(-t / (2 * s ** 2)); val.append(Ai[m, :, vi, vj])
                wv = torch.softmax(torch.tensor(lg, dtype=DT, device=DEV), 0)
                els_bf[q, :, i, j] = (wv[:, None] * torch.stack(val)).sum(0)
    got = masked_softmax_denoise(Y, A, s, ls_window_mask(h, w, k, DEV), True, C=c, pool_chunk=4)
    print(f"  LS brute force k={k}: {ok(got, ls_bf.reshape(b, dd))}")
    assert ok(got, ls_bf.reshape(b, dd))
    for dt, tol in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
        got = els_denoise(Yi, Ai, s, [k, 5], pool_chunk=3, els_dtype=dt)[k]
        print(f"  ELS brute force k={k} ({dt}): {ok(got, els_bf, tol)}")
        assert ok(got, els_bf, tol)

    # (e) sigma -> infinity: LS -> pool mean per coordinate; ELS -> per-channel grand mean
    big = 1e4
    got = masked_softmax_denoise(Y, A, big, ls_window_mask(h, w, 3, DEV), True, C=c)
    print(f"  LS sigma->inf == pool mean: {ok(got, A.mean(0).expand_as(got), 1e-6)}")
    assert ok(got, A.mean(0).expand_as(got), 1e-6)
    got = els_denoise(Yi, Ai, big, [3])[3]
    gm = Ai.mean((0, 2, 3))[None, :, None, None].expand_as(got)
    print(f"  ELS sigma->inf == channel grand mean: {ok(got, gm, 1e-6)}")
    assert ok(got, gm, 1e-6)

    # (f) N_eff: a single atom has N_eff = 1; a flat posterior has N_eff = n
    _, ne = masked_softmax_denoise(Y, A[:1], 0.5, ls_window_mask(h, w, 3, DEV), True, C=c,
                                   return_neff=True)
    _, ne2 = masked_softmax_denoise(Y, A, big, ls_window_mask(h, w, 3, DEV), True, C=c,
                                    pool_chunk=4, return_neff=True)
    print(f"  N_eff one atom = {float(ne.max()):.6f}, flat = {float(ne2.min()):.4f} / {n}")
    assert abs(float(ne.max()) - 1) < 1e-9 and abs(float(ne2.min()) - n) < 1e-3

    # (g) Lukoianov mask at sigma -> 0 is the identity (W -> I): each coordinate sees itself
    mu, ev, U = pca(A)          # rank-deficient (n < dd), so use a full-rank toy here
    B2 = torch.randn(200, dd, generator=g, device=DEV, dtype=DT)
    mu, ev, U = pca(B2)
    M0 = wiener_locality_mask(ev, U, 1e-4, 0.5)
    print(f"  LUK mask sigma->0 is identity: {bool(torch.equal(M0, torch.eye(dd, device=DEV, dtype=DT)))}")
    assert torch.equal(M0, torch.eye(dd, device=DEV, dtype=DT))
    print("all validations passed")


if __name__ == '__main__':
    if int(os.environ.get('VALIDATE', '0')):
        _validate()
    else:
        main()
