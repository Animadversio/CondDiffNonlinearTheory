"""SAMPLING WITH THE ANALYTIC DENOISERS vs U-NETS: Heun, 30 steps, paired seeds (2026-09-29).

    michimin: "run another batch of test on generated sample using these different denoisers
    (linear, LS, Lukianov, softmax all) using Heun sampler 30 steps.  these denoisers vs UNet.
    cache mid results in the STORE DIR; summary stats tables figure go into the github repo.
    Try CIFAR and FFHQ at least"

THE SAMPLER.  EDM's deterministic Heun sampler (Karras et al. 2022, Alg. 1 with S_churn = 0):
30 steps, sigma_edm from 80 to 0.002 with rho = 7, then a final Euler step to 0.  Every method
starts from the SAME x_T = 80 z for sample i, so each analytic sample has a U-net twin.

THE DENOISERS, all in EDM units (x in [-1,1]), pool = the 10,000 images of the held-out study:
  wiener   x_hat = mu + W (y - mu), W from the pool covariance
  ls       Kamb & Ganguli LS, window k(sigma)          } k / tau at each step = the value
  luk      Lukoianov et al., Wiener mask tau(sigma)    } VAL-selected in the held-out tables
                                                          at the nearest grid sigma (log)
  global   empirical-Bayes softmax over the whole pool (LS with a whole-image window)
  unet-*   U-nets (EDM preconditioning):
           cifar10: edm-uncond-vp (official, 50k train, EMA) and 10000_split1 (DiffusionLearning
                    Curve SongUNet trained on EXACTLY the pool)
           ffhq32:  10000_split1 (trained on exactly the pool) and full_longtrain (all 70k, 250k
                    steps)
So k and tau follow the MSE-optimal schedule, NOT Kamb & Ganguli's U-net-matched calibration.

METRICS (per sample, then summarised):
  1. on-trajectory agreement: along each reference U-net's own trajectory, apply every other
     denoiser to the U-net's x_t and compare with the U-net's D(x_t):
         R^2_t = 1 - ||D_m(x_t) - D_u(x_t)||^2 / ||D_u(x_t) - mean_pix D_u(x_t)||^2
     (analytic-diffusion-studio's per-sample R^2 with the U-net as target), and the MSE.
  2. final samples: MSE and R^2 of x_m against the U-net sample from the same seed.
  3. copying: L2 distance of each final sample (clipped to [-1,1]) to its nearest and second
     nearest pool image; "copy" = d1/d2 < 1/3 (the standard memorisation ratio test), and d1^2
     next to the held-out test images' own d1^2 (what a novel image looks like).
Losses in [0,1] pixel units, summed over the 3072 coordinates (= the held-out tables' units).

CACHE (STORE_DIR/CondDiffNonlinearTheory/local_softmax_samples/{dataset}/N{n}_steps{s}_seed{k}/):
  {method}.pt       final sample + full trajectory: x_t and D(x_t) at each of the 30 steps
  ontraj_{m}_on_{u}.pt   D_m evaluated on U-net u's trajectory states
REPO:  tables/local_softmax_samples_{dataset}.npz  (every per-sample metric + the schedules)
       figures/local_softmax_samples_{dataset}_{grid,metrics}.{png,pdf}, *_summary.csv

    DATASET=cifar10 python scripts/local_softmax_sampling.py
    DATASET=ffhq32 NSAMP=64 python scripts/local_softmax_sampling.py      # smaller
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

DATASET = os.environ.setdefault('DATASET', 'cifar10')
os.environ.setdefault('TEST_START', {'cifar10': '10000', 'ffhq32': '60000',
                                     'afhq32': '10000'}[DATASET])
os.environ.setdefault('NTEST', '1000')
os.environ.setdefault('NVAL', '200')
import scripts.local_softmax_heldout as L
from core.local_softmax import (ls_window_mask, wiener_locality_mask, masked_softmax_denoise,
                                wiener_denoise)

DEV, DT, C, H, W, d = L.DEV, L.DT, L.C, L.H, L.W, L.d
NSAMP = int(os.environ.get('NSAMP', '256'))
STEPS = int(os.environ.get('STEPS', '30'))
SEED = int(os.environ.get('SEED', '0'))
SMIN, SMAX, RHO = 0.002, 80.0, 7.0
UNETS = os.environ.get('SAMPLE_UNETS', {'cifar10': 'edm,10000_split1',
                                        'ffhq32': '10000_split1,full_longtrain',
                                        'afhq32': '10000_split1,full'}[DATASET]).split(',')
ANALYTIC = os.environ.get('METHODS', 'wiener,ls,luk,global').split(',')
CACHE = os.path.join(L.STORE, 'CondDiffNonlinearTheory', 'local_softmax_samples', DATASET,
                     f'N{NSAMP}_steps{STEPS}_seed{SEED}')      # every cached file is keyed on these
TABLE = ('tables/local_softmax_heldout.npz' if DATASET == 'cifar10'
         else f'tables/local_softmax_heldout_{DATASET}.npz')
OUT = os.environ.get('OUT', f'tables/local_softmax_samples_{DATASET}.npz')


def sigma_steps():
    i = torch.arange(STEPS, dtype=torch.float64)
    t = (SMAX ** (1 / RHO) + i / (STEPS - 1) * (SMIN ** (1 / RHO) - SMAX ** (1 / RHO))) ** RHO
    return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).tolist()


def schedule(arm):
    """{grid sigma (pixel): val-selected param} from the held-out table, over every param
    present at that sigma (the held-out plot's selection, minus its completeness guard: the
    tables are complete now)."""
    A = dict(np.load(TABLE, allow_pickle=True))
    out = {}
    for k in A:
        if not (k.startswith(f'val:{arm}|')):
            continue
        _, s, p = k.split('|', 2)
        if arm == 'ls' and p == '63':          # whole-image window = the global arm
            continue
        s = float(s)
        if s not in out or A[k].mean() < out[s][1]:
            out[s] = (p, float(A[k].mean()))
    return {s: v[0] for s, v in sorted(out.items())}


def nearest(sched, s_pix):
    g = np.array(list(sched))
    return sched[float(g[np.argmin(np.abs(np.log(g) - np.log(s_pix)))])]


def build():
    """{name: D(x (B,C,H,W) float64 EDM units, sigma_edm float) -> x0-hat float64}."""
    Xp, *_ = L.load_data()
    Pp = (2.0 * Xp - 1.0).reshape(-1, d)
    mu, ev, U = L.pca(Pp)
    Mg = ls_window_mask(H, W, 2 * H - 1, DEV, DT)
    ls_s, luk_s = schedule('ls'), schedule('luk')
    masks = {}
    flat = lambda f: (lambda x, s: f(x.reshape(x.shape[0], d), s).reshape(x.shape))
    D = {
        'wiener': flat(lambda y, s: wiener_denoise(y, mu, ev, U, s)),
        'global': flat(lambda y, s: masked_softmax_denoise(y, Pp, s, Mg, True, C=C)),
    }

    def ls(y, s):
        k = int(nearest(ls_s, s / 2))
        if k not in masks:
            masks[k] = ls_window_mask(H, W, k, DEV, DT)
        return masked_softmax_denoise(y, Pp, s, masks[k], True, C=C)

    def luk(y, s):
        tau, mode = nearest(luk_s, s / 2).split('|')
        return masked_softmax_denoise(y, Pp, s, wiener_locality_mask(ev, U, s, float(tau), mode),
                                      False, C=C, pool_chunk=512)
    D['ls'], D['luk'] = flat(ls), flat(luk)
    for u in UNETS:
        net, _ = L.load_net(None if u == 'edm' else u)

        def Du(x, s, net=net):
            out = []
            with torch.no_grad():
                for a in range(0, x.shape[0], 256):
                    xb = x[a:a + 256].float()
                    out.append(net(xb, torch.full([len(xb)], s, device=DEV), None).to(DT))
            return torch.cat(out)
        D[f'unet-{u}'] = Du
    D = {k: v for k, v in D.items() if k in ANALYTIC or k.startswith('unet-')}
    return D, Pp, {'ls': ls_s, 'luk': luk_s}


@torch.no_grad()
def heun(Dm, x_T, ts):
    """EDM Alg. 1, deterministic.  Returns final x and per-step (x_t, D(x_t)) at t_cur."""
    x = x_T.clone()
    xs, ds = [], []
    for i, (tc, tn) in enumerate(zip(ts[:-1], ts[1:])):
        den = Dm(x, tc)
        xs.append(x.float().cpu()); ds.append(den.float().cpu())
        dcur = (x - den) / tc
        xn = x + (tn - tc) * dcur
        if tn > 0:
            dn = (xn - Dm(xn, tn)) / tn
            xn = x + (tn - tc) * (0.5 * dcur + 0.5 * dn)
        x = xn
    return x, torch.stack(xs), torch.stack(ds)


def r2(pred, tgt):
    """Per-sample R^2 of pred against target, over all coordinates (tensors (N, ...))."""
    p, t = pred.reshape(pred.shape[0], -1).double(), tgt.reshape(tgt.shape[0], -1).double()
    ss_res = ((p - t) ** 2).sum(1)
    ss_tot = ((t - t.mean(1, keepdim=True)) ** 2).sum(1)
    return (1 - ss_res / ss_tot).numpy()


def mse_pix(a, b):
    """[0,1]-pixel-unit per-sample summed squared error from EDM-unit tensors."""
    return (((a.double() - b.double()).reshape(a.shape[0], -1) ** 2).sum(1) / 4).numpy()


def nn_stats(X, pool, chunk=256):
    """L2 distance (pixel units) of each row to its nearest and 2nd nearest pool row."""
    Xp = ((X.to(DEV, DT).reshape(X.shape[0], -1).clamp(-1, 1)) + 1) / 2
    P = (pool + 1) / 2
    d1, d2 = [], []
    for a in range(0, Xp.shape[0], chunk):
        dd = torch.cdist(Xp[a:a + chunk], P)
        v = dd.topk(2, dim=1, largest=False).values
        d1.append(v[:, 0]); d2.append(v[:, 1])
    return torch.cat(d1).cpu().numpy(), torch.cat(d2).cpu().numpy()


def main():
    os.makedirs(CACHE, exist_ok=True)
    D, Pp, sched = build()
    ts = sigma_steps()
    g = torch.Generator(device=DEV).manual_seed(9000 + SEED)
    x_T = torch.randn((NSAMP, C, H, W), generator=g, device=DEV, dtype=DT) * ts[0]
    print(f"SAMPLING {DATASET}  N={NSAMP}  Heun {STEPS} steps  sigma_edm {SMAX}->{SMIN} rho={RHO}"
          f"  methods={list(D)}\n  cache {CACHE}", flush=True)
    for arm in ('ls', 'luk'):
        print(f"  {arm} schedule (pixel sigma -> param): {sched[arm]}", flush=True)

    # 1. sample every method from the same x_T (cached)
    res = {}
    for m, Dm in D.items():
        f = os.path.join(CACHE, f'{m}.pt')
        if os.path.exists(f) and torch.load(f)['x_T_hash'] == float(x_T.sum()):
            res[m] = torch.load(f)
            print(f"  {m:22s} cached", flush=True)
            continue
        t0 = time.time()
        x, xs, ds = heun(Dm, x_T, ts)
        res[m] = dict(x=x.float().cpu(), xs=xs, ds=ds, ts=ts, x_T_hash=float(x_T.sum()))
        torch.save(res[m], f)
        print(f"  {m:22s} sampled  [{time.time() - t0:.0f}s]", flush=True)

    # 2. on-trajectory agreement with each U-net (cached)
    store = {'ts': np.array(ts[:-1]), 'meta_NSAMP': np.array(NSAMP),
             'meta_STEPS': np.array(STEPS), 'meta_methods': np.array(list(D)),
             'meta_unets': np.array([m for m in D if m.startswith('unet-')])}
    for arm in ('ls', 'luk'):
        store[f'sched_{arm}_sigma'] = np.array(list(sched[arm]))
        store[f'sched_{arm}_param'] = np.array(list(sched[arm].values()))
    unets = [m for m in D if m.startswith('unet-')]
    for u in unets:
        xs_u, ds_u = res[u]['xs'], res[u]['ds']
        for m, Dm in D.items():
            if m == u:
                continue
            f = os.path.join(CACHE, f'ontraj_{m}_on_{u}.pt')
            if os.path.exists(f):
                dm = torch.load(f)
            else:
                t0 = time.time()
                dm = torch.stack([Dm(xs_u[i].to(DEV, DT), ts[i]).float().cpu()
                                  for i in range(STEPS)])
                torch.save(dm, f)
                print(f"  on-traj {m} on {u}  [{time.time() - t0:.0f}s]", flush=True)
            store[f'onr2:{m}|{u}'] = np.stack([r2(dm[i], ds_u[i]) for i in range(STEPS)])
            store[f'onmse:{m}|{u}'] = np.stack([mse_pix(dm[i], ds_u[i]) for i in range(STEPS)])
            store[f'finr2:{m}|{u}'] = r2(res[m]['x'].clamp(-1, 1), res[u]['x'].clamp(-1, 1))
            store[f'finmse:{m}|{u}'] = mse_pix(res[m]['x'].clamp(-1, 1), res[u]['x'].clamp(-1, 1))

    # 3. copying: nearest pool image
    for m in D:
        d1, d2 = nn_stats(res[m]['x'], Pp)
        store[f'nn1:{m}'], store[f'nn2:{m}'] = d1, d2
    Xp, _, _, _, Xt, _, _ = L.load_data()
    d1, d2 = nn_stats((2 * Xt - 1).float(), Pp)
    store['nn1:test'], store['nn2:test'] = d1, d2
    np.savez(OUT, **store)

    print(f"\n{'method':22s} {'copy%':>6s} {'med d1^2':>9s} | " +
          ' | '.join(f'finR2 vs {u:>16s}  onR2(mean)' for u in unets))
    for m in list(D) + ['test']:
        c = 100 * np.mean(store[f'nn1:{m}'] / store[f'nn2:{m}'] < 1 / 3)
        row = f"{m:22s} {c:6.1f} {np.median(store[f'nn1:{m}'] ** 2):9.2f} | "
        cells = []
        for u in unets:
            if m in (u, 'test'):
                cells.append(f"{'--':>28s}")
            else:
                cells.append(f"{np.mean(store[f'finr2:{m}|{u}']):8.3f} "
                             f"{np.mean(store[f'onr2:{m}|{u}']):19.3f}")
        print(row + ' | '.join(cells), flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == '__main__':
    main()
