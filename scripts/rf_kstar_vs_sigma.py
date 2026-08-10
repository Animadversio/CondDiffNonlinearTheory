"""
rf_kstar_vs_sigma.py — k_*(sigma) for MNIST / CIFAR-10, plotted against the gap an RF
denoiser could actually close. Four panels: top row = the gaps, bottom row = the thresholds.

Supersedes scripts/rf_kstar_for_dnn_experiment.py, in three ways.

1. NO SHRINKAGE. The defect is computed in the Cov(r)-weighted (inverse-free) form of
   Remark rem:covr / eq:invfree:
       Cov(r) Delta = sigma^2 Sigma_y^{-1} Delta^raw,
       Delta^raw    = Cov(x0, psi_s(theta^T x0 + eps)) - alphabar(theta,eps) Sigma theta,
   so the only inversion is Sigma_y = Sigma + sigma^2 I, which is well conditioned for every
   sigma > 0. The old script needed a shrinkage sweep on Sigma^{-1} and reported a threshold
   that moved by an order of magnitude with the shrinkage level. MNIST's Sigma is EXACTLY
   singular (71 always-black border pixels), so there the old form was not merely
   ill-conditioned but undefined.

2. THE QUANTITY PLOTTED IS THE OPERATIONAL THRESHOLD, not d/hat_eps^2. The writeup's
   k_* = d/hat_eps_d^2 drops the eps_hat -> eps_check conversion Xi = ~1/gamma, which is
   d-independent (so all d-scaling results stand) but is ~1e4 at small sigma for ReLU, since
   the Mehler band energy scales like gamma ~ s^{2 n_0}, s = sigma ||theta||. We therefore
   plot the constant-complete
       k(gap <= tau) = tau * d / (4 * check_eps_w^2)     [rho_* = delta = 1/2]
   and show d/hat_eps_w^2 only as a dashed reference.

3. THE LOW-SIGMA GAP IS MARKED AS AN ARTIFACT. An "oracle Bayes" curve computed from a
   finite sample is the Bayes denoiser for the ATOMIC empirical measure. Once
   sigma*sqrt(d) falls below the typical nearest-neighbour distance, the posterior collapses
   onto a single training point and the reported MMSE -> 0 by memorisation, not because the
   population MMSE is small. Below that sigma the plotted "gap" is not a gap. This matters
   because the honest analytic statement runs the other way: as sigma -> 0 the linear
   denoiser is asymptotically OPTIMAL (both L_lin and L_Bayes are d sigma^2 - O(sigma^4)),
   so there is asymptotically nothing for a nonlinear denoiser to close there.

    python scripts/rf_kstar_vs_sigma.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import factorial
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
N_THETA = int(os.environ.get('N_THETA', '2048'))
NIMG    = int(os.environ.get('NIMG', '50000'))
N0, NB  = 2, 6
TOLS    = [0.1, 0.01]
CIFAR_ROOT = '/n/home12/binxuwang/.keras/datasets'
MNIST_ROOT = os.path.expanduser('~/.keras/datasets')
TBL = {'CIFAR-10': 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz',
       'MNIST':    'tables/dnn_feature_mmse_mnist_N10000_noise5_sigma30.npz'}
SIG_LO, SIG_HI = 0.02, 10.0          # window where the figure's gap lives
NEFF_MIN = 2.0                       # posterior must spread over >=2 atoms to count as genuine
N_SIG = int(os.environ.get('N_SIG', '12'))


def load(name):
    import torchvision, torchvision.transforms as T
    tf = T.ToTensor()                       # raw [0,1]; matches dnn_feature_mmse.py
    ds = (torchvision.datasets.MNIST(MNIST_ROOT, train=True, download=False, transform=tf)
          if name == 'MNIST' else
          torchvision.datasets.CIFAR10(CIFAR_ROOT, train=True, download=False, transform=tf))
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, num_workers=2)
    out = []
    for xb, _ in dl:
        out.append(xb.reshape(xb.shape[0], -1))
        if sum(o.shape[0] for o in out) >= NIMG:
            break
    X = torch.cat(out)[:NIMG].to(DEV, DT)
    g = torch.Generator(device=DEV); g.manual_seed(5)
    return X[torch.randperm(X.shape[0], device=DEV, generator=g)]


def _cn(M, s, n):
    """ReLU Hermite coefficient c_n(M,s), n >= 2."""
    z = M / s
    he = [torch.ones_like(z), z]
    for j in range(2, n - 1):
        he.append(z * he[-1] - (j - 1) * he[-2])
    return ((-1) ** n) * s * he[n - 2] * torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / factorial(n)


def posterior_neff(X, sigmas, n_query=256, n_atom=10000, seed=3):
    """Effective number of training atoms carrying the empirical-measure posterior.

    The 'oracle Bayes' curve of dnn_feature_mmse.py is the Bayes denoiser for the ATOMIC
    empirical measure over n_atom images. Its MMSE tends to 0 by memorisation whenever the
    posterior p(x_j | y) concentrates on the single atom that generated y. We measure that
    directly rather than by a distance heuristic: N_eff := exp(H) with H the Shannon entropy
    of the posterior weights w_j proportional to exp(-||y - x_j||^2 / 2 sigma^2).

        N_eff ~ 1   -> posterior is a point mass; the reported MMSE is memorisation
        N_eff >> 1  -> genuine posterior spread; the reported MMSE is meaningful
    """
    A = X[:n_atom]
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    idx = torch.randperm(A.shape[0], device=DEV, generator=g)[:n_query]
    x0 = A[idx]
    Z = torch.randn(x0.shape, device=DEV, dtype=DT, generator=g)
    neff = []
    for sg in sigmas:
        y = x0 + sg * Z
        logw = -torch.cdist(y, A) ** 2 / (2 * sg ** 2)
        logw = logw - logw.max(1, keepdim=True).values
        w = torch.softmax(logw, dim=1)
        H = -(w * torch.log(w.clamp(min=1e-300))).sum(1)
        neff.append(float(torch.exp(H).median()))
    return np.array(neff)


def defect_curves(X, Theta, sigmas):
    """Split-half, no shrinkage. Returns (hat_eps_w^2, check_eps_w^2, mean gamma) per sigma."""
    n, d = X.shape
    half = n // 2
    I = torch.eye(d, device=DEV, dtype=DT)
    tn = torch.linalg.norm(Theta, dim=1)
    ok = (tn >= 0.5) & (tn <= 2.0)                       # the ||theta|| in [1/2,2] indicator
    Dw = [[None, None] for _ in sigmas]
    gam = [None] * len(sigmas)
    for h, Xh in enumerate((X[:half], X[half:2 * half])):
        mu = Xh.mean(0); Xc = Xh - mu
        Sig = (Xc.T @ Xc) / Xh.shape[0]
        U = Xh @ Theta.T                                  # sigma-independent: hoisted
        SigTh = Sig @ Theta.T
        for si, sg in enumerate(sigmas):
            s = sg * tn
            z = U / s[None, :]
            Phi = torch.special.ndtr(z)
            ph = torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
            cc = U * Phi + s[None, :] * ph                # psi_s(u)
            Cv = (Xc.T @ (cc - cc.mean(0))) / Xh.shape[0] # Cov(x0, psi)
            Draw = Cv - SigTh * Phi.mean(0)[None, :]      # Delta^raw = Sigma Delta
            Syi = torch.linalg.inv(Sig + sg ** 2 * I)     # only inversion; Sy >= sigma^2 I
            Dw[si][h] = sg ** 2 * (Syi @ Draw)            # Cov(r) Delta
            if h == 0:
                acc = torch.zeros_like(U)
                for nn in range(N0, NB + 1):
                    acc += factorial(nn) * _cn(U, s[None, :], nn) ** 2
                gam[si] = acc.mean(0)
            del Syi, Draw, Cv, cc, Phi, ph, z
        del Sig, U, SigTh, Xc
        torch.cuda.empty_cache()
    hat, chk, gmean = [], [], []
    for si in range(len(sigmas)):
        nw = (Dw[si][0] * Dw[si][1]).sum(0)               # unbiased ||Cov(r)Delta||^2
        hat.append(float(np.sqrt(max(float((nw.clamp(min=0) ** 2).mean()), 0.0))))
        chk.append(float((nw[ok] / gam[si][ok].clamp(min=1e-300)).mean()))
        gmean.append(float(gam[si].mean()))
    return np.array(hat), np.array(chk), np.array(gmean)


def main():
    sigmas = np.exp(np.linspace(np.log(SIG_LO), np.log(SIG_HI), N_SIG))
    R = {}
    for name in ('MNIST', 'CIFAR-10'):
        tab = np.load(TBL[name], allow_pickle=True)
        X = load(name); d = X.shape[1]
        ts_all = tab['sigma']
        mwin = (ts_all >= SIG_LO) & (ts_all <= SIG_HI)
        neff_t = posterior_neff(X, ts_all[mwin])          # on the table's grid, for shading
        gen_mask = neff_t >= NEFF_MIN
        sg_mem = float(ts_all[mwin][gen_mask][0]) if gen_mask.any() else float(SIG_HI)
        gen = torch.Generator(device=DEV); gen.manual_seed(0)
        Theta = torch.randn(N_THETA, d, device=DEV, dtype=DT, generator=gen) / np.sqrt(d)
        print(f"=== {name}: d={d} ===")
        print("  posterior effective support N_eff (empirical-measure Bayes, N=1e4 atoms):")
        for s, ne in zip(ts_all[mwin], neff_t):
            print(f"    sigma={s:>7.4f}  N_eff={ne:>10.2f}   "
                  f"{'MEMORISES' if ne < NEFF_MIN else 'genuine'}")
        print(f"  -> gap is genuine only for sigma >= {sg_mem:.3f}")
        hat, chk, gm = defect_curves(X, Theta, sigmas)
        print(f"{'sigma':>8} {'gamma':>11} {'hat_w^2':>12} {'chk_w^2':>12} "
              f"{'k(<=0.1)':>11} {'k(<=0.01)':>11} {'d/hat_w^2':>12}")
        for i, s in enumerate(sigmas):
            print(f"{s:>8.4f} {gm[i]:>11.3e} {hat[i]:>12.3e} {chk[i]:>12.3e} "
                  f"{0.1*d/(4*chk[i]):>11.3e} {0.01*d/(4*chk[i]):>11.3e} {d/hat[i]:>12.3e}")
        R[name] = dict(d=d, sg_mem=sg_mem, neff=neff_t, hat=hat, chk=chk, gam=gm,
                       tsig=tab['sigma'], lin=tab['linear_uncond'], bay=tab['bayes_uncond'])
        del X, Theta; torch.cuda.empty_cache()
        print()

    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), sharex='col')
    for j, name in enumerate(('MNIST', 'CIFAR-10')):
        r = R[name]; d = r['d']; sm = r['sg_mem']
        ts, lin, bay = r['tsig'], r['lin'], r['bay']
        m = (ts >= SIG_LO) & (ts <= SIG_HI)
        gen_ok = m & (ts >= sm)                       # gap that is NOT a memorisation artifact

        # ---- top: the gap an RF denoiser could close (LINEAR MSE axis) ----
        ax = axes[0][j]
        ax.semilogx(ts[m], lin[m], color='darkorange', lw=2.6,
                    label=r'linear (Wiener) $\mathcal{L}^{\rm lin}_\sigma$')
        ax.semilogx(ts[m], bay[m], color='crimson', lw=2.6,
                    label=r'oracle Bayes $\mathcal{L}^{\rm Bayes}_\sigma$')
        ax.fill_between(ts[gen_ok], bay[gen_ok], lin[gen_ok], color='seagreen', alpha=.25,
                        label='gap an RF denoiser could close')
        ax.fill_between(ts[m & (ts < sm)], bay[m & (ts < sm)], lin[m & (ts < sm)],
                        color='0.55', alpha=.30, hatch='//', ec='0.35', lw=0,
                        label='memorisation artifact ($N_{\\rm eff}<2$)')
        ax.set_xlim(SIG_LO, SIG_HI)
        ax.axvline(sm, color='0.30', ls='-', lw=1.6)
        ax.annotate(f'$N_{{\\rm eff}}\\!=\\!2$\nat $\\sigma$={sm:.2f}', xy=(sm, ax.get_ylim()[1] * 0.10),
                    xytext=(-8, 0), textcoords='offset points', ha='right',
                    fontsize=9.5, color='0.20')
        pk = ts[gen_ok][np.argmax((lin - bay)[gen_ok])]
        ax.axvline(pk, color='seagreen', ls=':', lw=2.0)
        ax.annotate(f'gap peaks  $\\sigma$={pk:.2f}', xy=(pk, ax.get_ylim()[1] * 0.62),
                    xytext=(7, 0), textcoords='offset points', fontsize=10.5,
                    color='seagreen', fontweight='bold')
        ax.set_ylabel('MSE'); ax.grid(True, alpha=.3)
        ax.set_title(f'{name}  ($d={d}$):  where a nonlinear denoiser has room')
        ax.legend(fontsize=9.5, loc='upper left')

        # ---- bottom: the threshold ----
        ax2 = axes[1][j]
        for tol, c, mk in zip(TOLS, ('#1f4e9c', '#5aa2e8'), ('o', 's')):
            ax2.loglog(sigmas, tol * d / (4 * r['chk']), color=c, marker=mk, lw=2.4, ms=5,
                       label=rf'$k(\mathrm{{gap}}\leq{tol})=\tau d/(4\check\varepsilon_w^2)$')
        ax2.loglog(sigmas, d / r['hat'], color='0.45', ls='--', lw=2.0,
                   label=r'$d/\hat\varepsilon_w^{\,2}$  (writeup form, $\Xi$ dropped)')
        ax2.axhline(float(d) ** N0, color='indianred', ls='-.', lw=1.8,
                    label=rf'$d^{{{N0}}}$ (tensor branch, $c_0{{=}}1$)')
        ax2.axvspan(SIG_LO, sm, color='0.55', alpha=.16, hatch='//', ec='0.45', lw=0)
        ax2.axvline(sm, color='0.30', ls='-', lw=1.4)
        ax2.axvline(pk, color='seagreen', ls=':', lw=2.0)
        ax2.set_xlim(SIG_LO, SIG_HI)
        ax2.set_xlabel(r'$\sigma$'); ax2.set_ylabel('number of RF features $k$')
        ax2.grid(True, alpha=.3, which='both')
        ax2.set_title(f'{name}: features below which linear is guaranteed to win')
        ax2.legend(fontsize=9.5, loc='upper left')

    fig.suptitle('How many ReLU random features before an RF denoiser may beat the linear one?\n'
                 r'Thresholds are shrinkage-free ($\mathrm{Cov}(r)\Delta=\sigma^2\Sigma_y^{-1}\Delta^{\rm raw}$). '
                 'Hatched: the plotted Bayes curve is nearest-neighbour memorisation of the '
                 r'$N{=}10^4$ sample, not population MMSE — most of the apparent gap is not real',
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    os.makedirs('figures', exist_ok=True)
    out = 'figures/rf_kstar_vs_sigma.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    np.savez('tables/rf_kstar_vs_sigma.npz', sigmas=sigmas,
             **{f'{k}_{q}': R[k][q] for k in R for q in ('hat', 'chk', 'gam')},
             **{f'{k}_d': R[k]['d'] for k in R},
             **{f'{k}_sgmem': R[k]['sg_mem'] for k in R},
             **{f'{k}_neff': R[k]['neff'] for k in R})


if __name__ == '__main__':
    main()
