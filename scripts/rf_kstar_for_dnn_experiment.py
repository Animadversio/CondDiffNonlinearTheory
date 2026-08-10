"""
rf_kstar_for_dnn_experiment.py — how many RF-ReLU features would be needed before a
random-feature denoiser can beat the LINEAR denoiser, on exactly the data used in
figures/dnn_feature_mmse_{cifar10,mnist}.png.

Question this answers
---------------------
In that figure there is an open gap between the linear denoiser and the oracle Bayes
denoiser at low-to-intermediate sigma. A random-feature ReLU denoiser is a candidate for
closing part of it. Theorem (thm:prob) says it CANNOT, at any width below

    k_* = min( d / hat_eps_d^2 ,  K_tensor ),      K_tensor = c_0 d^{n_0} (log d)^{-A}.

So k_* is the number of features you must exceed before the attempt is even permitted.

Matching the experiment exactly
-------------------------------
hat_eps_d is NOT scale- or basis-free (under x -> A x the defect maps Delta -> A^{-T}Delta),
and sigma is measured in the same units as x0. So k_* is only comparable to that figure's
sigma axis if x0 uses the SAME convention. scripts/dnn_feature_mmse.py uses
    raw_tf = T.ToTensor()      -> raw pixels in [0,1], NOT centered, NOT standardized,
d = 3072 (CIFAR RGB) / 784 (MNIST). This script uses exactly that.

k_* is sigma-dependent
----------------------
The defect is evaluated at s = sigma ||theta||, so hat_eps_d = hat_eps_d(sigma) and hence
k_* = k_*(sigma). Reporting a single k_* (as an earlier pass did) is wrong for a figure
whose whole point is the sigma sweep. We report the curve.

The two branches, honestly
--------------------------
  * DEFECT branch  d/hat_eps_d^2 : fully computable here.
  * TENSOR branch  K_tensor = c_0 d^{n_0}(log d)^{-A} : c_0 and A are the (unquantified)
    constants imported from the kernel-matrix-concentration results [MMM21, MZ22]; the
    note does not pin them, so K_tensor cannot be evaluated numerically. We report the
    idealized c_0 = 1, A = 0 value d^{n_0} for reference and flag the constants as unknown.
    For ReLU with a compactly supported offset law (ours: epsilon = 0, a point mass),
    Prop. A2 lets n_0 be taken arbitrarily large, so K_tensor is not the binding branch and
        k_*  =  d / hat_eps_d^2
    is the operative threshold. That is the number quoted.

Estimator: validated split-half (see scripts/rf_kstar_image_data.py) -- per-half Sigma,
per-half centering of both x and psi. Sharing Sigma across halves, or using E[x psi] in
place of Cov(x,psi), silently produces negative (impossible) values.

    python scripts/rf_kstar_for_dnn_experiment.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
N_THETA = int(os.environ.get('N_THETA', '4096'))
NIMG    = int(os.environ.get('NIMG', '50000'))
SHRINKS = [float(x) for x in os.environ.get('SHRINKS', '1e-4,1e-3,1e-2').split(',')]
N0      = int(os.environ.get('N0', '2'))

CIFAR_ROOT = '/n/home12/binxuwang/.keras/datasets'
MNIST_ROOT = os.path.expanduser('~/.keras/datasets')
TBL = {'CIFAR-10': 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz',
       'MNIST':    'tables/dnn_feature_mmse_mnist_N10000_noise5_sigma30.npz'}


def load_raw(name):
    """EXACTLY the x0 convention of dnn_feature_mmse.py: ToTensor() -> [0,1], no centering."""
    import torchvision, torchvision.transforms as T
    tf = T.ToTensor()
    if name == 'CIFAR-10':
        ds = torchvision.datasets.CIFAR10(CIFAR_ROOT, train=True, download=False, transform=tf)
    else:
        ds = torchvision.datasets.MNIST(MNIST_ROOT, train=True, download=False, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, num_workers=2)
    out = []
    for xb, _ in dl:
        out.append(xb.reshape(xb.shape[0], -1))
        if sum(o.shape[0] for o in out) >= NIMG:
            break
    X = torch.cat(out)[:NIMG].to(DEV, DT)
    g = torch.Generator(device=DEV); g.manual_seed(5)
    return X[torch.randperm(X.shape[0], device=DEV, generator=g)]


def hat_eps2(X, sigma, Theta, shrink):
    """Validated split-half estimate of E||Delta||^2 and hat_eps^2 = sqrt(E||Delta||^4)."""
    n, d = X.shape
    s = sigma * torch.linalg.norm(Theta, dim=1)
    half = n // 2
    Ds = []
    for Xh in (X[:half], X[half:2 * half]):
        mu = Xh.mean(0); Xc = Xh - mu
        Sg = (Xc.T @ Xc) / Xh.shape[0]
        Sg = Sg + shrink * torch.trace(Sg) / d * torch.eye(d, device=DEV, dtype=DT)
        Si = torch.linalg.inv(Sg)
        U = Xh @ Theta.T
        z = U / s[None, :]
        Phi = torch.special.ndtr(z)
        ph = torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        cc = U * Phi + s[None, :] * ph
        Cv = (Xc.T @ (cc - cc.mean(0))) / Xh.shape[0]
        Ds.append(Si @ Cv - Phi.mean(0)[None, :] * Theta.T)
    n2 = (Ds[0] * Ds[1]).sum(0)
    m2 = float(n2.mean())
    hat2 = float(np.sqrt(max(float((n2.clamp(min=0) ** 2).mean()), 0.0)))
    return m2, hat2, float(n2.std() / np.sqrt(len(n2)))


def main():
    print(f"x0 convention: raw pixels in [0,1] (ToTensor), uncentered -- matches "
          f"dnn_feature_mmse.py exactly")
    print(f"n_theta={N_THETA}  n_img={NIMG}  shrink sweep={SHRINKS}\n")
    results = {}
    for name in ('MNIST', 'CIFAR-10'):
        if not os.path.exists(TBL[name]):
            print(f"[skip] {TBL[name]} missing"); continue
        tab = np.load(TBL[name], allow_pickle=True)
        sig_grid = tab['sigma']
        lin, bay = tab['linear_uncond'], tab['bayes_uncond']
        X = load_raw(name); d = X.shape[1]
        gen = torch.Generator(device=DEV); gen.manual_seed(0)
        Theta = torch.randn(N_THETA, d, device=DEV, dtype=DT, generator=gen) / np.sqrt(d)
        print(f"=== {name}:  d={d},  Tr(Sigma_p0)={float(((X-X.mean(0))**2).sum(1).mean()):.3f} "
              f"(raw [0,1] units) ===")
        print(f"{'sigma':>8} {'L_lin':>9} {'L_bayes':>9} {'gap':>9} "
              + " ".join(f"{'k_*(sh='+f'{s:.0e}'+')':>16}" for s in SHRINKS))
        rows = []
        # restrict to the low/intermediate-noise window where the gap is open
        sel = [i for i, sg in enumerate(sig_grid) if 0.02 <= sg <= 10.0]
        for i in sel:
            sg = float(sig_grid[i])
            ks = []
            for sh in SHRINKS:
                m2, h2, se = hat_eps2(X, sg, Theta, sh)
                ks.append(d / h2 if h2 > 0 else np.inf)
            rows.append((sg, float(lin[i]), float(bay[i]), float(lin[i] - bay[i]), ks))
            print(f"{sg:>8.4f} {lin[i]:>9.3f} {bay[i]:>9.3f} {lin[i]-bay[i]:>9.3f} "
                  + " ".join(f"{k:>16.3e}" for k in ks))
        results[name] = dict(d=d, rows=rows, sig=sig_grid, lin=lin, bay=bay)
        kk = np.array([r[4] for r in rows])
        print(f"  -> over this window k_* = d/hat_eps^2 spans "
              f"[{np.nanmin(kk):.2e}, {np.nanmax(kk):.2e}]   (d^{N0} = {float(d)**N0:.2e} "
              f"for reference; K_tensor's c_0, A are unquantified in the note)")
        print()

    # figure
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6.4 * n, 8.4), squeeze=False)
    fig.suptitle('How many RF-ReLU features before a random-feature denoiser may beat the '
                 'linear one?\n'
                 r'$k_*(\sigma)=d/\hat\varepsilon_d^{\,2}(\sigma)$ on the same raw-pixel data as '
                 'figures/dnn_feature_mmse_*.png', fontsize=13)
    for j, (name, R) in enumerate(results.items()):
        sg = np.array([r[0] for r in R['rows']])
        ax = axes[0][j]
        ax.loglog(R['sig'], R['lin'], color='darkorange', lw=2, label='linear denoiser')
        ax.loglog(R['sig'], np.maximum(R['bay'], 1e-8), color='crimson', lw=2,
                  label='oracle Bayes')
        ax.fill_between(R['sig'], np.maximum(R['bay'], 1e-8), R['lin'], color='green',
                        alpha=.12, label='gap an RF denoiser could target')
        ax.axvspan(sg.min(), sg.max(), color='navy', alpha=.05)
        ax.set_xlabel('σ'); ax.set_ylabel('MSE'); ax.grid(True, alpha=.3, which='both')
        ax.set_title(f"{name} (d={R['d']}): the gap"); ax.legend(fontsize=8)
        ax2 = axes[1][j]
        for m, sh in enumerate(SHRINKS):
            ax2.loglog(sg, [r[4][m] for r in R['rows']], marker='o', lw=2,
                       label=f'shrink {sh:.0e}')
        ax2.axhline(float(R['d']) ** N0, color='gray', ls='--', lw=1.4,
                    label=f'$d^{{{N0}}}$ (K_tensor with $c_0$=1, A=0)')
        ax2.set_xlabel('σ'); ax2.set_ylabel(r'$k_*=d/\hat\varepsilon_d^{\,2}(\sigma)$')
        ax2.set_title(f'{name}: features needed before RF may beat linear')
        ax2.grid(True, alpha=.3, which='both'); ax2.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs('figures', exist_ok=True)
    out = 'figures/rf_kstar_for_dnn_experiment.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
