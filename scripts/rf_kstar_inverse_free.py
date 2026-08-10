"""
rf_kstar_inverse_free.py — k_* for CIFAR-10 / MNIST with NO shrinkage, NO Ledoit-Wolf.

Uses the Cov(r)-weighted form (Remark rem:covr of the writeup). Because Sigma and
Sigma_y = Sigma + sigma^2 I commute,

    Cov(r) Delta  =  sigma^2 Sigma_y^{-1} Delta^raw,
    Delta^raw     =  Cov(x0, psi_s(theta^T x0 + eps))  -  alphabar(theta,eps) Sigma theta,

so no inverse of Sigma appears anywhere. The only inversion is Sigma_y^{-1}, which exists
for every sigma > 0 (Sigma_y >= sigma^2 I) and is well conditioned: cond(Sigma_y) is
56-888 on CIFAR pixels for sigma in [0.25, 1] where cond(Sigma) = 3.6e7.

Reported, per sigma:
  hat_eps_w^2   := sqrt( E ||Cov(r) Delta||^4 )       (the Cov(r)-weighted analogue)
  check_eps_w^2 := E[ ||Cov(r) Delta||^2 / gamma ; ||theta|| in [1/2,2] ]
  k_*           := d / hat_eps_w^2                    (writeup convention: constants dropped)
  k_tau         := the largest k for which the bound GUARANTEES a normalized gap <= tau,
                   i.e. (1/rho_*)(k/d) check_eps_w^2 / delta <= tau with rho_* = delta = 1/2.

The old operator-norm form is printed alongside for comparison. NOTE the two are only
comparable once ||Sigma||_op^2 is restored on the old side, since the writeup's k_* drops
that constant -- it is dimensionless only under the assumption ||Sigma||_op = O(1), which
is exactly what fails for pixel data (||Sigma||_op = 55.4).

    python scripts/rf_kstar_inverse_free.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
N_THETA = int(os.environ.get('N_THETA', '4096'))
NIMG    = int(os.environ.get('NIMG', '50000'))
N0, NB  = 2, 6
SIGMAS  = [float(x) for x in os.environ.get('SIGMAS', '0.0672,0.1743,0.4520,1.1721,3.0392').split(',')]
TOLS    = [float(x) for x in os.environ.get('TOLS', '0.1,0.01').split(',')]
CIFAR = '/n/home12/binxuwang/.keras/datasets'
MNI   = os.path.expanduser('~/.keras/datasets')


def load(name):
    import torchvision, torchvision.transforms as T
    tf = T.ToTensor()                      # raw [0,1], exactly dnn_feature_mmse.py
    ds = (torchvision.datasets.MNIST(MNI, train=True, download=False, transform=tf)
          if name == 'MNIST' else
          torchvision.datasets.CIFAR10(CIFAR, train=True, download=False, transform=tf))
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
    """ReLU Hermite coefficient c_n(M,s)."""
    from math import factorial
    z = M / s
    if n == 1:
        return s * torch.special.ndtr(z)
    he = [torch.ones_like(z), z]
    for j in range(2, n - 1):
        he.append(z * he[-1] - (j - 1) * he[-2])
    H = he[n - 2]
    return ((-1) ** n) * s * H * torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / factorial(n)


def stats(X, sigma, Theta):
    """Split-half; returns weighted/unweighted defect moments and gamma. NO shrinkage."""
    from math import factorial
    n, d = X.shape
    I = torch.eye(d, device=DEV, dtype=DT)
    s = sigma * torch.linalg.norm(Theta, dim=1)
    half = n // 2
    Dw, Du, gam = [], [], None
    for Xh in (X[:half], X[half:2 * half]):
        mu = Xh.mean(0); Xc = Xh - mu
        Sig = (Xc.T @ Xc) / Xh.shape[0]
        Syi = torch.linalg.inv(Sig + sigma ** 2 * I)          # ONLY inversion; cond ~ 1e2
        U = Xh @ Theta.T
        z = U / s[None, :]
        Phi = torch.special.ndtr(z)
        ph = torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        cc = U * Phi + s[None, :] * ph
        Cv = (Xc.T @ (cc - cc.mean(0))) / Xh.shape[0]         # Cov(x0, psi)   (d,M)
        ab = Phi.mean(0)                                       # alphabar        (M,)
        Draw = Cv - (Sig @ Theta.T) * ab[None, :]              # Delta^raw = Sigma Delta
        Dw.append(sigma ** 2 * (Syi @ Draw))                   # Cov(r) Delta
        Du.append(Draw)                                        # for reference
        if gam is None:                                        # gamma from the same half
            acc = torch.zeros_like(U)
            for nn in range(N0, NB + 1):
                acc += factorial(nn) * _cn(U, s[None, :], nn) ** 2
            gam = acc.mean(0)                                  # (M,)
        del Sig, Syi
    nw = (Dw[0] * Dw[1]).sum(0)                                # unbiased ||Cov(r)Delta||^2
    nu = (Du[0] * Du[1]).sum(0)                                # unbiased ||Delta^raw||^2
    ok = (torch.linalg.norm(Theta, dim=1) >= 0.5) & (torch.linalg.norm(Theta, dim=1) <= 2.0)
    hat_w2 = float(np.sqrt(max(float((nw.clamp(min=0) ** 2).mean()), 0.0)))
    chk_w2 = float((nw[ok] / gam[ok].clamp(min=1e-300)).mean())
    return hat_w2, chk_w2, float(nw.mean()), float(nu.mean())


def main():
    print("k_* WITHOUT shrinkage or Ledoit-Wolf, via the Cov(r)-weighted (inverse-free) form")
    print("  Cov(r)Delta = sigma^2 Sigma_y^{-1}[Cov(x0,psi) - alphabar Sigma theta];"
          "  only Sigma_y is inverted\n")
    for name in ('MNIST', 'CIFAR-10'):
        X = load(name); d = X.shape[1]
        Xc = X - X.mean(0); Sig = (Xc.T @ Xc) / X.shape[0]
        ev = torch.linalg.eigvalsh(Sig)
        print(f"=== {name}: d={d}, cond(Sigma)={float(ev.max()/ev.min()):.2e}, "
              f"||Sigma||_op={float(ev.max()):.3f} ===")
        print(f"{'sigma':>8} {'cond(Sy)':>9} {'hat_eps_w^2':>13} {'chk_eps_w^2':>13} "
              f"{'k_*=d/hat_w^2':>15} " + " ".join(f"{'k(gap<='+str(t)+')':>15}" for t in TOLS))
        gen = torch.Generator(device=DEV); gen.manual_seed(0)
        Theta = torch.randn(N_THETA, d, device=DEV, dtype=DT, generator=gen) / np.sqrt(d)
        for sg in SIGMAS:
            I = torch.eye(d, device=DEV, dtype=DT)
            evy = torch.linalg.eigvalsh(Sig + sg ** 2 * I)
            cSy = float(evy.max() / evy.min())
            hw, cw, mw, mu_ = stats(X, sg, Theta)
            ks = d / hw if hw > 0 else float('inf')
            # bound: (1/rho_*)(k/d) chk / delta <= tau, rho_*=delta=1/2  ->  k <= tau d /(4 chk)
            kt = [t * d / (4 * cw) if cw > 0 else float('inf') for t in TOLS]
            print(f"{sg:>8.4f} {cSy:>9.1f} {hw:>13.4e} {cw:>13.4e} {ks:>15.3e} "
                  + " ".join(f"{v:>15.3e}" for v in kt))
        del X, Sig; torch.cuda.empty_cache()
        print()


if __name__ == '__main__':
    main()
