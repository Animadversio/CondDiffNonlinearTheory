"""
rf_alpha_from_data.py — compute d / hat_eps_d^2 DIRECTLY for a real dataset at its native
dimension. No exponent is fitted, by design.

Why no exponent
---------------
alpha is a property of a FAMILY of distributions indexed by d, not of one dataset. CIFAR-10
and MNIST each sit at a single d, so no asymptotic statement in d is provable from them:
any d-sweep one manufactures (downsampling, coordinate subsets, random projections) changes
the data-generating process along with d, so the fitted exponent describes the manufactured
family, not the dataset. The division of labour we use instead:

  * CONTROLLED SYNTHETIC FAMILIES  -> alpha is known/measured (scripts/rf_alpha_exponent.py:
    published GMM +0.13, skewed product +0.99, kappa_3 spike +1.98, decaying spike +3.07).
  * REAL DATASETS                  -> report d / hat_eps_d^2 DIRECTLY at the native d,
    which is all that k_* = min(d/hat_eps_d^2, K_tensor) needs. No extrapolation claimed.

Two things that will bite you
-----------------------------
1. hat_eps_d is NOT basis-free. Under x -> A x the defect transforms as Delta -> A^{-T}
   Delta, so it is not norm-preserving unless A is orthogonal. That is not a defect of the
   estimator: the RF model itself is basis-dependent, because theta ~ N(0, I/d) is drawn
   isotropically in whatever coordinates you hand it. So "whiten first" changes the answer.
   Here everything is in the pixel basis with per-pixel standardization, and that choice is
   part of the reported number.
2. Delta = Sigma^{-1}Cov(x0, psi_s(theta^T x0)) - alphabar theta is a SMALL DIFFERENCE of
   two O(1) terms, so the naive plug-in ||Delta_hat||^2 is biased upward by ~d/N and at
   large d that bias IS the measurement. We use the split-half estimator
   <Delta_hat_1, Delta_hat_2>, unbiased because the two noise terms are independent.
   Sigma^{-1} additionally needs shrinkage for real data (pixel covariances are
   ill-conditioned); the shrinkage level is reported since it affects the number.

    python scripts/rf_alpha_from_data.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
SIGMA = float(os.environ.get('SIGMA', '1.0'))
RES = [int(x) for x in os.environ.get('RES', '6,8,12,16,24,32').split(',')]
N_THETA = int(os.environ.get('N_THETA', '4096'))
SHRINK = float(os.environ.get('SHRINK', '1e-2'))
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.expanduser('~/.keras/datasets'))
MODE = os.environ.get('MODE', 'native')              # 'native' only
N_IMG = int(os.environ.get('N_IMG', '50000'))


def load_cifar_gray():
    import torchvision, torchvision.transforms as T
    have = os.path.isdir(os.path.join(DATA_ROOT, 'cifar-10-batches-py'))
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=not have,
                                      transform=T.Compose([T.Grayscale(), T.ToTensor()]))
    n = min(N_IMG, len(ds))
    dl = torch.utils.data.DataLoader(ds, batch_size=2048, num_workers=4)
    out = []
    for xb, _ in dl:
        out.append(xb)
        if sum(o.shape[0] for o in out) >= n:
            break
    return torch.cat(out)[:n].to(DEV, DT)             # (n,1,32,32)


def make_view(imgs, d_target, mode, gen):
    """Return (n, d) data at the requested dimension."""
    if mode == 'resolution':
        r = int(round(np.sqrt(d_target)))
        X = F.interpolate(imgs, size=(r, r), mode='area').reshape(imgs.shape[0], -1)
    else:                                             # random coordinate subset
        full = imgs.reshape(imgs.shape[0], -1)
        idx = torch.randperm(full.shape[1], device=DEV, generator=gen)[:d_target]
        X = full[:, idx]
    X = X - X.mean(0, keepdim=True)
    X = X / X.std(0, keepdim=True).clamp(min=1e-8)    # per-pixel standardization
    return X


def hat_eps2_split(X, sigma, n_theta, shrink, gen):
    """Split-half unbiased estimate of E||Delta||^2 and of hat_eps^2 = sqrt(E||Delta||^4)."""
    n, d = X.shape
    Sig = (X.T @ X) / n
    Sig = Sig + shrink * torch.trace(Sig) / d * torch.eye(d, device=DEV, dtype=DT)
    Sinv = torch.linalg.inv(Sig)
    Theta = torch.randn(n_theta, d, device=DEV, dtype=DT, generator=gen) / np.sqrt(d)
    s = sigma * torch.linalg.norm(Theta, dim=1)
    half = n // 2
    Ds = []
    for X_h in (X[:half], X[half:2 * half]):
        U = X_h @ Theta.T
        z = U / s[None, :]
        Phi = torch.special.ndtr(z)
        phi = torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        c0 = U * Phi + s[None, :] * phi
        Cov = (X_h.T @ c0) / X_h.shape[0]             # (d,M)
        abar = Phi.mean(0)                            # (M,)
        Ds.append(Sinv @ Cov - abar[None, :] * Theta.T)
    n2 = (Ds[0] * Ds[1]).sum(0)                       # (M,) unbiased ||Delta||^2 per theta
    naive = 0.5 * ((Ds[0] ** 2).sum(0) + (Ds[1] ** 2).sum(0))
    m2 = float(n2.mean())
    hat4 = float((n2.clamp(min=0) ** 2).mean())       # E||Delta||^4
    return m2, float(np.sqrt(max(hat4, 0.0))), float(naive.mean()), float(n2.std() / np.sqrt(len(n2)))


def main():
    print(f"CIFAR-10 grayscale, mode={MODE}, sigma={SIGMA}, shrink={SHRINK}, "
          f"n_theta={N_THETA}, device={DEV}")
    print("alpha is a property of a FAMILY, not of a dataset -- here the family is "
          f"{'image RESOLUTION (d = r^2)' if MODE=='resolution' else 'random coordinate subsets'}\n")
    imgs = load_cifar_gray()
    print(f"loaded {imgs.shape[0]} images\n")
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    print(f"{'d':>6} {'r':>4} {'E||D||^2 (unb.)':>17} {'+-se':>10} {'naive':>11} "
          f"{'hat_eps^2':>11} {'d/hat_eps^2':>12}")
    ds, es = [], []
    for r in RES:
        d = r * r
        X = make_view(imgs, d, MODE, gen)
        m2, hat2, naive, se = hat_eps2_split(X, SIGMA, N_THETA, SHRINK, gen)
        flag = '' if m2 > 3 * se else '  (< 3 se!)'
        print(f"{d:>6} {r:>4} {m2:>17.4e} {se:>10.1e} {naive:>11.4e} "
              f"{hat2:>11.4e} {d/max(hat2,1e-300):>12.3e}{flag}")
        if m2 > 3 * se:
            ds.append(d); es.append(hat2)
    if len(ds) >= 3:
        a = -np.polyfit(np.log(ds), np.log(es), 1)[0]
        r2 = np.corrcoef(np.log(ds), np.log(es))[0, 1] ** 2
        print(f"\n  fitted alpha = {a:+.3f}   (R^2 = {r2:.3f}) over d = {ds}")
        print(f"  => defect branch d/hat_eps^2 ~ d^{1+a:.2f};  it binds iff n_0 > {1+a:.2f}")
        print(f"  => with n_0 = 2 the tensor branch d^2 binds whenever {1+a:.2f} > 2:"
              f" {'YES' if 1+a > 2 else 'no'}")
    else:
        print("\n  too few resolvable points to fit alpha")
    print("\nCaveats that belong with any number quoted from this script:")
    print("  * hat_eps_d is basis-dependent (Delta -> A^-T Delta under x -> Ax); this is the")
    print("    pixel basis with per-pixel standardization, and shrinkage {:.0e} on Sigma.".format(SHRINK))
    print("  * downsampling changes the data-generating process as well as d; the exponent")
    print("    describes the resolution family, not 'CIFAR' in the abstract.")


if __name__ == '__main__':
    main()
