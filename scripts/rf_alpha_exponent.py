"""
rf_alpha_exponent.py — measure the exponent alpha in  hat_eps_d^2 ~ d^{-alpha}  for GMM
families, and manufacture a spiked third cumulant to hit a prescribed alpha.

Why alpha is the quantity that matters
--------------------------------------
    k_* = min( d / hat_eps_d^2 ,  K_tensor ),   K_tensor = c_0 d^{n_0} (log d)^{-A}.
If hat_eps_d^2 ~ d^{-alpha} then the defect branch is d^{1+alpha}. So:
    alpha  <  n_0 - 1   ->  the DEFECT branch binds, k_* ~ d^{1+alpha}   (data-dependent)
    alpha  >  n_0 - 1   ->  the TENSOR branch binds, k_* ~ d^{n_0}       (data-independent)
Measuring alpha for a family therefore says which regime that family lives in, and n_0 is
ours to choose (see the note at the bottom of this docstring).

Families
--------
(1) "gmm"    : the published make_gmm (A rescaled by sqrt(32/d) so Tr(S2) ~ d; identical
               at d=32). Structure = 3 fixed class means in a <=2-dim subspace plus
               component heteroscedasticity.
(2) "spike"  : a rank-one third-cumulant spike, realized EXACTLY as a 2-component GMM.
               Take x0 = z + s v with v a unit vector and s a centered 2-point variable,
               s = a w.p. p, s = b w.p. 1-p. Then x0 is a 2-component Gaussian mixture
               with means a v and b v and shared covariance, and
                   kappa_3(x0) = kappa_3(s) * v^{otimes 3}   (rank one, as in row (iii)).
               Parametrize a = t sqrt((1-p)/p), b = -t sqrt(p/(1-p)): then E s = 0,
               Var s = t^2, and
                   lambda := kappa_3(s) = t^3 (1-2p)/sqrt(p(1-p)).
               The base Gaussian gets covariance I - t^2 v v^T so that Sigma = I exactly
               (||Sigma||_op = 1, which also removes the loose 2||Sigma||^2 constant).
               lambda is dialed through p alone, at fixed variance.

               The note predicts (row (iii)) hat_eps_d^2 ~ lambda^2 / d^2. Hence
                   lambda fixed      -> alpha = 2  -> k_* ~ d^3
                   lambda ~ d^{-1/2} -> alpha = 3  -> k_* ~ d^4
               The second is the "manufacture alpha = 3" request: set 1-2p ~ d^{-1/2}.

On choosing n_0
---------------
n_0 is a free parameter of the bound, not a property of the data: we pick it, subject to
the activation admitting a Mehler band starting there. For ReLU with a compactly supported
offset law, Prop. A2 gives per-R positivity with n_0 ARBITRARY, so K_tensor = c_0 d^{n_0}
can be pushed above the defect branch for any family with finite alpha. The merit is
precisely that: with n_0 = 2 one is stuck reporting k_* ~ d^2 for every distribution, which
says nothing about the data. Choosing n_0 > alpha + 1 hands the binding role to
d/hat_eps_d^2, which is the informative, distribution-dependent branch. So the answer is
yes-with-a-purpose: higher n_0 buys nothing on its own, it buys the right to read k_* off
the data.

    FAMILY=gmm   DIMS=32,64,128,256 python scripts/rf_alpha_exponent.py
    FAMILY=spike SPIKE_MODE=fixed    python scripts/rf_alpha_exponent.py
    FAMILY=spike SPIKE_MODE=decay    python scripts/rf_alpha_exponent.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import norm

from core.gmm import GaussianMixture

SIGMA   = float(os.environ.get('SIGMA', '1.0'))
DIMS    = [int(x) for x in os.environ.get('DIMS', '32,64,128,256,512').split(',')]
M_THETA = int(os.environ.get('M_THETA', '200000'))
FAMILY  = os.environ.get('FAMILY', 'gmm')
SPIKE_P = float(os.environ.get('SPIKE_P', '0.25'))
SPIKE_T = float(os.environ.get('SPIKE_T', '0.7'))
SPIKE_MODE = os.environ.get('SPIKE_MODE', 'fixed')     # 'fixed' | 'decay'
SEED    = 42


def make_gmm(d, seed=SEED):
    rng = np.random.default_rng(seed)
    means = np.zeros((3, d))
    means[0, 0] = 2.0
    means[1, 0] = -1.0; means[1, 1] = 1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2
    s0 = np.full(d, 0.4); s0[0] = 1.2
    s1 = np.full(d, 0.4); s1[0] = 0.4; s1[1] = 1.0; s1[2] = 0.8
    A = rng.standard_normal((d, d)) * (0.3 * np.sqrt(32.0 / d))
    S2 = A @ A.T + 0.5 * np.eye(d)
    return GaussianMixture(weights=np.array([0.5, 0.3, 0.2]), means=means,
                           covs=np.stack([np.diag(s0), np.diag(s1), S2])), None


def make_spike(d, seed=SEED):
    """Rank-one kappa_3 spike as an exact 2-component GMM; Sigma = I."""
    p = SPIKE_P
    if SPIKE_MODE == 'decay':                 # 1-2p ~ d^{-1/2}  =>  lambda ~ d^{-1/2}
        p = 0.5 * (1.0 - (1.0 - 2.0 * SPIKE_P) * np.sqrt(32.0 / d))
    t = SPIKE_T
    a = t * np.sqrt((1 - p) / p)
    b = -t * np.sqrt(p / (1 - p))
    lam = t ** 3 * (1 - 2 * p) / np.sqrt(p * (1 - p))     # kappa_3(s)
    v = np.zeros(d); v[0] = 1.0
    base = np.eye(d) - (t ** 2) * np.outer(v, v)          # so total Sigma = I
    means = np.stack([a * v, b * v])
    return GaussianMixture(weights=np.array([p, 1 - p]), means=means,
                           covs=np.stack([base, base])), lam


def hat_eps2(gmm, sigma, M, seed=0):
    """Exact-in-x0 Stein defect for a GMM: hat_eps^2 = sqrt(E||Delta||^4)."""
    d = gmm.d
    Sinv = np.linalg.inv(gmm.Sigma)
    rng = np.random.default_rng(seed)
    Th = rng.standard_normal((M, d)) / np.sqrt(d)
    s = sigma * np.linalg.norm(Th, axis=1)
    m = Th @ gmm.means.T
    v = np.einsum('kd,cde,ke->kc', Th, gmm.covs, Th)
    S = np.sqrt(v + s[:, None] ** 2)
    z = m / S; Phi = norm.cdf(z); c0 = m * Phi + S * norm.pdf(z)
    dmu = gmm.means - gmm.mu
    SigTh = np.einsum('cde,ke->kcd', gmm.covs, Th)
    Cov = (np.einsum('c,kc,cd->kd', gmm.weights, c0, dmu)
           + np.einsum('c,kc,kcd->kd', gmm.weights, Phi, SigTh))
    abar = (gmm.weights[None, :] * Phi).sum(1)
    Dl = Cov @ Sinv.T - abar[:, None] * Th
    n2 = (Dl ** 2).sum(1)
    return float(np.sqrt((n2 ** 2).mean())), float(n2.mean())


def main():
    builder = {'gmm': make_gmm, 'spike': make_spike}[FAMILY]
    tag = FAMILY + (f'-{SPIKE_MODE}' if FAMILY == 'spike' else '')
    print(f"family={tag}  sigma={SIGMA}  M_theta={M_THETA}")
    if FAMILY == 'spike':
        print(f"  spike: p0={SPIKE_P}, t={SPIKE_T}, mode={SPIKE_MODE}"
              f"  (kappa_3 = lambda v^3, Sigma = I)")
        print("  predicted by row (iii): hat_eps^2 ~ lambda^2/d^2, so alpha = 2 (fixed lambda)"
              " or 3 (lambda ~ d^-1/2)")
    print(f"\n{'d':>6} {'lambda':>9} {'hat_eps^2':>12} {'E||D||^2':>12} {'d/hat_eps^2':>13} {'d^2':>10}")
    es, lams = [], []
    for d in DIMS:
        g, lam = builder(d)
        e2, m2 = hat_eps2(g, SIGMA, M_THETA)
        es.append(e2); lams.append(lam if lam is not None else np.nan)
        ls = f"{lam:9.4f}" if lam is not None else f"{'--':>9}"
        print(f"{d:>6} {ls} {e2:>12.3e} {m2:>12.3e} {d/e2:>13.3e} {float(d)**2:>10.0f}")
    alpha = -np.polyfit(np.log(DIMS), np.log(es), 1)[0]
    beta = np.polyfit(np.log(DIMS), np.log([d / e for d, e in zip(DIMS, es)]), 1)[0]
    print(f"\n  fitted alpha  (hat_eps^2 ~ d^-alpha)      = {alpha:+.3f}")
    print(f"  fitted slope of d/hat_eps^2 (= 1 + alpha)  = {beta:+.3f}")
    if FAMILY == 'spike':
        pred = 2.0 if SPIKE_MODE == 'fixed' else 3.0
        print(f"  predicted alpha for this mode              = {pred:.1f}"
              f"   -> {'MATCH' if abs(alpha - pred) < 0.35 else 'MISMATCH'}")
    for n0 in (2, 3, 4):
        which = ['defect' if d / e < float(d) ** n0 else 'tensor'
                 for d, e in zip(DIMS, es)]
        print(f"  with n_0={n0}: binding branch per d = {which}")
    os.makedirs('tables', exist_ok=True)
    np.savez(f'tables/rf_alpha_{tag}.npz', dims=np.array(DIMS), eps2=np.array(es),
             lam=np.array(lams), alpha=alpha)


if __name__ == '__main__':
    main()
