"""
core/equivariant_floor.py

Exact floors for shift-equivariant denoiser classes on a GMM.

Two model classes, two different floors:

  (A) STRICTLY equivariant,  D(Sy) = S D(y):
        floor = MMSE(pbar0),  pbar0 = cyclic-shift symmetrisation of p0.
      (Standard: for equivariant D the p0 loss equals the pbar0 loss, so the best
       equivariant D is the pbar0-Bayes denoiser.)

  (B) Equivariant PLUS A FREE BIAS,  D(y) = E(y) + b with E equivariant, b in R^d:
        floor = MMSE(pbar0) - g^T H^{-1} g   <=  MMSE(pbar0)      [this module]
      A free per-position bias breaks equivariance, so (A)'s floor no longer applies —
      the class is strictly larger and its floor is strictly lower. This is the floor to
      use whenever the readout bias is unconstrained (the usual affine-ridge setup).

Derivation of (B). For equivariant E, estimating x0 from y = x0 + sigma z with a bias b
is the same as equivariantly estimating x0 - b. Pushing the shift onto the data (the
standard symmetrisation argument, using E(S^t u) = S^t E(u)) turns this into: estimate
    X = xbar0 - S^T b        from     Y = ybar = xbar0 + sigma Z,
where xbar0 = S^T x0 and T ~ Unif{0..d-1}. Hence for a given b the best achievable is the
MMSE of X given Y, and

    loss(b) = E|| (xbar0 - E[xbar0|ybar]) - (S^T b - E[S^T b|ybar]) ||^2
            = E||r - M b||^2 ,
      r = xbar0 - E[xbar0|ybar]              (pbar0-Bayes residual)
      M = S^T - sum_t P(T=t|ybar) S^t        (d x d, linear in b)

which is QUADRATIC in b, so it minimises in closed form:

    floor_free = E||r||^2 - g^T H^{-1} g = MMSE(pbar0) - g^T H^{-1} g,
      g = E[M^T r]  (d,),   H = E[M^T M]  (d,d).

b = 0 recovers MMSE(pbar0), so the correction g^T H^{-1} g >= 0 is exactly what the free
bias buys. All expectations are Monte-Carlo over the GMM, with the (c, tau) posterior of
pbar0 available in closed form.

Sanity identity (checked in _selftest): as sigma -> inf both MMSEs tend to the traces and
    Tr(Sigma_pbar0) - Tr(Sigma_p0) = || (I - P_1) mu_p0 ||^2,   P_1 = (1/d) 11^T,
i.e. the non-stationary energy of the mean — the largest amount a constant bias can ever
recover.
"""

import numpy as np
from scipy.stats import multivariate_normal

from .gmm import GaussianMixture


def shift_symmetrized_gmm(gmm):
    """pbar0 as an explicit C*d-component GMM, ordered so index c*d + tau is
    (P_tau mu_c, P_tau Sigma_c P_tau^T, w_c/d) with (P_tau x)_i = x_{i-tau mod d}."""
    d = gmm.d
    Ps = []
    for tau in range(d):
        P = np.zeros((d, d))
        for i in range(d):
            P[i, (i - tau) % d] = 1.0
        Ps.append(P)
    means, covs, weights = [], [], []
    for c in range(gmm.C):
        for tau in range(d):
            P = Ps[tau]
            means.append(P @ gmm.means[c])
            covs.append(P @ gmm.covs[c] @ P.T)
            weights.append(gmm.weights[c] / d)
    return GaussianMixture(weights=np.array(weights), means=np.stack(means),
                           covs=np.stack(covs))


def _perm_matrices(d):
    Ps = np.zeros((d, d, d))
    for tau in range(d):
        for i in range(d):
            Ps[tau, i, (i - tau) % d] = 1.0
    return Ps                      # (d, d, d): Ps[tau] = S^tau


def mmse_equivariant_floors(gmm, sigma, N_mc=100_000, rng=None, batch=20_000):
    """
    Returns dict with
      'mmse_pbar0'  : floor for STRICTLY equivariant denoisers            (class A)
      'floor_free'  : floor for equivariant + FREE BIAS                   (class B)
      'bias_gain'   : mmse_pbar0 - floor_free = g^T H^{-1} g   (>= 0)
      'mmse_p0'     : unconstrained Bayes MMSE on p0 (reference)
    All by Monte Carlo with the closed-form (c,tau) posterior of pbar0.
    """
    if rng is None:
        rng = np.random.default_rng()
    d, C = gmm.d, gmm.C
    gbar = shift_symmetrized_gmm(gmm)
    Ps = _perm_matrices(d)                                   # (d,d,d)
    I = np.eye(d)
    # per-(c,tau) Wiener gains for pbar0
    gains = [gbar.covs[j] @ np.linalg.solve(gbar.covs[j] + sigma ** 2 * I, I)
             for j in range(gbar.C)]
    chol_ok = [gbar.covs[j] + sigma ** 2 * I for j in range(gbar.C)]

    sum_r2 = 0.0
    H = np.zeros((d, d))
    g = np.zeros(d)
    n_done = 0
    while n_done < N_mc:
        nb = min(batch, N_mc - n_done)
        # sample x0 ~ p0, T ~ Unif, xbar0 = S^T x0, ybar = xbar0 + sigma z
        x0, _, _ = gmm.sample(nb, rng=rng)
        T = rng.integers(0, d, nb)
        xbar = np.einsum('nij,nj->ni', Ps[T], x0)            # S^T x0
        ybar = xbar + sigma * rng.standard_normal((nb, d))

        # posterior over the C*d components of pbar0
        log_lik = np.stack([
            multivariate_normal.logpdf(ybar, mean=gbar.means[j], cov=chol_ok[j],
                                       allow_singular=True)
            for j in range(gbar.C)], axis=1)                 # (nb, C*d)
        log_post = np.log(gbar.weights + 1e-300)[None, :] + log_lik
        log_post -= log_post.max(1, keepdims=True)
        post = np.exp(log_post); post /= post.sum(1, keepdims=True)

        # E[xbar0 | ybar]
        Dstar = np.zeros((nb, d))
        for j in range(gbar.C):
            Dj = gbar.means[j] + (ybar - gbar.means[j]) @ gains[j].T
            Dstar += post[:, j:j + 1] * Dj
        r = xbar - Dstar                                     # (nb, d)
        sum_r2 += float((r ** 2).sum())

        # w_tau(ybar) = P(T=tau | ybar) = sum_c post[:, c*d + tau]
        w_tau = post.reshape(nb, C, d).sum(1)                # (nb, d)
        # M_n = S^{T_n} - sum_tau w_tau[n] S^tau
        M = Ps[T] - np.einsum('nt,tij->nij', w_tau, Ps)      # (nb, d, d)
        H += np.einsum('nij,nik->jk', M, M)
        g += np.einsum('nij,ni->j', M, r)
        n_done += nb

    H /= N_mc; g /= N_mc
    mmse_pbar0 = sum_r2 / N_mc
    # floor_free = E||r||^2 - g^T H^{-1} g   (H PSD; lstsq for safety)
    try:
        sol = np.linalg.solve(H + 1e-12 * np.eye(d), g)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(H, g, rcond=None)[0]
    bias_gain = float(g @ sol)
    return {
        'mmse_pbar0': float(mmse_pbar0),
        'floor_free': float(mmse_pbar0 - bias_gain),
        'bias_gain': bias_gain,
        'mmse_p0': float(gmm.mmse_uncond_exact(sigma, N_mc=min(N_mc, 200_000), rng=rng)),
    }


def _selftest():
    rng = np.random.default_rng(0)
    d, C = 8, 3
    means = np.zeros((C, d)); means[0, 0] = 1.5; means[1, 1] = -1.2; means[2, 2] = 1.0
    means += 0.4 * rng.standard_normal((C, d))
    covs = np.stack([np.diag(0.3 + 0.7 * rng.random(d)) for _ in range(C)])
    gmm = GaussianMixture(weights=np.array([0.5, 0.3, 0.2]), means=means, covs=covs)
    gbar = shift_symmetrized_gmm(gmm)

    # identity: Tr(Sigma_pbar0) - Tr(Sigma_p0) = ||(I-P1) mu||^2
    P1 = np.ones((d, d)) / d
    lhs = np.trace(gbar.Sigma) - np.trace(gmm.Sigma)
    rhs = float(np.sum(((np.eye(d) - P1) @ gmm.mu) ** 2))
    print(f"Tr(Sig_pbar0)-Tr(Sig_p0) = {lhs:.8f}   ||(I-P1)mu||^2 = {rhs:.8f}   "
          f"|diff|={abs(lhs-rhs):.2e}")
    assert abs(lhs - rhs) < 1e-8

    for sigma in (0.5, 2.0):
        out = mmse_equivariant_floors(gmm, sigma, N_mc=40_000,
                                      rng=np.random.default_rng(1))
        print(f"sigma={sigma}: MMSE(p0)={out['mmse_p0']:.4f}  "
              f"MMSE(pbar0)={out['mmse_pbar0']:.4f}  floor_free={out['floor_free']:.4f}  "
              f"bias_gain={out['bias_gain']:.4f}")
        assert out['bias_gain'] >= -1e-9, "bias gain must be >= 0"
        assert out['floor_free'] <= out['mmse_pbar0'] + 1e-9
        assert out['floor_free'] >= out['mmse_p0'] - 5e-2, \
            "free-bias floor cannot go below the unconstrained Bayes MMSE"
    print("equivariant_floor self-test PASSED")


if __name__ == '__main__':
    _selftest()
