"""
rf_kstar_scaling_gmm.py — test Theorem (thm:prob) directly in the GMM setup, at large d.

What is plotted / measured
--------------------------
For each d, with sigma fixed and the RF population theory (N -> infinity):

  (a) normalized gain      G(k) := (L^lin - L^RF(k)) / d,  MAXIMISED over ~20 Theta seeds
      (the theorem is a high-probability statement over the design, so the adversarial
      quantity is the best draw, not the average),
  (b) the slack the theorem allows,  2||Sigma||_op^2 k check_eps_d^2 / (delta d),
  (c) the threshold k_* = min(d/hat_eps_d^2, d^{n_0}).

Claim under test: G_max(k) <= slack(k) for all k < k_*.

  (d) tolerance crossings  k_tau(d) := min{ k : G_max(k) >= tau }  for several tau.
      If pushing tau -> 0 forces k_tau -> infinity at a rate growing with d, that is the
      empirical signature that a d-dependent threshold must exist at all. Fitting
      log k_tau vs log d gives the exponent to compare with k_* ~ d^{n_0} / d/hat_eps^2.

GMM across d
------------
make_gmm in scripts/rf_gmm_finite_sample.py uses S2 = A A^T with A ~ 0.3 N(0,1), so
Tr(S2) ~ 0.09 d^2 -- growing d would make the components ever more heteroscedastic and
silently change the distribution's character (this is what made hat_eps_d RISE with d in
an earlier check). Here A is rescaled by sqrt(32/d) so Tr(S2) ~ d like the other
components, and the construction reduces EXACTLY to the published one at d = 32.

    D_LIST=32,64,128 SIGMA=1.0 N_SEED=20 python scripts/rf_kstar_scaling_gmm.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.gmm import GaussianMixture
from core.rf_gmm_estimators_torch import mmse_theory_gmm_pop_t

NC       = 3
WEIGHTS  = [0.5, 0.3, 0.2]
SEED     = 42
LAM      = 1e-4
SIGMA    = float(os.environ.get('SIGMA', '1.0'))
D_LIST   = [int(x) for x in os.environ.get('D_LIST', '32,64,128').split(',')]
N_SEED   = int(os.environ.get('N_SEED', '20'))
M_THETA  = int(os.environ.get('M_THETA', '120000'))
DELTA    = float(os.environ.get('DELTA', '0.5'))
N0       = int(os.environ.get('N0', '2'))
NBAND    = 6
TOLS     = [float(t) for t in os.environ.get('TOLS', '0.03,0.01,0.003,0.001').split(',')]
KMAX     = int(os.environ.get('KMAX', '4096'))


def make_gmm(d, seed=SEED):
    """Published make_gmm, with A rescaled so Tr(S2) ~ d (identical at d = 32)."""
    rng = np.random.default_rng(seed)
    means = np.zeros((NC, d))
    means[0, 0] = 2.0
    means[1, 0] = -1.0; means[1, 1] = 1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2
    s0 = np.full(d, 0.4); s0[0] = 1.2
    s1 = np.full(d, 0.4); s1[0] = 0.4; s1[1] = 1.0; s1[2] = 0.8
    A = rng.standard_normal((d, d)) * (0.3 * np.sqrt(32.0 / d))
    S2 = A @ A.T + 0.5 * np.eye(d)
    return GaussianMixture(weights=np.array(WEIGHTS), means=means,
                           covs=np.stack([np.diag(s0), np.diag(s1), S2]))


def _cn(M, s, n):
    z = M / np.maximum(s, 1e-12)
    if n == 1:
        return s * norm.cdf(z)
    he = [np.ones_like(z), z]
    for j in range(2, n - 1):
        he.append(z * he[-1] - (j - 1) * he[-2])
    H = he[n - 2]
    from math import factorial
    return ((-1) ** n) * s * H * norm.pdf(z) / factorial(n)


def defect_moments(gmm, sigma, M, n0=N0, N=NBAND, seed=0, n_quad=32):
    """Exact-in-x0 Stein defect for a GMM; returns (hat_eps^2, check_eps^2)."""
    from math import factorial
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
    hat2 = float(np.sqrt((n2 ** 2).mean()))
    # gamma: Mehler band energy, 1-D quadrature per component
    nodes, wts = np.polynomial.hermite_e.hermegauss(n_quad); wts = wts / np.sqrt(2 * np.pi)
    gam = np.zeros(M)
    for ci in range(gmm.C):
        u = m[:, ci:ci + 1] + np.sqrt(v[:, ci:ci + 1]) * nodes[None, :]
        acc = np.zeros_like(u)
        for n in range(n0, N + 1):
            acc += factorial(n) * _cn(u, s[:, None], n) ** 2
        gam += gmm.weights[ci] * (acc * wts[None, :]).sum(1)
    ok = (np.linalg.norm(Th, axis=1) >= 0.5) & (np.linalg.norm(Th, axis=1) <= 2.0)
    chk = float((n2[ok] / np.maximum(gam[ok], 1e-300)).mean())
    return hat2, chk


def main():
    print(f"sigma={SIGMA}  seeds={N_SEED}  delta={DELTA}  n0={N0}  d list={D_LIST}")
    res = {}
    for d in D_LIST:
        gmm = make_gmm(d)
        op = float(np.linalg.eigvalsh(gmm.Sigma).max())
        hat2, chk = defect_moments(gmm, SIGMA, M_THETA)
        k_star = min(d / hat2, float(d) ** N0)
        lin = gmm.mmse_uncond_wiener(SIGMA)
        KG = [k for k in (2 ** np.arange(3, 20)) if d <= k <= KMAX]
        gains = np.zeros((len(KG), N_SEED))
        for i, k in enumerate(KG):
            for sd in range(N_SEED):
                rp = np.random.default_rng(9000 + 131 * sd + k)
                Th = rp.standard_normal((k, d)) / np.sqrt(d)
                L = mmse_theory_gmm_pop_t(gmm, Th, np.zeros((k, NC)), SIGMA, lam=LAM,
                                          conditional=False, device='cuda', dtype=torch.float64)
                gains[i, sd] = (lin - L) / d
        gmax = gains.max(1)
        slack = 2 * op ** 2 * np.array(KG, float) * chk / (DELTA * d)
        res[d] = dict(KG=np.array(KG, float), gmax=gmax, gmean=gains.mean(1), slack=slack,
                      k_star=k_star, hat2=hat2, chk=chk, lin=lin, op=op,
                      k_def=d / hat2, k_ten=float(d) ** N0)
        viol = [(k, g, s) for k, g, s in zip(KG, gmax, slack) if k < k_star and g > s]
        print(f"\n d={d:>4}  ||Sigma||={op:.3f}  hat_eps^2={hat2:.5f}  check_eps^2={chk:.5f}"
              f"  k_*={k_star:.0f} (d/hat={d/hat2:.0f}, d^{N0}={float(d)**N0:.0f})")
        print(f"   {'k':>7} {'max gain/d':>11} {'mean gain/d':>12} {'slack':>11} {'ok?':>5}")
        for k, g, gm, s in zip(KG, gmax, gains.mean(1), slack):
            tag = 'ok' if (k >= k_star or g <= s) else 'VIOL'
            print(f"   {k:>7} {g:>11.5f} {gm:>12.5f} {s:>11.5f} {tag:>5}")
        print(f"   -> violations of  G_max <= slack  for k < k_*: {len(viol)}")

    # tolerance crossings
    print("\n" + "=" * 78)
    print("SMALLEST k WITH max-over-seeds gain/d >= tau     k_tau(d)")
    print("=" * 78)
    print(f"{'tau':>8} " + " ".join(f"{'d='+str(d):>10}" for d in D_LIST) + "   fitted slope")
    ktab = {}
    for tau in TOLS:
        row = []
        for d in D_LIST:
            r = res[d]
            idx = np.where(r['gmax'] >= tau)[0]
            row.append(float(r['KG'][idx[0]]) if len(idx) else np.nan)
        ktab[tau] = row
        ok = ~np.isnan(row)
        sl = (np.polyfit(np.log(np.array(D_LIST)[ok]), np.log(np.array(row)[ok]), 1)[0]
              if ok.sum() >= 2 else np.nan)
        print(f"{tau:>8} " + " ".join(f"{v:>10.0f}" if not np.isnan(v) else f"{'--':>10}"
                                      for v in row) + f"   {sl:+.2f}")
    print(f"\n  k_* itself scales as: " + " ".join(
        f"d={d}: {res[d]['k_star']:.0f}" for d in D_LIST))
    sl_ks = np.polyfit(np.log(D_LIST), np.log([res[d]['k_star'] for d in D_LIST]), 1)[0]
    sl_kd = np.polyfit(np.log(D_LIST), np.log([res[d]['k_def'] for d in D_LIST]), 1)[0]
    print(f"  fitted slope of k_*      vs d : {sl_ks:+.2f}")
    print(f"  fitted slope of d/hat_eps^2   : {sl_kd:+.2f}")

    # ---- figure ----
    nD = len(D_LIST)
    fig, axes = plt.subplots(2, nD, figsize=(5.2 * nD, 8.4), squeeze=False)
    fig.suptitle(f'Theorem test in the GMM setup, population / $N\\to\\infty$, σ={SIGMA}, '
                 f'{N_SEED} Θ-seeds\n'
                 r'top: normalized gain $(L^{\rm lin}-L^{\rm RF})/d$ vs the allowed slack '
                 r'$2\|\Sigma\|^2k\check\varepsilon_d^2/(\delta d)$;  bottom: $k_\tau(d)$',
                 fontsize=13)
    for j, d in enumerate(D_LIST):
        r = res[d]; ax = axes[0][j]
        ax.loglog(r['KG'], np.maximum(r['gmax'], 1e-8), color='crimson', lw=2.2, marker='o',
                  ms=4, label='max over Θ-seeds')
        ax.loglog(r['KG'], np.maximum(r['gmean'], 1e-8), color='crimson', lw=1.0, ls=':',
                  alpha=.7, label='mean over Θ-seeds')
        ax.loglog(r['KG'], r['slack'], color='navy', lw=2, ls='--', label='allowed slack')
        ax.axvline(r['k_star'], color='green', lw=2, label=f"$k_*$={r['k_star']:.0f}")
        for tau in TOLS:
            ax.axhline(tau, color='gray', lw=.6, alpha=.5)
        ax.set_xlabel('k'); ax.set_ylabel(r'$(L^{\rm lin}-L^{\rm RF})/d$')
        ax.set_title(f'd = {d}'); ax.grid(True, alpha=.3, which='both')
        if j == nD - 1: ax.legend(fontsize=8, loc='lower right')
    ax = axes[1][0]
    for tau in TOLS:
        v = np.array(ktab[tau], float)
        ax.loglog(D_LIST, v, marker='o', lw=2, label=f'τ = {tau}')
    ax.loglog(D_LIST, [res[d]['k_star'] for d in D_LIST], color='green', lw=2.4, ls='--',
              marker='s', label='$k_*$')
    ax.set_xlabel('d'); ax.set_ylabel(r'$k_\tau$ = smallest k with gain/d ≥ τ')
    ax.set_title('tolerance crossings vs d'); ax.grid(True, alpha=.3, which='both')
    ax.legend(fontsize=9)
    for j in range(1, nD): axes[1][j].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs('figures', exist_ok=True)
    out = f'figures/rf_kstar_scaling_gmm_s{SIGMA}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")
    np.savez(f'tables/rf_kstar_scaling_gmm_s{SIGMA}.npz',
             d_list=np.array(D_LIST), tols=np.array(TOLS),
             **{f'gmax_d{d}': res[d]['gmax'] for d in D_LIST},
             **{f'slack_d{d}': res[d]['slack'] for d in D_LIST},
             **{f'KG_d{d}': res[d]['KG'] for d in D_LIST},
             k_star=np.array([res[d]['k_star'] for d in D_LIST]),
             ktau=np.array([ktab[t] for t in TOLS]))


if __name__ == '__main__':
    main()
