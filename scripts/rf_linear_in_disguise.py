"""
rf_linear_in_disguise.py — instantiate the "linear-in-disguise" threshold k_* of
docs/prop1_linear_in_disguise_proof_1.tex on the d=32 GMM, and test it against the
losses we actually measured.

Theory recap (Theorem thm:prob of the note)
-------------------------------------------
    L^RF_sigma  >=  L^lin_sigma  -  2 ||Sigma||_op^2 * k * check_eps_d^2 / delta
w.p. >= 1 - delta - 2k e^{-9d/64} - eta_{k,d}, hence (1/d)L^RF >= (1/d)L^lin - o(1)
whenever k * hat_eps_d^2 = o(d), i.e. below

    k_* = min( d / hat_eps_d^2 ,  d^{n_0 - o(1)} ).

The two ingredients, both computed here EXACTLY for a GMM (no product-measure
assumption, so Lemma 9 is not used and not needed):

  Stein defect      Delta_{theta,eps} = Sigma^{-1} Cov(x0, psi_s(theta^T x0 + eps))
                                        - alphabar(theta,eps) * theta,
                    psi_s = relu * N(0,s^2),  s = sigma||theta||.
  raw 4th moment    hat_eps_d^4 := E_{theta,eps} ||Delta||^4
  normalized defect check_eps_d^2 := E[ ||Delta||^2 / gamma(theta,eps) ; ||theta|| in [1/2,2] ],
                    gamma = E_{x0}[ sum_{n=n_0}^N n! c_n(theta^T x0 + eps, s)^2 ].

Closed forms used (all exact for a Gaussian mixture; x0|c ~ N(mu_c, Sigma_c), so
u = theta^T x0 + eps | c ~ N(m_c, v_c) with m_c = theta^T mu_c + eps, v_c = theta^T Sigma_c theta):

    E[psi_s(u)|c]      = c0(m_c, S_c),         S_c = sqrt(v_c + s^2)
    E[psi_s'(u)|c]     = Phi(m_c / S_c)
    Cov(x0, psi_s(u))  = sum_c w_c [ (mu_c - mu) c0(m_c,S_c) + Sigma_c theta Phi(m_c/S_c) ]
    alphabar           = sum_c w_c Phi(m_c/S_c)

gamma is a 1-D expectation over u and is done by Gauss-Hermite quadrature per component.

    D=32 python scripts/rf_linear_in_disguise.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.gmm import GaussianMixture

D        = int(os.environ.get('D', '32'))
N_CLASS  = 3
WEIGHTS  = [0.5, 0.3, 0.2]
SEED     = int(os.environ.get('SEED', '42'))
SIGMAS   = [float(s) for s in os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]
M_MC     = int(os.environ.get('M_MC', '200000'))     # theta draws for the defect moments
N0       = int(os.environ.get('N0', '2'))            # band start (>=2)
NBAND    = int(os.environ.get('NBAND', '6'))         # band end N
SIGMA_EPS = float(os.environ.get('SIGMA_EPS', '0.0'))  # offset scale; 0 = our uncond RF
TBL      = f'tables/rf_gmm_finite_sample/d{D}'


def make_gmm(seed=SEED, d=D):
    """The exact GMM used by scripts/rf_gmm_finite_sample.py."""
    rng = np.random.default_rng(seed)
    means = np.zeros((N_CLASS, d))
    means[0, 0] = 2.0
    means[1, 0] = -1.0; means[1, 1] = 1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2
    s0 = np.full(d, 0.4); s0[0] = 1.2
    s1 = np.full(d, 0.4); s1[0] = 0.4; s1[1] = 1.0; s1[2] = 0.8
    A = rng.standard_normal((d, d)) * 0.3
    S2 = A @ A.T + 0.5 * np.eye(d)
    return GaussianMixture(weights=np.array(WEIGHTS), means=means,
                           covs=np.stack([np.diag(s0), np.diag(s1), S2]))


# ---------------------------------------------------------------------------
# Assumption diagnostics
# ---------------------------------------------------------------------------
def check_assumptions(gmm):
    print("=" * 78)
    print("ASSUMPTION CHECK  (GMM d=%d, C=%d)" % (gmm.d, gmm.C))
    print("=" * 78)
    ev = np.linalg.eigvalsh(gmm.Sigma)
    ok1 = ev.min() > 1e-3
    print(f"[1] Sigma uniformly PD and bounded")
    print(f"    eig(Sigma) in [{ev.min():.4f}, {ev.max():.4f}], cond = {ev.max()/ev.min():.2f}"
          f"   -> {'HOLDS' if ok1 else 'FAILS'}")

    # sub-Gaussianity: a Gaussian mixture with finitely many bounded-mean, bounded-cov
    # components is sub-Gaussian; report the worst coordinate kurtosis as a sanity proxy.
    rng = np.random.default_rng(0)
    x0, _, _ = gmm.sample(200000, rng=rng)
    xc = x0 - x0.mean(0)
    kurt = (xc ** 4).mean(0) / (xc ** 2).mean(0) ** 2
    print(f"[2] coordinates uniformly sub-Gaussian")
    print(f"    max coordinate kurtosis = {kurt.max():.3f} (Gaussian = 3);"
          f" finite mixture of Gaussians is sub-Gaussian -> HOLDS")

    # product measure? (Lemma 9 only) -- report worst off-diagonal correlation
    Cr = np.corrcoef(x0, rowvar=False)
    off = np.abs(Cr - np.eye(gmm.d)).max()
    print(f"[3] product measure (needed ONLY by Lemma 9 / the cumulant table)")
    print(f"    max |off-diagonal correlation| = {off:.4f}  -> {'FAILS' if off > 0.02 else 'holds'}"
          f"  (we bypass Lemma 9 and compute the defect directly, so this is not needed)")
    print(f"[4] bias compatibility Xi < infty, dimension-independent")
    print(f"    offset law here is a POINT MASS at 0 (uncond RF) / {gmm.C} atoms (cond RF):")
    print(f"    compactly supported => Assumption 4' applies, Xi <= m(R_0)^-1 < infty -> HOLDS")
    return ev


# ---------------------------------------------------------------------------
# Stein defect (exact for a GMM)
# ---------------------------------------------------------------------------
def stein_defect(gmm, Theta, eps, sigma, Sinv):
    s = sigma * np.linalg.norm(Theta, axis=1)                    # (M,)
    m = Theta @ gmm.means.T + eps[:, None]                       # (M,C)
    v = np.einsum('kd,cde,ke->kc', Theta, gmm.covs, Theta)       # (M,C)
    S = np.sqrt(v + s[:, None] ** 2)
    z = m / S
    Phi = norm.cdf(z)
    c0 = m * Phi + S * norm.pdf(z)
    dmu = gmm.means - gmm.mu                                     # (C,d)
    SigTh = np.einsum('cde,ke->kcd', gmm.covs, Theta)            # (M,C,d)
    Cov = (np.einsum('c,kc,cd->kd', gmm.weights, c0, dmu)
           + np.einsum('c,kc,kcd->kd', gmm.weights, Phi, SigTh))
    abar = (gmm.weights[None, :] * Phi).sum(1)
    return Cov @ Sinv.T - abar[:, None] * Theta, s


def _cn(M, s, n):
    """ReLU Hermite coefficient c_n(M,s): c_1 = s Phi(z); c_n = (-1)^n s He_{n-2}(z) phi(z)/n!"""
    z = M / np.maximum(s, 1e-12)
    if n == 1:
        return s * norm.cdf(z)
    he = [np.ones_like(z), z]
    for j in range(2, n - 1):
        he.append(z * he[-1] - (j - 1) * he[-2])
    H = he[n - 2] if n - 2 < len(he) else (z * he[-1] - (len(he) - 1) * he[-2])
    from math import factorial
    return ((-1) ** n) * s * H * norm.pdf(z) / factorial(n)


def gamma_band(gmm, Theta, eps, sigma, n0, N, n_quad=40):
    """gamma(theta,eps) = E_{x0}[ sum_{n=n0}^N n! c_n(theta^T x0 + eps, s)^2 ].
    u | c ~ N(m_c, v_c): 1-D Gauss-Hermite quadrature per component."""
    from math import factorial
    s = sigma * np.linalg.norm(Theta, axis=1)                    # (M,)
    m = Theta @ gmm.means.T + eps[:, None]                       # (M,C)
    v = np.einsum('kd,cde,ke->kc', Theta, gmm.covs, Theta)       # (M,C)
    nodes, wts = np.polynomial.hermite_e.hermegauss(n_quad)      # weight e^{-x^2/2}
    wts = wts / np.sqrt(2 * np.pi)
    out = np.zeros(len(Theta))
    for ci in range(gmm.C):
        u = m[:, ci:ci + 1] + np.sqrt(v[:, ci:ci + 1]) * nodes[None, :]   # (M,q)
        acc = np.zeros_like(u)
        for n in range(n0, N + 1):
            acc += factorial(n) * _cn(u, s[:, None], n) ** 2
        out += gmm.weights[ci] * (acc * wts[None, :]).sum(1)
    return out


def main():
    gmm = make_gmm()
    ev = check_assumptions(gmm)
    Sinv = np.linalg.inv(gmm.Sigma)
    op_Sig = float(ev.max())

    rng = np.random.default_rng(0)
    Theta = rng.standard_normal((M_MC, D)) / np.sqrt(D)
    eps = (SIGMA_EPS * rng.standard_normal(M_MC)) if SIGMA_EPS > 0 else np.zeros(M_MC)
    norm_ok = (np.linalg.norm(Theta, axis=1) >= 0.5) & (np.linalg.norm(Theta, axis=1) <= 2.0)

    print("\n" + "=" * 78)
    print(f"STEIN DEFECT AND k_*   (offset sigma_eps={SIGMA_EPS}, band n0={N0}..N={NBAND},"
          f" {M_MC} theta draws)")
    print("=" * 78)
    print(f"{'sigma':>6} {'E||D||^2':>10} {'eps_hat^2':>10} {'chk_eps^2':>10} "
          f"{'d/eps_hat^2':>12} {'d^n0':>8} {'k_*':>10} {'k_*/d':>8}")
    rows = []
    for sg in SIGMAS:
        Dl, s = stein_defect(gmm, Theta, eps, sg, Sinv)
        n2 = (Dl ** 2).sum(1)
        eps_hat2 = float(np.sqrt((n2 ** 2).mean()))          # sqrt(E||D||^4)
        gam = gamma_band(gmm, Theta, eps, sg, N0, NBAND)
        chk = float((n2[norm_ok] / np.maximum(gam[norm_ok], 1e-300)).mean())
        k_def = D / eps_hat2
        k_ten = float(D ** N0)
        k_star = min(k_def, k_ten)
        rows.append(dict(sigma=sg, eps_hat2=eps_hat2, chk=chk, k_def=k_def,
                         k_ten=k_ten, k_star=k_star, mean_n2=float(n2.mean())))
        print(f"{sg:>6} {n2.mean():>10.5f} {eps_hat2:>10.5f} {chk:>10.5f} "
              f"{k_def:>12.1f} {k_ten:>8.0f} {k_star:>10.1f} {k_star/D:>8.1f}")
    print(f"\n  k_* = min(d/eps_hat^2, d^n0).  With n0={N0} the tensor budget d^n0={D**N0:.0f}"
          f" is {'BINDING' if any(r['k_ten'] < r['k_def'] for r in rows) else 'not binding'}.")
    print("  For ReLU (piecewise-linear, one kink) with a compactly supported offset law,")
    print("  Prop. A2/A2b let n0 be taken arbitrarily large, so the defect term d/eps_hat^2")
    print("  is the operative threshold; both are reported.")

    # ---- confront with the measured population losses -----------------------
    print("\n" + "=" * 78)
    print("DOES THE BOUND BIND?   observed (L^lin - L^RF) vs the slack the theorem allows")
    print("=" * 78)
    d0 = np.load(os.path.join(TBL, 'N1024.npz'), allow_pickle=True)
    kg = d0['k_grid']
    delta = 0.5
    for r in rows:
        sg = r['sigma']
        rf = d0[f'gmm_pop_u_s{sg}']                 # population RF theory (N->inf)
        lin = float(d0[f'wiener_pop_s{sg}'])        # population linear Wiener
        print(f"\n  sigma={sg}   k_* = {r['k_star']:.0f} ({r['k_star']/D:.0f}d)   "
              f"L^lin = {lin:.4f}")
        print(f"    {'k':>6} {'k/d':>6} {'L^RF':>9} {'gap=lin-RF':>11} {'gap/d':>8} "
              f"{'slack allowed':>14} {'bound binds?':>13}")
        for i, k in enumerate(kg):
            gap = lin - rf[i]
            slack = 2 * op_Sig ** 2 * k * r['chk'] / delta
            binds = slack < gap
            if k in (32, 128, 512, 2048, 4096):
                print(f"    {k:>6} {k/D:>6.0f} {rf[i]:>9.4f} {gap:>11.4f} {gap/D:>8.4f} "
                      f"{slack:>14.2f} {'YES' if binds else 'no (vacuous)':>13}")

    # ---- figure -------------------------------------------------------------
    nS = len(SIGMAS)
    fig, axes = plt.subplots(2, nS, figsize=(4.6 * nS, 8.0), squeeze=False)
    fig.suptitle(f'Linear-in-disguise threshold $k_*$ on the GMM  (d={D}, C={N_CLASS}, '
                 f'population / $N\\to\\infty$)\n'
                 r'$k_*=\min(d/\hat\varepsilon_d^{\,2},\ d^{\,n_0})$; '
                 'below it the theorem forbids the RF denoiser from beating the linear one '
                 'by more than $O(k\\check\\varepsilon_d^2)$', fontsize=12)
    for j, r in enumerate(rows):
        sg = r['sigma']
        rf = d0[f'gmm_pop_u_s{sg}']; lin = float(d0[f'wiener_pop_s{sg}'])
        bayes = float(d0[f'bayes_pop_s{sg}'])
        ax = axes[0][j]
        ax.semilogx(kg, rf, color='crimson', lw=2.2, marker='o', ms=4,
                    label='RF denoiser (pop theory)')
        ax.axhline(lin, color='darkorange', lw=1.8, label='linear denoiser (Wiener)')
        ax.axhline(bayes, color='black', ls=':', lw=1.2, label='Bayes MMSE')
        ax.axvline(r['k_star'], color='navy', ls='--', lw=2,
                   label=f"$k_*$ = {r['k_star']:.0f}")
        if r['k_ten'] < r['k_def']:
            ax.axvline(r['k_def'], color='navy', ls=':', lw=1.2, alpha=.6,
                       label=f"$d/\\hat\\varepsilon^2$ = {r['k_def']:.0f}")
        ax.set_xlabel('k'); ax.set_ylabel('MSE'); ax.grid(True, alpha=.3)
        ax.set_title(f'σ = {sg}')
        if j == nS - 1:
            ax.legend(fontsize=8, loc='upper right')

        ax2 = axes[1][j]
        gap = lin - rf
        slack = 2 * op_Sig ** 2 * np.array(kg) * r['chk'] / 0.5
        ax2.loglog(kg, np.maximum(gap, 1e-6), color='crimson', lw=2.2, marker='o', ms=4,
                   label=r'observed gap $L^{lin}-L^{RF}$')
        ax2.loglog(kg, slack, color='navy', lw=1.8, ls='--',
                   label=r'slack allowed: $2\|\Sigma\|^2k\check\varepsilon_d^2/\delta$')
        ax2.axvline(r['k_star'], color='navy', ls='--', lw=2, alpha=.6)
        ax2.set_xlabel('k'); ax2.set_ylabel('loss gap')
        ax2.set_title(f'σ = {sg}: bound is vacuous where blue > red')
        ax2.grid(True, alpha=.3, which='both')
        if j == nS - 1:
            ax2.legend(fontsize=8, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs('figures', exist_ok=True)
    out = f'figures/rf_linear_in_disguise_d{D}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")
    np.savez(f'tables/rf_linear_in_disguise_d{D}.npz',
             sigmas=np.array(SIGMAS),
             eps_hat2=np.array([r['eps_hat2'] for r in rows]),
             check_eps2=np.array([r['chk'] for r in rows]),
             k_star=np.array([r['k_star'] for r in rows]),
             k_defect=np.array([r['k_def'] for r in rows]), n0=N0, d=D)


if __name__ == '__main__':
    main()
