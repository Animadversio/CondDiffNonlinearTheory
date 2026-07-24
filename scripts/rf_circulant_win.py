"""
rf_circulant_win.py  —  Section 4 of newfile5.tex: does L^circ beat L^dense?

For the GMM d=32 setup, sweep the RF width k (only k divisible by d, i.e. whole
d×d circulant blocks) and compare, on the SAME large "population" sample and the
SAME marginal projection law N(0,1/d):

    L^dense(k) : dense i.i.d. Theta,  UNCONSTRAINED readout W   (Stein, exact-in-noise)
    L^circ(k)  : block-circulant Theta, block-circulant readout W  (per-frequency solve)

both averaged over N_REP independent projection draws.

Writeup decomposition (eq 4.x):
    L^circ(k) - L^dense(k) = A_circ(k) + Delta_stat(sigma) - A_dense(k)
      A_dense(k)     = L^dense(k) - MMSE(p0)
      A_circ(k)      = L^circ(k)  - MMSE(pbar0)
      Delta_stat     = MMSE(pbar0) - MMSE(p0)   (stationarisation penalty, k-independent)
    pbar0 = shift-symmetrised data: xbar0 = S^T x0, T ~ Unif{0..d-1}. As a GMM this is
    the C*d components {(P_tau mu_c, P_tau Sigma_c P_tau^T, w_c/d)}.

The "win set" is {k : L^circ(k) < L^dense(k)}. Asymptotically (k->inf) both A->0 so
L^circ - L^dense -> Delta_stat > 0 (dense wins); at small k A_dense can blow up faster
than A_circ, opening a possible win window. This script computes/plots all of it and
prints the win set.

    D=32 python scripts/rf_circulant_win.py
    DEVICE=cuda K_MAX=4096 N_POP=16000 N_REP=6 python scripts/rf_circulant_win.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.gmm import GaussianMixture

D          = int(os.environ.get('D', '32'))
N_CLASSES  = 3
WEIGHTS    = [0.5, 0.3, 0.2]
SEED       = int(os.environ.get('SEED', '42'))
LAM        = float(os.environ.get('LAM', '1e-4'))
K_MAX      = int(os.environ.get('K_MAX', '4096'))
N_POP      = int(os.environ.get('N_POP', '16000'))     # population proxy sample size
N_REP      = int(os.environ.get('N_REP', '6'))         # projection draws to average
N_MC_EXACT = int(os.environ.get('N_MC_EXACT', '200000'))
SIGMA_VALUES = [float(s) for s in os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]
K_CIRC = [2 ** i for i in range(3, 16) if 2 ** i <= K_MAX and (2 ** i) % D == 0]

DEVICE = os.environ.get('DEVICE', 'cpu')
if DEVICE == 'cuda':
    import torch
    from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t
    from core.rf_circulant_torch import circulant_rf_mmse_t
    _DT = torch.float64
    def _dense(x0, U, Th, s):
        return stein_finiteN_mmse_t(x0, U, Th, np.zeros((Th.shape[0], N_CLASSES)), s, LAM,
                                    conditional=False, device='cuda', dtype=_DT)
    def _circ(x0, U, Th, s):
        return circulant_rf_mmse_t(x0, U, Th, np.zeros((Th.shape[0], N_CLASSES)), s, LAM,
                                   conditional=False, device='cuda', dtype=_DT)
else:
    from core.rf_gmm_estimators import stein_finiteN_mmse
    from core.rf_circulant import circulant_rf_mmse
    def _dense(x0, U, Th, s):
        return stein_finiteN_mmse(x0, U, Th, np.zeros((Th.shape[0], N_CLASSES)), s, LAM,
                                  conditional=False)
    def _circ(x0, U, Th, s):
        return circulant_rf_mmse(x0, U, Th, np.zeros((Th.shape[0], N_CLASSES)), s, LAM,
                                 conditional=False)

from core.rf_circulant import build_circulant_theta


def make_gmm(seed=SEED):
    """Same GMM as scripts/rf_gmm_finite_sample.py, at dimension D."""
    rng = np.random.default_rng(seed)
    d = D
    means = np.zeros((N_CLASSES, d))
    means[0, 0] = 2.0
    means[1, 0] = -1.0; means[1, 1] = 1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2
    s0 = np.full(d, 0.4); s0[0] = 1.2
    s1 = np.full(d, 0.4); s1[0] = 0.4; s1[1] = 1.0; s1[2] = 0.8
    A = rng.standard_normal((d, d)) * 0.3
    S2 = A @ A.T + 0.5 * np.eye(d)
    return GaussianMixture(weights=np.array(WEIGHTS), means=means,
                           covs=np.stack([np.diag(s0), np.diag(s1), S2]))


def shift_symmetrized_gmm(gmm):
    """pbar0: cyclic-shift symmetrisation. C*d components (P_tau mu_c, P_tau Sig_c P_tau^T,
    w_c/d), P_tau[i,j]=1 iff j=(i-tau) mod d (i.e. (P_tau x)_i = x_{i-tau})."""
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


def main():
    print(f"[circulant-win] d={D}, K_CIRC={K_CIRC}, N_POP={N_POP}, N_REP={N_REP}, "
          f"DEVICE={DEVICE}, sigmas={SIGMA_VALUES}")
    gmm = make_gmm(SEED)
    gmm_bar = shift_symmetrized_gmm(gmm)
    print(f"Tr(Sigma_p0)={np.trace(gmm.Sigma):.4f}   Tr(Sigma_pbar0)={np.trace(gmm_bar.Sigma):.4f}")

    # Population proxy sample (fixed across k and projections)
    rng = np.random.default_rng(SEED + 7)
    x0_pop, _, U_pop = gmm.sample(N_POP, rng=rng)

    # Population Bayes MMSE for p0 and pbar0 (per sigma)
    mmse_p0, mmse_pbar0 = {}, {}
    rng_mc = np.random.default_rng(SEED + 11)
    for s in SIGMA_VALUES:
        mmse_p0[s]    = gmm.mmse_uncond_exact(s, N_mc=N_MC_EXACT, rng=rng_mc)
        mmse_pbar0[s] = gmm_bar.mmse_uncond_exact(s, N_mc=N_MC_EXACT, rng=rng_mc)
        print(f"  σ={s}: MMSE(p0)={mmse_p0[s]:.4f}  MMSE(pbar0)={mmse_pbar0[s]:.4f}  "
              f"Δ_stat={mmse_pbar0[s]-mmse_p0[s]:.4f}")

    # Sweep k, averaging over N_REP projection draws
    L_dense = {s: [] for s in SIGMA_VALUES}
    L_circ  = {s: [] for s in SIGMA_VALUES}
    rng_proj = np.random.default_rng(SEED + 100)
    for k in tqdm(K_CIRC, desc='k'):
        dense_Th = [rng_proj.standard_normal((k, D)) / np.sqrt(D) for _ in range(N_REP)]
        circ_Th  = [build_circulant_theta(k, D, rng_proj) for _ in range(N_REP)]
        for s in SIGMA_VALUES:
            ld = np.mean([_dense(x0_pop, U_pop, Th, s) for Th in dense_Th])
            lc = np.mean([_circ(x0_pop, U_pop, Th, s) for Th in circ_Th])
            L_dense[s].append(float(ld))
            L_circ[s].append(float(lc))

    kd = np.array(K_CIRC) / D
    os.makedirs('tables', exist_ok=True)
    np.savez('tables/rf_circulant_win.npz',
             k_circ=np.array(K_CIRC), kd=kd, sigma_values=np.array(SIGMA_VALUES),
             **{f'L_dense_s{s}': np.array(L_dense[s]) for s in SIGMA_VALUES},
             **{f'L_circ_s{s}':  np.array(L_circ[s])  for s in SIGMA_VALUES},
             mmse_p0=np.array([mmse_p0[s] for s in SIGMA_VALUES]),
             mmse_pbar0=np.array([mmse_pbar0[s] for s in SIGMA_VALUES]))

    # ---- report win set ----
    print("\n=== Win set {k : L^circ(k) < L^dense(k)} ===")
    for s in SIGMA_VALUES:
        ld = np.array(L_dense[s]); lc = np.array(L_circ[s])
        win = [int(k) for k, a, b in zip(K_CIRC, lc, ld) if a < b]
        print(f"  σ={s}: win_k={win if win else 'none'}   "
              f"(min L^circ-L^dense = {np.min(lc-ld):+.4f} at k={K_CIRC[int(np.argmin(lc-ld))]})")

    # ---- figure: L^circ vs L^dense, plus the A_dense-A_circ vs Δ_stat view ----
    nS = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, nS, figsize=(5 * nS, 8.5))
    axes = np.asarray(axes).reshape(2, nS)
    fig.suptitle(f'Circulant vs dense nonlinear RF denoiser (newfile5 §4) — GMM d={D}, C={N_CLASSES}\n'
                 f'population proxy N={N_POP}, {N_REP} projection draws averaged', fontsize=11)
    for col, s in enumerate(SIGMA_VALUES):
        ld = np.array(L_dense[s]); lc = np.array(L_circ[s])
        # top: absolute losses
        ax = axes[0, col]
        ax.plot(kd, ld, color='teal', lw=2, marker='o', ms=4, label='$\\mathcal{L}^{\\mathrm{dense}}$ (unconstrained W)')
        ax.plot(kd, lc, color='darkviolet', lw=2, marker='s', ms=4, label='$\\mathcal{L}^{\\mathrm{circ}}$ (circulant W)')
        ax.axhline(mmse_p0[s], color='black', ls=':', lw=1.2, label='MMSE($p_0$)')
        ax.axhline(mmse_pbar0[s], color='gray', ls='--', lw=1.2, label='MMSE($\\bar p_0$)')
        win_mask = lc < ld
        if win_mask.any():
            ax.scatter(kd[win_mask], lc[win_mask], s=90, facecolors='none',
                       edgecolors='red', lw=1.8, zorder=5, label='win (circ<dense)')
        ax.set_xscale('log'); ax.set_xlabel('k / d'); ax.set_ylabel('MSE')
        ax.set_title(f'σ={s}'); ax.grid(True, alpha=.3)
        if col == nS - 1:
            ax.legend(fontsize=7, loc='upper right')
        # bottom: A_dense - A_circ vs Δ_stat  (win where this > Δ_stat)
        ax2 = axes[1, col]
        A_dense = ld - mmse_p0[s]
        A_circ  = lc - mmse_pbar0[s]
        dstat = mmse_pbar0[s] - mmse_p0[s]
        ax2.plot(kd, A_dense - A_circ, color='crimson', lw=2, marker='d', ms=4,
                 label='$A_{\\mathrm{dense}}(k)-A_{\\mathrm{circ}}(k)$')
        ax2.axhline(dstat, color='gray', ls='--', lw=1.4, label='$\\Delta_{\\mathrm{stat}}$ (win threshold)')
        ax2.axhline(0, color='k', lw=0.6)
        above = (A_dense - A_circ) > dstat
        if above.any():
            ax2.scatter(kd[above], (A_dense - A_circ)[above], s=90, facecolors='none',
                        edgecolors='red', lw=1.8, zorder=5, label='win region')
        ax2.set_xscale('log'); ax2.set_xlabel('k / d'); ax2.set_ylabel('excess-risk gap')
        ax2.set_title(f'σ={s}: circ wins where gap > Δ_stat'); ax2.grid(True, alpha=.3)
        if col == nS - 1:
            ax2.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/rf_circulant_win.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
