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
      A_circ(k)      = L^circ(k)  - floor_circ
      Delta_stat     = floor_circ - MMSE(p0)    (stationarisation penalty, k-independent)
    pbar0 = shift-symmetrised data: xbar0 = S^T x0, T ~ Unif{0..d-1}. As a GMM this is
    the C*d components {(P_tau mu_c, P_tau Sigma_c P_tau^T, w_c/d)}.

IMPORTANT — which floor.  The readout here uses a FREE bias b in R^d (the usual affine
setup, equivariant_bias=False).  W phi(.) is shift-equivariant but a free per-position
bias is NOT, so the model class is "equivariant map + arbitrary constant", strictly
larger than strictly-equivariant.  Its floor is therefore NOT MMSE(pbar0) but the lower
    floor_free = MMSE(pbar0) - g^T H^{-1} g
computed exactly in core.equivariant_floor (the bias-optimisation is quadratic).  Using
MMSE(pbar0) with a free bias produces spurious "L^circ below its own floor" readings —
the free bias buys back part of the mean's non-stationary energy, at most
||(I - P_1) mu_p0||^2 = Tr(Sigma_pbar0) - Tr(Sigma_p0).  So floor_circ := floor_free here.

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
from core.equivariant_floor import mmse_equivariant_floors, shift_symmetrized_gmm as _ssg

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
# Effective-dimension knob: number of coordinates carrying non-trivial (mean +
# anisotropy) structure. The data stays NON-stationary (axis-pinned to the first
# m_active coords) — we do NOT stationarise (that would trivially favour circulant).
# Comma-separated -> sweep. Signal budget is normalised (below) so total mean
# separation and anisotropic variance are m-independent, isolating effective dim.
M_ACTIVE_LIST = [int(x) for x in os.environ.get('M_ACTIVE', '3,8,14,20,26').split(',')]
# Filter bandwidth sweep: support width w of the circulant generating kernel h_a.
# w<d gives a LOCAL (banded) filter — the true conv analogue, each feature reads a
# length-w window; h_a[:w] ~ N(0,I/w) keeps E||h_a||^2=1 so row norms match dense for
# every w. w=d is the original full-width kernel (shift-equivariant but non-local:
# all d coords weighted equally, so no locality advantage). Only L^circ depends on w —
# L^dense, MMSE(p0), MMSE(pbar0), Delta_stat are all w-independent.
W_BAND_LIST = [int(x) for x in os.environ.get('W_BAND', '2,4,8,16,32').split(',')]

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


BASE_VAR = 0.4       # baseline (filler) coordinate variance
SEP      = 2.5       # per-class mean norm (fixed across m_active)
ANISO    = 3.0       # total excess (above-baseline) variance per class (fixed across m_active)


def make_gmm_active(m_active, seed=SEED):
    """Non-stationary GMM (D dims, N_CLASSES comps) whose mean + anisotropy structure
    is confined to the first m_active coordinates (axis-pinned -> NOT shift-invariant).
    Normalised so the total signal budget is m-independent: each class mean has norm
    SEP, and each class's excess variance above BASE_VAR sums to ANISO, spread over the
    m_active active coords. This isolates *effective dimension* (how many coords carry
    structure) from raw signal power, so a win opening as m grows is about dimension,
    not power. Filler coords (m_active..D-1): zero mean, BASE_VAR variance."""
    rng = np.random.default_rng(seed + 1000 * m_active)
    d = D
    m = int(np.clip(m_active, 1, d))
    # class means: centred unit directions in the active block, scaled to norm SEP
    G = rng.standard_normal((N_CLASSES, m))
    G -= G.mean(0, keepdims=True)                       # centre so overall mean modest
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
    means = np.zeros((N_CLASSES, d))
    means[:, :m] = SEP * G
    covs = []
    for c in range(N_CLASSES):
        exc = rng.random(m) + 0.2
        exc *= ANISO / exc.sum()                        # excess var, total ANISO over m coords
        diag = np.full(d, BASE_VAR)
        diag[:m] = BASE_VAR + exc
        S = np.diag(diag)
        if c == N_CLASSES - 1:                           # one class: dense (rotated) active block
            Q = np.linalg.qr(rng.standard_normal((m, m)))[0]
            S[:m, :m] = Q @ np.diag(BASE_VAR + exc) @ Q.T
        covs.append(S)
    return GaussianMixture(weights=np.array(WEIGHTS), means=means, covs=np.stack(covs))


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


def run_one(m_active):
    """Compute L^dense(k), L^circ(k) and the MMSE floors for one active-dim GMM.
    Saves a detailed per-σ figure; returns a results dict for the trend summary."""
    gmm = make_gmm_active(m_active, SEED)
    gmm_bar = shift_symmetrized_gmm(gmm)
    print(f"\n=== m_active={m_active} ===  Tr(Σp0)={np.trace(gmm.Sigma):.4f}  "
          f"Tr(Σpbar0)={np.trace(gmm_bar.Sigma):.4f}")

    rng = np.random.default_rng(SEED + 7)
    x0_pop, _, U_pop = gmm.sample(N_POP, rng=rng)

    # Floors. The readout bias is FREE (equivariant_bias=False), so the denoiser is
    # "equivariant map + arbitrary constant" — a strictly larger class than strictly
    # shift-equivariant. Its floor is therefore NOT MMSE(pbar0) but the lower
    # 'floor_free' = MMSE(pbar0) - g^T H^{-1} g (core.equivariant_floor). Using
    # MMSE(pbar0) here yields spurious "L^circ below its own floor" violations.
    # A_circ and Delta_stat are defined against floor_free to keep the Sec.4
    # decomposition L^circ - L^dense = A_circ + Delta_stat - A_dense exact.
    mmse_p0, mmse_pbar0, floor_free = {}, {}, {}
    rng_mc = np.random.default_rng(SEED + 11)
    for s in SIGMA_VALUES:
        fl = mmse_equivariant_floors(gmm, s, N_mc=min(N_MC_EXACT, 60_000), rng=rng_mc)
        mmse_p0[s]    = fl['mmse_p0']
        mmse_pbar0[s] = fl['mmse_pbar0']
        floor_free[s] = fl['floor_free']
        print(f"  σ={s}: MMSE(p0)={mmse_p0[s]:.4f}  floor_free={floor_free[s]:.4f}  "
              f"MMSE(pbar0)={mmse_pbar0[s]:.4f}  (bias gain {fl['bias_gain']:.4f})  "
              f"Δ_stat={floor_free[s]-mmse_p0[s]:.4f}")

    # L_dense / floors are w-independent; L_circ is swept over filter bandwidth w.
    L_dense = {s: [] for s in SIGMA_VALUES}
    L_circ  = {w: {s: [] for s in SIGMA_VALUES} for w in W_BAND_LIST}
    rng_proj = np.random.default_rng(SEED + 100)
    for k in tqdm(K_CIRC, desc=f'k (m={m_active})'):
        dense_Th = [rng_proj.standard_normal((k, D)) / np.sqrt(D) for _ in range(N_REP)]
        circ_Th = {w: [build_circulant_theta(k, D, rng_proj, w=w) for _ in range(N_REP)]
                   for w in W_BAND_LIST}
        for s in SIGMA_VALUES:
            L_dense[s].append(float(np.mean([_dense(x0_pop, U_pop, Th, s) for Th in dense_Th])))
            for w in W_BAND_LIST:
                L_circ[w][s].append(float(np.mean([_circ(x0_pop, U_pop, Th, s)
                                                   for Th in circ_Th[w]])))

    kd = np.array(K_CIRC) / D
    print(f"  Win set {{k : L^circ<L^dense}} (m_active={m_active}), by filter width w:")
    for s in SIGMA_VALUES:
        ld = np.array(L_dense[s])
        row = []
        for w in W_BAND_LIST:
            lc = np.array(L_circ[w][s])
            gap = np.min(lc - ld)
            win = [int(k) for k, a, b in zip(K_CIRC, lc, ld) if a < b]
            row.append(f"w={w}:{gap:+.3f}{'*' if win else ''}")
        best_w = W_BAND_LIST[int(np.argmin([np.min(np.array(L_circ[w][s]) - ld)
                                            for w in W_BAND_LIST]))]
        print(f"    σ={s}: min(L^circ-L^dense) by w -> {'  '.join(row)}   [best w={best_w}]"
              f"   (* = win region exists)")

    # detailed per-σ figure
    nS = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, nS, figsize=(5 * nS, 8.5))
    axes = np.asarray(axes).reshape(2, nS)
    fig.suptitle(f'Circulant vs dense nonlinear RF denoiser (newfile5 §4) — GMM d={D}, C={N_CLASSES}, '
                 f'{m_active} active coords (non-stationary)\n'
                 f'population proxy N={N_POP}, {N_REP} projection draws averaged', fontsize=11)
    wcolors = plt.cm.viridis(np.linspace(0.15, 0.9, len(W_BAND_LIST)))
    for col, s in enumerate(SIGMA_VALUES):
        ld = np.array(L_dense[s])
        ax = axes[0, col]
        ax.plot(kd, ld, color='teal', lw=2.4, marker='o', ms=4,
                label='$\\mathcal{L}^{\\mathrm{dense}}$ (unconstrained W)')
        for wi, w in enumerate(W_BAND_LIST):
            lc = np.array(L_circ[w][s])
            ax.plot(kd, lc, color=wcolors[wi], lw=1.6, marker='s', ms=3.5,
                    label=f'$\\mathcal{{L}}^{{\\mathrm{{circ}}}}$ w={w}' + (' (full)' if w >= D else ''))
            win_mask = lc < ld
            if win_mask.any():
                ax.scatter(kd[win_mask], lc[win_mask], s=90, facecolors='none',
                           edgecolors='red', lw=1.8, zorder=5)
        ax.axhline(mmse_p0[s], color='black', ls=':', lw=1.2, label='MMSE($p_0$)')
        ax.axhline(floor_free[s], color='dimgray', ls='--', lw=1.4,
                   label='floor$_{\\mathrm{free}}$ (equiv + free bias)')
        ax.axhline(mmse_pbar0[s], color='silver', ls=':', lw=1.1,
                   label='MMSE($\\bar p_0$) [strict-equiv only]')
        ax.set_xscale('log'); ax.set_xlabel('k / d'); ax.set_ylabel('MSE')
        ax.set_title(f'σ={s}   (red ring = circ beats dense)'); ax.grid(True, alpha=.3)
        if col == nS - 1:
            ax.legend(fontsize=6.5, loc='upper right')
        ax2 = axes[1, col]
        A_dense = ld - mmse_p0[s]
        dstat = floor_free[s] - mmse_p0[s]
        for wi, w in enumerate(W_BAND_LIST):
            A_circ = np.array(L_circ[w][s]) - floor_free[s]
            ax2.plot(kd, A_dense - A_circ, color=wcolors[wi], lw=1.6, marker='d', ms=3.5,
                     label=f'w={w}')
        ax2.axhline(dstat, color='gray', ls='--', lw=1.6, label='$\\Delta_{\\mathrm{stat}}$ (win threshold)')
        ax2.axhline(0, color='k', lw=0.6)
        ax2.set_xscale('log'); ax2.set_xlabel('k / d'); ax2.set_ylabel('$A_{dense}-A_{circ}$')
        ax2.set_title(f'σ={s}: circ wins where gap > Δ_stat'); ax2.grid(True, alpha=.3)
        if col == nS - 1:
            ax2.legend(fontsize=6.5, loc='upper right')
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f'figures/rf_circulant_win_mact{m_active}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved {out}")

    return dict(m_active=m_active, kd=kd, L_dense=L_dense, L_circ=L_circ,
                mmse_p0=mmse_p0, mmse_pbar0=mmse_pbar0, floor_free=floor_free)


def main():
    print(f"[circulant-win] d={D}, K_CIRC={K_CIRC}, N_POP={N_POP}, N_REP={N_REP}, "
          f"DEVICE={DEVICE}, sigmas={SIGMA_VALUES}, M_ACTIVE={M_ACTIVE_LIST}, "
          f"W_BAND={W_BAND_LIST}")
    results = [run_one(m) for m in M_ACTIVE_LIST]

    os.makedirs('tables', exist_ok=True)
    sd = {'k_circ': np.array(K_CIRC), 'sigma_values': np.array(SIGMA_VALUES),
          'm_active_list': np.array(M_ACTIVE_LIST), 'w_band_list': np.array(W_BAND_LIST)}
    for r in results:
        m = r['m_active']
        for s in SIGMA_VALUES:
            sd[f'L_dense_m{m}_s{s}'] = np.array(r['L_dense'][s])
            for w in W_BAND_LIST:
                sd[f'L_circ_m{m}_w{w}_s{s}'] = np.array(r['L_circ'][w][s])
            sd[f'mmse_p0_m{m}_s{s}']    = r['mmse_p0'][s]
            sd[f'mmse_pbar0_m{m}_s{s}'] = r['mmse_pbar0'][s]
            sd[f'floor_free_m{m}_s{s}'] = r['floor_free'][s]
    np.savez('tables/rf_circulant_win.npz', **sd)

    # ---- trend summary: min gap vs filter width w (and vs m_active) ----
    nS = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, nS, figsize=(5 * nS, 8.5))
    axes = np.asarray(axes).reshape(2, nS)
    fig.suptitle('Does a LOCAL (banded) circulant filter open a win? '
                 f'(non-stationary GMM, d={D})\n'
                 'top: min_k (L^circ−L^dense) vs filter width w  (<0 ⇒ win);  '
                 'bottom: same vs #active coords, at the best w', fontsize=11)
    ms = np.array([r['m_active'] for r in results])
    ws = np.array(W_BAND_LIST)
    mcolors = plt.cm.plasma(np.linspace(0.1, 0.85, len(results)))
    for col, s in enumerate(SIGMA_VALUES):
        ax = axes[0, col]
        for ri, r in enumerate(results):
            ld = np.array(r['L_dense'][s])
            gaps = np.array([np.min(np.array(r['L_circ'][w][s]) - ld) for w in W_BAND_LIST])
            ax.plot(ws, gaps, color=mcolors[ri], lw=1.8, marker='o', ms=5,
                    label=f"m={r['m_active']}")
            winm = gaps < 0
            if winm.any():
                ax.scatter(ws[winm], gaps[winm], s=110, facecolors='none',
                           edgecolors='green', lw=2, zorder=5)
        ax.axhline(0, color='k', lw=1.2, ls='--')
        ax.set_xscale('log', base=2); ax.set_xlabel('filter width w')
        ax.set_ylabel('min_k (L^circ − L^dense)')
        ax.set_title(f'σ={s}  (green ring = win)'); ax.grid(True, alpha=.3)
        if col == nS - 1:
            ax.legend(fontsize=7)
        # bottom: best-over-w gap vs m_active
        ax2 = axes[1, col]
        best_gap = np.array([np.min([np.min(np.array(r['L_circ'][w][s]) - np.array(r['L_dense'][s]))
                                     for w in W_BAND_LIST]) for r in results])
        ax2.plot(ms, best_gap, color='crimson', lw=2, marker='o', ms=6, label='best over w')
        full_gap = np.array([np.min(np.array(r['L_circ'][W_BAND_LIST[-1]][s]) - np.array(r['L_dense'][s]))
                             for r in results])
        ax2.plot(ms, full_gap, color='gray', lw=1.6, ls='--', marker='s', ms=5,
                 label=f'full-width w={W_BAND_LIST[-1]}')
        ax2.axhline(0, color='k', lw=1.2, ls='--')
        ax2.set_xlabel('# active coords'); ax2.set_ylabel('min_k (L^circ − L^dense)')
        ax2.set_title(f'σ={s}: locality gain vs effective dim'); ax2.grid(True, alpha=.3)
        if col == nS - 1:
            ax2.legend(fontsize=7)
    fig.tight_layout()
    out = 'figures/rf_circulant_win_trend.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
