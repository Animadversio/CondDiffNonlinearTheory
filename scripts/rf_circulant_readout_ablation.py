"""
rf_circulant_readout_ablation.py — separate the circulant FEATURE MAP from the
circulant READOUT.

Three models, all with the same population moments (exact per-component Stein/Hermite,
no sampling error) and the same marginal projection law N(0, 1/d):

  1. dense Theta   + FREE W, b     "L^dense"            (mmse_theory_gmm_pop_t)
  2. circulant Th  + FREE W, b     "L^circTh-freeW"     (same formula, circulant Theta)
  3. circulant Th  + CIRCULANT W   "L^circ"             (per-frequency solve)

(2) is the new one requested: it keeps the block-circulant / banded feature map but lets
the readout be unconstrained, so the standard Hermite/Mehler denoiser loss applies
directly. Comparing 2 vs 1 isolates the value of the circulant+local FEATURE MAP;
comparing 3 vs 2 isolates the COST of constraining the readout.

Floors: models 1 and 2 have a free (non-equivariant) readout, so their floor is the full
MMSE(p0). Model 3 is equivariant-plus-constant, floor = floor_free (core.equivariant_floor).

    DEVICE=cuda python scripts/rf_circulant_readout_ablation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from core.gmm import GaussianMixture
from core.rf_circulant import build_circulant_theta
from core.rf_circulant_torch import circulant_rf_mmse_pop_t
from core.rf_gmm_estimators_torch import mmse_theory_gmm_pop_t
from core.equivariant_floor import mmse_equivariant_floors

D = int(os.environ.get('D', '32')); N_CLASSES = 3; WEIGHTS = [0.5, 0.3, 0.2]
SEED = int(os.environ.get('SEED', '42')); LAM = float(os.environ.get('LAM', '1e-4'))
K_MAX = int(os.environ.get('K_MAX', '4096')); N_REP = int(os.environ.get('N_REP', '6'))
N_MC = int(os.environ.get('N_MC_EXACT', '200000'))
M_ACTIVE = int(os.environ.get('M_ACTIVE', '20'))
SIGMA_VALUES = [float(s) for s in os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]
W_BAND_LIST = [int(x) for x in os.environ.get('W_BAND', '2,4,8,32').split(',')]
K_CIRC = [2 ** i for i in range(3, 16) if 2 ** i <= K_MAX and (2 ** i) % D == 0]
BASE_VAR, SEP, ANISO = 0.4, 2.5, 3.0
_DT = torch.float64


def make_gmm_active(m_active, seed=SEED):
    rng = np.random.default_rng(seed + 1000 * m_active)
    m = int(np.clip(m_active, 1, D))
    G = rng.standard_normal((N_CLASSES, m)); G -= G.mean(0, keepdims=True)
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
    means = np.zeros((N_CLASSES, D)); means[:, :m] = SEP * G
    covs = []
    for c in range(N_CLASSES):
        exc = rng.random(m) + 0.2; exc *= ANISO / exc.sum()
        diag = np.full(D, BASE_VAR); diag[:m] = BASE_VAR + exc
        S = np.diag(diag)
        if c == N_CLASSES - 1:
            Q = np.linalg.qr(rng.standard_normal((m, m)))[0]
            S[:m, :m] = Q @ np.diag(BASE_VAR + exc) @ Q.T
        covs.append(S)
    return GaussianMixture(weights=np.array(WEIGHTS), means=means, covs=np.stack(covs))


def main():
    gmm = make_gmm_active(M_ACTIVE)
    print(f"[readout-ablation] d={D}, m_active={M_ACTIVE}, K={K_CIRC}, W_BAND={W_BAND_LIST}, "
          f"N_REP={N_REP}  (exact population moments)")
    Z = np.zeros((1, N_CLASSES))
    floors = {}
    rm = np.random.default_rng(SEED + 11)
    for s in SIGMA_VALUES:
        fl = mmse_equivariant_floors(gmm, s, N_mc=N_MC, rng=rm)
        floors[s] = fl
        print(f"  σ={s}: MMSE(p0)={fl['mmse_p0']:.4f}  floor_free={fl['floor_free']:.4f}  "
              f"MMSE(pbar0)={fl['mmse_pbar0']:.4f}")

    L_dense = {s: [] for s in SIGMA_VALUES}
    L_cf = {w: {s: [] for s in SIGMA_VALUES} for w in W_BAND_LIST}   # circTh + free W
    L_cc = {w: {s: [] for s in SIGMA_VALUES} for w in W_BAND_LIST}   # circTh + circ W
    rp = np.random.default_rng(SEED + 100)
    for k in tqdm(K_CIRC, desc='k'):
        Ga = np.zeros((k, N_CLASSES))
        dTh = [rp.standard_normal((k, D)) / np.sqrt(D) for _ in range(N_REP)]
        cTh = {w: [build_circulant_theta(k, D, rp, w=w) for _ in range(N_REP)] for w in W_BAND_LIST}
        for s in SIGMA_VALUES:
            L_dense[s].append(float(np.mean([
                mmse_theory_gmm_pop_t(gmm, T, Ga, s, lam=LAM, conditional=False,
                                      device='cuda', dtype=_DT) for T in dTh])))
            for w in W_BAND_LIST:
                L_cf[w][s].append(float(np.mean([
                    mmse_theory_gmm_pop_t(gmm, T, Ga, s, lam=LAM, conditional=False,
                                          device='cuda', dtype=_DT) for T in cTh[w]])))
                L_cc[w][s].append(float(np.mean([
                    circulant_rf_mmse_pop_t(gmm, T, Ga, s, lam=LAM, conditional=False,
                                            device='cuda', dtype=_DT) for T in cTh[w]])))

    kd = np.array(K_CIRC) / D
    os.makedirs('tables', exist_ok=True)
    sd = {'k_circ': np.array(K_CIRC), 'sigma_values': np.array(SIGMA_VALUES),
          'w_band_list': np.array(W_BAND_LIST), 'm_active': M_ACTIVE}
    for s in SIGMA_VALUES:
        sd[f'L_dense_s{s}'] = np.array(L_dense[s])
        sd[f'mmse_p0_s{s}'] = floors[s]['mmse_p0']
        sd[f'floor_free_s{s}'] = floors[s]['floor_free']
        for w in W_BAND_LIST:
            sd[f'L_circTh_freeW_w{w}_s{s}'] = np.array(L_cf[w][s])
            sd[f'L_circ_w{w}_s{s}'] = np.array(L_cc[w][s])
    np.savez('tables/rf_circulant_readout_ablation.npz', **sd)

    print("\n=== Does circulant Theta + FREE W beat dense Theta + free W? ===")
    for s in SIGMA_VALUES:
        ld = np.array(L_dense[s])
        for w in W_BAND_LIST:
            g = np.array(L_cf[w][s]) - ld
            win = [int(kk) for kk, v in zip(K_CIRC, g) if v < 0]
            print(f"  σ={s} w={w:>2}: min(circTh_freeW - dense) = {g.min():+.4f} at k={K_CIRC[int(g.argmin())]}"
                  f"   win_k={win if win else 'none'}")
    print("\n=== Cost of constraining the readout (L^circ - L^circTh_freeW) ===")
    for s in SIGMA_VALUES:
        for w in W_BAND_LIST:
            c = np.array(L_cc[w][s]) - np.array(L_cf[w][s])
            print(f"  σ={s} w={w:>2}: k->small {c[0]:+.4f}  k->large {c[-1]:+.4f}")

    nS = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, nS, figsize=(5 * nS, 8.5))
    axes = np.asarray(axes).reshape(2, nS)
    fig.suptitle(f'Circulant FEATURE MAP vs circulant READOUT — GMM d={D}, {M_ACTIVE} active coords\n'
                 f'exact population moments, {N_REP} projection draws.  '
                 f'top: absolute loss;  bottom: circTh+freeW minus dense (<0 = circulant features win)',
                 fontsize=11)
    wc = plt.cm.viridis(np.linspace(0.15, 0.85, len(W_BAND_LIST)))
    for col, s in enumerate(SIGMA_VALUES):
        ld = np.array(L_dense[s]); ax = axes[0, col]
        ax.plot(kd, ld, color='crimson', lw=2.6, marker='o', ms=4.5, zorder=4,
                label='dense $\\Theta$ + free $W$')
        for wi, w in enumerate(W_BAND_LIST):
            ax.plot(kd, L_cf[w][s], color=wc[wi], lw=1.8, marker='^', ms=4, ls='-',
                    label=f'circ $\\Theta$(w={w}) + free $W$')
            ax.plot(kd, L_cc[w][s], color=wc[wi], lw=1.2, marker='s', ms=3, ls=':',
                    alpha=.8, label=f'circ $\\Theta$(w={w}) + circ $W$')
        ax.axhline(floors[s]['mmse_p0'], color='black', ls=':', lw=1.2, label='MMSE($p_0$)')
        ax.axhline(floors[s]['floor_free'], color='dimgray', ls='--', lw=1.2,
                   label='floor$_{\\mathrm{free}}$ (equiv+bias)')
        ax.set_xscale('log'); ax.set_xlabel('k / d'); ax.set_ylabel('MSE')
        ax.set_title(f'σ={s}'); ax.grid(True, alpha=.3)
        if col == nS - 1:
            ax.legend(fontsize=6, loc='upper right')
        ax2 = axes[1, col]
        for wi, w in enumerate(W_BAND_LIST):
            g = np.array(L_cf[w][s]) - ld
            ax2.plot(kd, g, color=wc[wi], lw=1.8, marker='^', ms=4, label=f'w={w}')
            m = g < 0
            if m.any():
                ax2.scatter(kd[m], g[m], s=90, facecolors='none', edgecolors='red',
                            lw=1.8, zorder=5)
        ax2.axhline(0, color='k', lw=1.2, ls='--')
        ax2.set_xscale('log'); ax2.set_xlabel('k / d')
        ax2.set_ylabel('$L^{circ\\Theta,freeW} - L^{dense}$')
        ax2.set_title(f'σ={s}: red ring = circulant features beat dense'); ax2.grid(True, alpha=.3)
        if col == nS - 1:
            ax2.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/rf_circulant_readout_ablation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
