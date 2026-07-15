"""
rf_gmm_finite_sample.py  —  CANONICAL RF-denoiser finite-sample sweep.

Merges the former rf_gmm_overfit_sweep.py and rf_gmm_finite_ntrain.py.

For each N_train, treat the N_train GMM samples as the target distribution and
sweep the random-feature width k (and sigma). Compare the RF denoiser's minimal
MSE against theory (jointly-Gaussian at finite-N and its N->inf limit; and the
non-Gaussian empirical-Stein theory) and against the finite-dataset oracles.

Curves (see core/rf_gmm_estimators.py + core/gmm.py)
---------------------------------------------------
RF empirical (k-dependent), uncond + cond:
    rf_analytic      : PREFERRED — fit the head empirically, evaluate its risk
                       ANALYTICALLY vs the Stein covariances, pick lambda by the
                       analytic risk. >= Stein, monotone, ~zero eval variance.  [headline]
    rf_optridge      : pure-MC opt-lambda (fit + fresh-noise eval). Kept for
                       comparison; wobbles and can dip below Stein (double-MC).
    rf_fixed_test    : fixed-lambda, fresh-noise eval (double-descent).   [FIXED_RIDGE=1]
    rf_fixed_train   : fixed-lambda, in-sample eval (train error).        [FIXED_RIDGE=1]

RF theory (k-dependent), uncond + cond:
    jg_pop           : jointly-Gaussian §3, POPULATION moments  (N-independent, inf-N limit)
    jg_finiteN       : jointly-Gaussian §3, EMPIRICAL  moments  (N-dependent)
    stein_finiteN    : empirical-Stein (non-Gaussian, real samples + Hermite noise)

Oracles / baselines (k-independent horizontal lines):
    nw_bayes         : NW Bayes MMSE of the empirical dist (uncond)
    nw_bayes_cond    : class-conditional NW Bayes MMSE      (cond)
    wiener_emp       : empirical linear Wiener (uncond)
    wiener_pop       : population linear Wiener (uncond)     |  cond_wiener_pop (cond)
    bayes_pop        : population nonlinear Bayes MMSE (uncond)

Usage
-----
    python scripts/rf_gmm_finite_sample.py
    FIXED_RIDGE=1 python scripts/rf_gmm_finite_sample.py      # add double-descent study
    K_MAX=8192 N_TRAIN_VALUES=64,1024 python scripts/rf_gmm_finite_sample.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.gmm import GaussianMixture, mmse_theory_joint_gaussian, mmse_theory_gmm_pop, precompute_gmm_pop
from core.rf_gmm_estimators import (
    mmse_nw, mmse_nw_cond, wiener_emp, wiener_cond_emp,
    rf_optridge_mmse, rf_fit_analytic_risk, rf_fixedridge_mmse, stein_finiteN_mmse,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D          = int(os.environ.get('D', '8'))
N_CLASSES  = 3
WEIGHTS    = [0.5, 0.3, 0.2]
SEED       = int(os.environ.get('SEED', '42'))
LAM        = float(os.environ.get('LAM', '1e-4'))          # fixed ridge (study only)
K_MAX      = int(os.environ.get('K_MAX', '4096'))
N_NOISE    = int(os.environ.get('N_NOISE', '20'))          # noise draws for RF empirical
N_NOISE_NW = int(os.environ.get('N_NOISE_NW', '100'))      # noise draws for NW oracles
NW_MAX_N   = int(os.environ.get('NW_MAX_N', '2000'))       # skip NW oracle above this N
N_MC_EXACT = int(os.environ.get('N_MC_EXACT', '200000'))   # MC for population Bayes MMSE
STUDY_FIXED_RIDGE = os.environ.get('FIXED_RIDGE', '0') == '1'
# noise draws stacked into the RF fit. The measured min-MSE sits ~k/(N*n_fit)
# ABOVE the analytic RF-MMSE (Stein) from finite-noise estimation error; larger
# n_fit closes that gap. 'adaptive' keeps N*n_fit ~ TARGET_M (cheap for large N).
N_FIT      = os.environ.get('N_FIT', 'adaptive')
TARGET_M   = int(os.environ.get('TARGET_M', '4096'))

N_TRAIN_VALUES = [int(x) for x in
    os.environ.get('N_TRAIN_VALUES', '8,64,128,256,1024,50000').split(',')]
SIGMA_VALUES   = [float(s) for s in
    os.environ.get('SIGMA_VALUES', '0.5,1.0,2.0,5.0').split(',')]
K_GRID = [2 ** i for i in range(3, 16) if 2 ** i <= K_MAX]

# ---- backend dispatch: numpy (CPU) vs torch/CUDA for the (k,k)-heavy estimators ----
# DEVICE=cuda routes stein / analytic / JG-theory to core.rf_gmm_estimators_torch
# (the (k,k) eigh/solve dominate at large k). Cheap estimators stay on numpy.
DEVICE = os.environ.get('DEVICE', 'cpu')
if DEVICE == 'cuda':
    import torch
    from core.rf_gmm_estimators_torch import (
        stein_finiteN_mmse_t, rf_fit_analytic_risk_t, mmse_theory_joint_gaussian_t,
        mmse_theory_gmm_pop_t)
    _DT = torch.float32 if os.environ.get('TORCH_DTYPE', 'float64') == 'float32' else torch.float64
    def _gmm_pop(gmm, Th, Ga, s, cond, precomp=None):   # precomp ignored on CUDA (computed in torch)
        return mmse_theory_gmm_pop_t(gmm, Th, Ga, s, lam=LAM, conditional=cond, device='cuda', dtype=_DT)
    def _jg(Sig, mu, Th, Ga, s, CxU, SU, muU):
        return mmse_theory_joint_gaussian_t(Sig, mu, Th, Ga, s, C_xU=CxU, Sigma_U=SU, mu_U=muU,
                                            lam=LAM, device='cuda', dtype=_DT)
    def _stein(x0, U, Th, Ga, s, cond):
        return stein_finiteN_mmse_t(x0, U, Th, Ga, s, LAM, conditional=cond, device='cuda', dtype=_DT)
    def _analytic(x0, U, Th, Ga, s, nf, seed, cond):
        return rf_fit_analytic_risk_t(x0, U, Th, Ga, s, nf, conditional=cond, lam_eval=LAM,
                                      n_reps=2, device='cuda', dtype=_DT, seed=seed)[0]
else:
    def _gmm_pop(gmm, Th, Ga, s, cond, precomp=None):
        return mmse_theory_gmm_pop(gmm, Th, Ga, s, lam=LAM, conditional=cond, _precomp=precomp)
    def _jg(Sig, mu, Th, Ga, s, CxU, SU, muU):
        return mmse_theory_joint_gaussian(Sig, mu, Th, Ga, s, C_xU=CxU, Sigma_U=SU, mu_U=muU, lam=LAM)
    def _stein(x0, U, Th, Ga, s, cond):
        return stein_finiteN_mmse(x0, U, Th, Ga, s, LAM, conditional=cond)
    def _analytic(x0, U, Th, Ga, s, nf, seed, cond):
        return rf_fit_analytic_risk(x0, U, Th, Ga, s, nf, np.random.default_rng(seed),
                                    conditional=cond, lam_eval=LAM, n_reps=2)[0]


# ---------------------------------------------------------------------------
# Target GMM (same as the retired scripts)
# ---------------------------------------------------------------------------
def make_gmm(seed=SEED):
    rng = np.random.default_rng(seed)
    d = D
    means = np.zeros((N_CLASSES, d))
    means[0, 0] = 2.0
    means[1, 0] = -1.0; means[1, 1] = 1.5
    means[2, 0] = -1.0; means[2, 1] = -1.0; means[2, 2] = 1.2
    S0 = np.diag([1.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    S1 = np.diag([0.4, 1.0, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4])
    A = rng.standard_normal((d, d)) * 0.3
    S2 = A @ A.T + 0.5 * np.eye(d)
    return GaussianMixture(weights=np.array(WEIGHTS), means=means,
                           covs=np.stack([S0, S1, S2]))


def n_noise_adaptive(N_train):
    """Reps for the SECONDARY pure-MC opt-λ comparison curve. Kept modest (the
    analytic estimator is preferred) — enough to show its wobble, not so many it
    dominates runtime."""
    return max(8, min(N_NOISE, 4000 // max(N_train, 1)))


def n_fit_for(N_train):
    """Noise draws stacked into the RF fit (design has N*n_fit rows). Closes the
    measured-vs-Stein gap (~k/(N*n_fit)) as n_fit grows, at FIXED N. 'adaptive'
    keeps N*n_fit ~ TARGET_M so small-N gets many draws, large-N gets n_fit=1."""
    if N_FIT == 'adaptive':
        return max(1, min(64, TARGET_M // max(N_train, 1)))
    return int(N_FIT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Config: d={D}, C={N_CLASSES}, K_GRID={K_GRID}, N_NOISE={N_NOISE}, "
          f"FIXED_RIDGE={STUDY_FIXED_RIDGE}")
    print(f"N_TRAIN_VALUES={N_TRAIN_VALUES}  SIGMA_VALUES={SIGMA_VALUES}")

    gmm = make_gmm(SEED)
    trace_p0_pop = float(np.trace(gmm.Sigma))
    print(f"GMM Tr(Sigma_p0)={trace_p0_pop:.4f}")

    # Shared random projections (same Theta/Gamma across all N_train for fairness)
    rng_proj = np.random.default_rng(SEED + 100)
    Theta_cache, Gamma_cache = {}, {}
    for k in K_GRID:
        Theta_cache[k] = rng_proj.standard_normal((k, D)) / np.sqrt(D)
        Gamma_cache[k] = rng_proj.standard_normal((k, N_CLASSES)) / np.sqrt(N_CLASSES)

    # Population baselines (N-independent reference lines)
    rng_mc = np.random.default_rng(SEED + 1)
    pop_base = {}
    for sigma in SIGMA_VALUES:
        pop_base[sigma] = {
            'wiener_pop':      gmm.mmse_uncond_wiener(sigma),
            'cond_wiener_pop': gmm.mmse_cond_wiener(sigma),
            'bayes_pop':       gmm.mmse_uncond_exact(sigma, N_mc=N_MC_EXACT, rng=rng_mc),
        }

    all_results = {}
    for N_train in N_TRAIN_VALUES:
        print(f"\n=== N_train={N_train} ===")
        rng = np.random.default_rng(SEED + N_train)
        x0_tr, labels_tr, U_tr = gmm.sample(N_train, rng=rng)

        # Empirical moments (for jg_finiteN and baselines). /N (empirical-distribution
        # convention) so these are consistent with the /N feature covariances in
        # stein / wiener / analytic — mixing /(N-1) here injects a spurious ~Tr(Σp0)/N
        # offset into the MMSE (catastrophic at small N + low σ).
        mu_x0 = x0_tr.mean(0); mu_U = U_tr.mean(0)
        den = max(N_train, 1)
        X0_c = x0_tr - mu_x0; U_c = U_tr - mu_U
        Sig_p0_emp = X0_c.T @ X0_c / den
        C_xU_emp   = X0_c.T @ U_c / den
        Sig_U_emp  = U_c.T @ U_c / den
        trace_p0_emp = float(np.trace(Sig_p0_emp))
        print(f"  Tr(Sigma_emp)={trace_p0_emp:.4f}  (pop {trace_p0_pop:.4f})")

        # k-independent empirical baselines
        emp_base = {}
        for sigma in SIGMA_VALUES:
            b = {'wiener_emp': wiener_emp(x0_tr, sigma),
                 'wiener_cond_emp': wiener_cond_emp(x0_tr, labels_tr, sigma),
                 'nw_bayes': None, 'nw_bayes_cond': None}
            if N_train <= NW_MAX_N:
                b['nw_bayes']      = mmse_nw(x0_tr, sigma, N_NOISE_NW,
                                             np.random.default_rng(SEED + N_train + 777))
                b['nw_bayes_cond'] = mmse_nw_cond(x0_tr, labels_tr, sigma, N_NOISE_NW,
                                                  np.random.default_rng(SEED + N_train + 778))
            emp_base[sigma] = b

        # per-(sigma) result arrays over k
        keys = ['gmm_pop_u', 'gmm_pop_c', 'jg_finiteN_u', 'jg_finiteN_c',
                'stein_u', 'stein_c',
                'rf_analytic_u', 'rf_analytic_c',      # preferred: stable analytic-eval
                'rf_optridge_u', 'rf_optridge_c']      # kept: pure-MC opt-λ (double-MC, wobbly)
        if STUDY_FIXED_RIDGE:
            keys += ['rf_fixed_test_u', 'rf_fixed_test_c',
                     'rf_fixed_train_u', 'rf_fixed_train_c']
        res = {sg: {kk: [] for kk in keys} for sg in SIGMA_VALUES}

        nne = n_noise_adaptive(N_train)
        nf_train = n_fit_for(N_train)
        print(f"  n_noise={nne}, n_fit={nf_train}  (stacked fit design ~ {N_train * nf_train} rows)")
        for k in tqdm(K_GRID, desc=f'k (N={N_train})'):
            Theta, Gamma = Theta_cache[k], Gamma_cache[k]
            Zg = np.zeros_like(Gamma)
            # Precompute sigma-independent (k,k) matrices for gmm_pop (CPU path only;
            # CUDA path recomputes in torch but that's fast on GPU).
            gmm_precomp = precompute_gmm_pop(gmm, Theta) if DEVICE == 'cpu' else None
            for sigma in SIGMA_VALUES:
                base_seed = SEED + N_train + k + int(sigma * 1000)
                # --- GMM per-component population theory (correct, N→∞ limit) ---
                res[sigma]['gmm_pop_u'].append(_gmm_pop(gmm, Theta, Zg, sigma, False, gmm_precomp))
                res[sigma]['gmm_pop_c'].append(_gmm_pop(gmm, Theta, Gamma, sigma, True, gmm_precomp))
                # --- JG theory: empirical moments (finite-N) ---
                res[sigma]['jg_finiteN_u'].append(_jg(Sig_p0_emp, mu_x0, Theta, Zg, sigma, None, None, None))
                res[sigma]['jg_finiteN_c'].append(_jg(Sig_p0_emp, mu_x0, Theta, Gamma, sigma,
                                                      C_xU_emp, Sig_U_emp, mu_U))
                # --- Stein (non-Gaussian, empirical) ---
                res[sigma]['stein_u'].append(_stein(x0_tr, U_tr, Theta, Gamma, sigma, False))
                res[sigma]['stein_c'].append(_stein(x0_tr, U_tr, Theta, Gamma, sigma, True))
                # --- RF empirical (PREFERRED): analytic-eval of the empirical fit ---
                # Fit W_hat on stacked noisy features, evaluate its risk analytically
                # against the Stein covariances (>= Stein, monotone, ~zero eval variance).
                res[sigma]['rf_analytic_u'].append(_analytic(x0_tr, U_tr, Theta, Gamma, sigma,
                                                             nf_train, base_seed + 3, False))
                res[sigma]['rf_analytic_c'].append(_analytic(x0_tr, U_tr, Theta, Gamma, sigma,
                                                             nf_train, base_seed + 5, True))
                # --- RF empirical (kept for comparison): pure-MC optimal ridge ---
                ru, _ = rf_optridge_mmse(x0_tr, U_tr, Theta, Gamma, sigma, nne,
                                         np.random.default_rng(SEED + N_train + k + int(sigma * 1000)),
                                         conditional=False, n_fit=nf_train)
                rc, _ = rf_optridge_mmse(x0_tr, U_tr, Theta, Gamma, sigma, nne,
                                         np.random.default_rng(SEED + N_train + k + int(sigma * 1000) + 7),
                                         conditional=True, n_fit=nf_train)
                res[sigma]['rf_optridge_u'].append(ru)
                res[sigma]['rf_optridge_c'].append(rc)
                # --- RF empirical: fixed-ridge double-descent study (opt-in) ---
                if STUDY_FIXED_RIDGE:
                    rng_f = np.random.default_rng(SEED + N_train + k + int(sigma * 1000) + 11)
                    for cond, uc in [(False, 'u'), (True, 'c')]:
                        res[sigma][f'rf_fixed_test_{uc}'].append(rf_fixedridge_mmse(
                            x0_tr, U_tr, Theta, Gamma, sigma, LAM, nne, rng_f, conditional=cond, mode='test'))
                        res[sigma][f'rf_fixed_train_{uc}'].append(rf_fixedridge_mmse(
                            x0_tr, U_tr, Theta, Gamma, sigma, LAM, nne, rng_f, conditional=cond, mode='train'))

        all_results[N_train] = dict(res=res, emp_base=emp_base, trace_p0_emp=trace_p0_emp)
        _save_table(N_train, res, emp_base, pop_base, trace_p0_emp)
        _plot(N_train, res, emp_base, pop_base, trace_p0_emp)
    print("\nDone.")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _save_table(N_train, res, emp_base, pop_base, trace_p0_emp):
    os.makedirs('tables', exist_ok=True)
    sd = {'k_grid': np.array(K_GRID), 'sigma_values': np.array(SIGMA_VALUES),
          'N_train': N_train, 'trace_p0_emp': trace_p0_emp}
    for sg in SIGMA_VALUES:
        for kk, vals in res[sg].items():
            sd[f'{kk}_s{sg}'] = np.array(vals)
        for bk, bv in emp_base[sg].items():
            if bv is not None:
                sd[f'{bk}_s{sg}'] = bv
        for bk, bv in pop_base[sg].items():
            sd[f'{bk}_s{sg}'] = bv
    np.savez(f'tables/rf_gmm_finite_sample_N{N_train}.npz', **sd)
    print(f"  Saved tables/rf_gmm_finite_sample_N{N_train}.npz")


# ---------------------------------------------------------------------------
# Plot: 2 rows (uncond/cond) x len(sigma) cols, one figure per N_train
# ---------------------------------------------------------------------------
def _plot(N_train, res, emp_base, pop_base, trace_p0_emp):
    kd = np.array(K_GRID) / D
    nS = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, nS, figsize=(5 * nS, 8.5))
    axes = np.asarray(axes).reshape(2, nS)   # robust when nS == 1
    tag = " (+ fixed-ridge study)" if STUDY_FIXED_RIDGE else ""
    fig.suptitle(
        f'RF Denoiser — finite dataset N_train={N_train}{tag}\n'
        f'd={D}, C={N_CLASSES}, Tr(Σ_emp)={trace_p0_emp:.3f}   '
        f'[GMM pop = per-component Stein/Hermite, N→∞; JG fin-N = JG approx, emp moments]', fontsize=10)

    # per row: (title, u/c, NW-Bayes key, pop-Wiener key, EMPIRICAL-linear-Wiener key)
    # The linear baseline is row-matched: uncond -> unconditional Wiener;
    # cond -> conditional (per-branch) Wiener, so the nonlinear-gain zone in the
    # cond row isolates the NONLINEAR conditioning gain (conditioning already in it).
    rows = [('Unconditional', 'u', 'nw_bayes',      'wiener_pop',      'wiener_emp'),
            ('Conditional (U=class)', 'c', 'nw_bayes_cond', 'cond_wiener_pop', 'wiener_cond_emp')]
    for r, (title, uc, nw_key, popw_key, linemp_key) in enumerate(rows):
        for col, sigma in enumerate(SIGMA_VALUES):
            ax = axes[r, col]; R = res[sigma]; eb = emp_base[sigma]; pb = pop_base[sigma]
            lin_emp = eb[linemp_key]   # row-matched linear baseline

            # nonlinear-gain zone: band between the (row-matched) linear Wiener (top)
            # and the (row-matched) Bayes floor (bottom). RF dipping in = beating the
            # best LINEAR denoiser of the same conditioning, heading toward Bayes.
            if eb[nw_key] is not None and lin_emp > eb[nw_key]:
                ax.axhspan(eb[nw_key], lin_emp, color='green', alpha=0.06, zorder=0,
                           label=('nonlinear-gain zone' if (r == 0 and col == nS - 1) else None))

            # population references (gray)
            ax.axhline(pb['bayes_pop'], color='gray', ls='--', lw=1.1, alpha=.7, label='Bayes MMSE (pop)')
            ax.axhline(pb[popw_key],    color='gray', ls=':',  lw=1.1, alpha=.7,
                       label=('Wiener (pop)' if r == 0 else 'cond Wiener (pop)'))
            # empirical linear baseline (row-matched)
            ax.axhline(lin_emp, color='darkorange', ls='-', lw=1.5,
                       label=('linear Wiener (emp)' if r == 0 else 'cond linear Wiener (emp)'))
            if r == 1:   # faint uncond linear Wiener for reference in the cond row
                ax.axhline(eb['wiener_emp'], color='darkorange', ls=':', lw=1.0, alpha=.5,
                           label='uncond linear Wiener (ref)')
            if eb[nw_key] is not None:
                ax.axhline(eb[nw_key], color='forestgreen', ls='-', lw=1.8,
                           label=('NW Bayes (emp)' if r == 0 else 'NW Bayes cond (emp)'))

            # theory curves (k-dependent)
            ax.plot(kd, R[f'gmm_pop_{uc}'],    color='crimson',  lw=2,   ls='--', label='GMM theory (per-comp, N→∞)')
            ax.plot(kd, R[f'jg_finiteN_{uc}'], color='purple',   lw=1.8, ls='-',  label='JG approx (emp moments, finite-N)')
            ax.plot(kd, R[f'stein_{uc}'],      color='teal',     lw=1.8, ls='-.', label='Stein (non-Gaussian, emp)')

            # RF empirical (PREFERRED, headline): stable analytic-eval estimate
            ax.plot(kd, R[f'rf_analytic_{uc}'], color='steelblue', lw=2.4, marker='o', ms=4,
                    label='RF empirical (analytic-eval, preferred)')
            # RF empirical: pure-MC opt-λ, kept for comparison (wobbly, can dip < Stein)
            ax.plot(kd, R[f'rf_optridge_{uc}'], color='slategray', lw=1.1, ls=':', marker='.', ms=3,
                    alpha=0.8, label='RF empirical (pure-MC opt-λ)')
            # RF empirical: fixed-ridge study (opt-in)
            if STUDY_FIXED_RIDGE:
                ax.plot(kd, R[f'rf_fixed_test_{uc}'],  color='slategray', lw=1.2, ls='-',
                        label=f'RF fixed-λ test (λ={LAM:g})')
                ax.plot(kd, R[f'rf_fixed_train_{uc}'], color='slategray', lw=1.0, ls=':',
                        label='RF fixed-λ train')

            ax.set_xscale('log'); ax.set_xlabel('k / d'); ax.set_ylabel('MSE')
            ax.set_title(f'{title}  σ={sigma}'); ax.grid(True, alpha=.3)
            ax.set_ylim(bottom=-0.01)
            if r == 0 and col == nS - 1:
                ax.legend(fontsize=6.5, loc='upper right')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    path = f'figures/rf_gmm_finite_sample_N{N_train}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved {path}")


if __name__ == '__main__':
    main()
