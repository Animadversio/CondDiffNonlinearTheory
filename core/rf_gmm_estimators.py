"""
core/rf_gmm_estimators.py

Sampling-based estimators for the random-feature (RF) denoiser on a FINITE dataset
of N clean samples treated as the target distribution. Shared by
scripts/rf_gmm_finite_sample.py (canonical driver).

Estimator groups
----------------
Oracles for the empirical (finite-atom) distribution p0_emp = (1/N) sum_i delta(x0_i):
    mmse_nw        : unconditional Nadaraya-Watson Bayes MMSE (kernel posterior mean)
    mmse_nw_cond   : class-conditional NW Bayes MMSE (same-class kernel pool)

Linear baseline:
    wiener_emp     : unconditional linear Wiener from the empirical Sigma_p0

RF denoiser (fit an affine head W phi(y)+b on ReLU RF features of noisy y):
    rf_fixedridge_mmse : fixed-lambda ridge; eval on fresh noise ('test', double
                         descent) or in-sample ('train'). For the peak study.
    rf_optridge_mmse   : lambda chosen on held-out fresh noise -> the MINIMAL MSE
                         (removes the double-descent peak). Headline estimator.

RF theory (non-Gaussian, sample-based):
    stein_finiteN_mmse : empirical-Stein Cov + Hermite (n<=3) Sigma_phi from samples.

The jointly-Gaussian closed form (population OR empirical moments) lives in
core.gmm.mmse_theory_joint_gaussian:
    jg_pop      = mmse_theory_joint_gaussian(..., population moments)   [N-independent, inf-N limit]
    jg_finiteN  = mmse_theory_joint_gaussian(..., empirical moments)    [N-dependent]

All MSE values follow the "sum over d, mean over N" (trace) convention.
"""

import numpy as np
from scipy.stats import norm as scipy_norm

from .gmm import _c0   # ReLU Gaussian-smoothed activation, for the empirical-Stein Cov


# ---------------------------------------------------------------------------
# ReLU random-feature map
# ---------------------------------------------------------------------------

def relu_features(y, Theta, U=None, Gamma=None):
    """phi(y[,U]) = relu(Theta y [+ Gamma U]).  y:(N,d) Theta:(k,d) -> (N,k)."""
    pre = y @ Theta.T
    if U is not None and Gamma is not None:
        pre = pre + U @ Gamma.T
    return np.maximum(pre, 0.0)


# ---------------------------------------------------------------------------
# Oracles: Nadaraya-Watson Bayes MMSE for the empirical distribution
# ---------------------------------------------------------------------------

def mmse_nw(x0, sigma, n_noise=100, rng=None):
    """
    Unconditional NW Bayes MMSE for p0_emp. D_NW(y) = sum_j x_j K(y,x_j)/Z,
    K Gaussian with bandwidth sigma. This is the exact posterior mean E[x0|y]
    under the atom prior, i.e. the Bayes (minimum) MSE of the empirical dist.
    Evaluated at y_i = x0_i + sigma z_i (fresh z).  O(N^2 d n_noise).
    """
    if rng is None:
        rng = np.random.default_rng()
    N, d = x0.shape
    total = 0.0
    for _ in range(n_noise):
        z = rng.standard_normal((N, d))
        y = x0 + sigma * z
        diff = y[:, None, :] - x0[None, :, :]
        log_w = -np.einsum('ijk,ijk->ij', diff, diff) / (2 * sigma ** 2)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        total += float(np.sum((w @ x0 - x0) ** 2))
    return total / (n_noise * N)


def mmse_nw_cond(x0, labels, sigma, n_noise=100, rng=None):
    """
    Class-conditional NW Bayes MMSE for p0_emp when the class label is observed:
    the kernel pool is restricted to same-class atoms. This is the correct
    conditional oracle floor (<= unconditional mmse_nw) and the valid lower
    bound for the conditional RF denoiser.
    """
    if rng is None:
        rng = np.random.default_rng()
    N, d = x0.shape
    same = (labels[:, None] == labels[None, :])
    total = 0.0
    for _ in range(n_noise):
        z = rng.standard_normal((N, d))
        y = x0 + sigma * z
        diff = y[:, None, :] - x0[None, :, :]
        log_w = -np.einsum('ijk,ijk->ij', diff, diff) / (2 * sigma ** 2)
        log_w = np.where(same, log_w, -np.inf)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        total += float(np.sum((w @ x0 - x0) ** 2))
    return total / (n_noise * N)


# ---------------------------------------------------------------------------
# Linear (Wiener) baseline from the empirical covariance
# ---------------------------------------------------------------------------

def wiener_emp(x0, sigma):
    """Unconditional linear Wiener MMSE from empirical Sigma_p0: sum_i s^2 l_i/(l_i+s^2)."""
    N = x0.shape[0]
    Xc = x0 - x0.mean(0)
    ev = np.linalg.eigvalsh(Xc.T @ Xc / max(N - 1, 1))
    ev = np.maximum(ev, 0.0)
    return float((sigma ** 2 * ev / (ev + sigma ** 2)).sum())


def wiener_cond_emp(x0, labels, sigma):
    """
    CLASS-CONDITIONAL (per-branch) linear Wiener MMSE for the empirical distribution.
    The best LINEAR denoiser that knows the class: within each class c, apply the
    Wiener filter for that class's (empirical) covariance, then weight by class freq.
      L = sum_c (N_c/N) sum_i s^2 l_{c,i}/(l_{c,i}+s^2),   l_{c,i} = eig(Sigma_c^emp).
    For a Gaussian mixture this is the conditional linear (per-branch) Wiener.
    Nonzero eigenvalues come from the N_c x N_c Gram (zeros contribute 0 to the loss).
    """
    N = x0.shape[0]
    total = 0.0
    for c in np.unique(labels):
        Xc = x0[labels == c]
        Nc = Xc.shape[0]
        if Nc < 2:
            continue  # a single (or no) point has no within-class variance -> 0 loss
        Xcc = Xc - Xc.mean(0)
        ev = np.linalg.eigvalsh(Xcc @ Xcc.T / (Nc - 1))   # Gram -> nonzero eigenvalues
        ev = np.maximum(ev, 0.0)
        total += (Nc / N) * float((sigma ** 2 * ev / (ev + sigma ** 2)).sum())
    return float(total)


# ---------------------------------------------------------------------------
# Affine ridge on RF features: eigendecomposition-based lambda sweep
# ---------------------------------------------------------------------------

def _ridge_prepare(phi_tr_c, x0_c):
    """
    Eigendecompose once so many ridge lambdas are cheap.
    Returns a prep tuple; use with _ridge_risks. Primal (k x k) if N>=k else dual (N x N).
    phi_tr_c, x0_c must already be centered by the TRAIN means.
    """
    N, k = phi_tr_c.shape
    if N >= k:
        A = phi_tr_c.T @ phi_tr_c                      # (k, k)
        Lam, Q = np.linalg.eigh(A)
        QtB = Q.T @ (phi_tr_c.T @ x0_c)                # (k, d)
        return ('primal', Lam, Q, QtB, phi_tr_c)
    else:
        G = phi_tr_c @ phi_tr_c.T                      # (N, N)
        Lam, Q = np.linalg.eigh(G)
        Qtx = Q.T @ x0_c                               # (N, d)
        return ('dual', Lam, Q, Qtx, phi_tr_c)


def _ridge_risks(prep, phi_ev_c, x0_c, lam_values):
    """MSE (sum-d, mean-N) of the affine ridge fit for each lambda in lam_values."""
    mode, Lam, Q, mat, phi_tr_c = prep
    if mode == 'primal':
        Pe = phi_ev_c @ Q                              # (N, k)
        out = []
        for lam in lam_values:
            pred_c = (Pe * (1.0 / (Lam + lam))[None, :]) @ mat   # (N, d)
            out.append(float(np.mean(np.sum((pred_c - x0_c) ** 2, axis=1))))
        return np.array(out)
    else:
        MeQ = (phi_ev_c @ phi_tr_c.T) @ Q              # (N, N)
        out = []
        for lam in lam_values:
            pred_c = (MeQ * (1.0 / (Lam + lam))[None, :]) @ mat  # (N, d)
            out.append(float(np.mean(np.sum((pred_c - x0_c) ** 2, axis=1))))
        return np.array(out)


def rf_fixedridge_mmse(x0, U, Theta, Gamma, sigma, lam, n_noise, rng,
                       conditional=True, mode='test'):
    """
    Fixed-lambda affine RF ridge, fit on x0+train_noise.
      mode='test'  : evaluate on x0 + FRESH noise      (double-descent risk)
      mode='train' : evaluate in-sample (train error)  (monotone; the 'train' half of the split)
    Average over n_noise draws.
    """
    N, d = x0.shape
    x0_c = x0 - x0.mean(0)
    Uc = U if conditional else None
    Gc = Gamma if conditional else None
    total = 0.0
    for _ in range(n_noise):
        z_tr = rng.standard_normal((N, d))
        phi_tr = relu_features(x0 + sigma * z_tr, Theta, Uc, Gc)
        mu_phi = phi_tr.mean(0)
        phi_tr_c = phi_tr - mu_phi
        prep = _ridge_prepare(phi_tr_c, x0_c)
        if mode == 'train':
            phi_ev_c = phi_tr_c
        else:
            z_ev = rng.standard_normal((N, d))
            phi_ev_c = relu_features(x0 + sigma * z_ev, Theta, Uc, Gc) - mu_phi
        total += _ridge_risks(prep, phi_ev_c, x0_c, [lam])[0]
    return total / n_noise


def rf_optridge_mmse(x0, U, Theta, Gamma, sigma, n_noise, rng,
                     conditional=True, rel_grid=None, n_fit=1):
    """
    MINIMAL-MSE affine RF ridge: fit on x0+train_noise, choose lambda on an
    independent 'val' fresh-noise draw, report risk on a separate 'test' fresh
    draw. Marginalizes noise and optimizes regularization -> no double-descent peak.

    n_fit : number of independent noise realizations per clean sample STACKED into
            the fit (design has M = N * n_fit rows). This is the knob that closes
            the gap to the analytic RF-linear MMSE (stein_finiteN): as n_fit -> inf
            at FIXED N the empirical covariances -> their exact noise-marginalized
            values, the fitted head -> the population-optimal W*, and this estimate
            -> stein_finiteN (up to Hermite truncation). n_fit=1 = one noisy
            observation per clean sample (largest estimation-error gap ~ k/N).
    n_noise : independent (fit, val, test) repetitions to average the risk estimate.

    lambda grid = rel_grid * mean(train Gram eigenvalue) (scale-adaptive).
    Returns (mmse, median_selected_rel_lambda).
    """
    if rel_grid is None:
        # relative to the mean train-Gram eigenvalue; wide enough to reach the
        # near-ridgeless optimum (small k) up to heavy regularization (k ~ M).
        rel_grid = np.logspace(-8, 2, 30)
    N, d = x0.shape
    mu_x0 = x0.mean(0)
    x0_c = x0 - mu_x0                                    # eval target (N, d)
    Uc = U if conditional else None
    Gc = Gamma if conditional else None
    if n_fit > 1:
        x0_fit = np.repeat(x0, n_fit, axis=0)           # (M, d)
        U_fit = np.repeat(U, n_fit, axis=0) if conditional else U
    else:
        x0_fit, U_fit = x0, U
    x0_fit_c = x0_fit - mu_x0                            # (M, d), same global mean
    M = x0_fit.shape[0]
    total = 0.0
    sel = []
    for _ in range(n_noise):
        z_tr = rng.standard_normal((M, d))
        phi_tr = relu_features(x0_fit + sigma * z_tr, Theta, U_fit, Gc)
        mu_phi = phi_tr.mean(0)
        phi_tr_c = phi_tr - mu_phi
        prep = _ridge_prepare(phi_tr_c, x0_fit_c)
        lam_values = rel_grid * max(float(prep[1].mean()), 1e-12)
        z_val = rng.standard_normal((N, d))
        phi_val_c = relu_features(x0 + sigma * z_val, Theta, Uc, Gc) - mu_phi
        risks_val = _ridge_risks(prep, phi_val_c, x0_c, lam_values)
        j = int(np.argmin(risks_val))
        z_te = rng.standard_normal((N, d))
        phi_te_c = relu_features(x0 + sigma * z_te, Theta, Uc, Gc) - mu_phi
        total += _ridge_risks(prep, phi_te_c, x0_c, [lam_values[j]])[0]
        sel.append(rel_grid[j])
    return total / n_noise, float(np.median(sel))


# ---------------------------------------------------------------------------
# RF theory: empirical Stein Cov + Hermite Sigma_phi (non-Gaussian, sample-based)
# ---------------------------------------------------------------------------

def stein_covariances(x0, U, Theta, Gamma, sigma, lam, conditional=True):
    """
    Analytic (noise-marginalized) covariances for the empirical distribution:
    real samples for the data expectation, Hermite (n<=3) for the noise.
      Cov(x0,phi)_ij = (1/N) sum_n (x0_n - mu)_i c0(M_nj, s_j)
      Sigma_phi_ij   = Cov_{x0}(g_i,g_j) + E_{x0}[ sum_{n>=1} n! rho^n c_n_i c_n_j ]
    Returns (Cov (d,k), Sigma_phi (k,k) incl. lam*I, trace_p0).
    """
    N, d = x0.shape
    k = Theta.shape[0]
    mu = x0.mean(0)
    X0_c = x0 - mu
    s = sigma * np.linalg.norm(Theta, axis=1)                       # (k,)
    M = x0 @ Theta.T + (U @ Gamma.T if conditional else 0.0)        # (N, k)

    G = _c0(M, s[None, :])                                          # (N, k) = E_z[phi | x0]
    Cov = X0_c.T @ G / N                                            # (d, k)

    z = M / np.maximum(s[None, :], 1e-12)
    Phi_z = scipy_norm.cdf(z)
    phi_z = scipy_norm.pdf(z)
    G_c = G - G.mean(0)
    Sig_data = G_c.T @ G_c / N                                     # (k, k)

    C1 = s[None, :] * Phi_z
    C2 = s[None, :] * phi_z / 2.0
    C3 = M * phi_z / 6.0
    nn = np.linalg.norm(Theta, axis=1)
    Tn = Theta / nn[:, None]
    rho = np.clip(Tn @ Tn.T, -1 + 1e-6, 1 - 1e-6)                  # (k, k)
    Sig_noise = (rho * (C1.T @ C1 / N)
                 + 2.0 * rho ** 2 * (C2.T @ C2 / N)
                 + 6.0 * rho ** 3 * (C3.T @ C3 / N))
    Sig = Sig_data + Sig_noise + lam * np.eye(k)
    trace_p0 = float(np.sum(X0_c ** 2) / max(N - 1, 1))
    return Cov, Sig, trace_p0


def stein_finiteN_mmse(x0, U, Theta, Gamma, sigma, lam, conditional=True):
    """
    RF-linear MMSE for the empirical distribution (analytic; non-Gaussian).
    L = Tr(Σ_p0) - Tr(Cov Σ_φ^{-1} Cov^T).  Exact up to Hermite truncation.
    """
    Cov, Sig, trace_p0 = stein_covariances(x0, U, Theta, Gamma, sigma, lam, conditional)
    try:
        A = np.linalg.solve(Sig, Cov.T)
        expl = float(np.trace(Cov @ A))
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(Sig, Cov.T, rcond=None)[0]
        expl = float(np.trace(Cov @ A))
    return max(0.0, trace_p0 - expl)


def rf_fit_analytic_risk(x0, U, Theta, Gamma, sigma, n_fit, rng, conditional=True,
                         lam_eval=1e-4, rel_grid=None, n_reps=1):
    """
    STABLE convergence estimator (no evaluation-noise wobble, no interpolation blowup).

    Fit the RF head W_hat(lambda) = Cov_hat (Sigma_hat + lambda I)^{-1} empirically on
    n_fit stacked noisy draws per sample, then evaluate its risk ANALYTICALLY against
    the Stein noise-marginalized covariances (Cov, Sigma_phi) and pick lambda that
    minimizes that analytic risk:

        R(W) = Tr(Σ_p0) - 2 Tr(W Cov^T) + Tr(W Σ_φ W^T)
             = L_stein + Tr((W - W*) Σ_φ (W - W*)^T)  >=  L_stein
        estimate = min_lambda R(W_hat(lambda))

    Both the evaluation and the lambda-selection are analytic, so this removes the two
    wobble sources of the pure-MC opt-λ estimator (fresh-noise eval variance and
    val-set selection jitter). The min over lambda regularizes W_hat, so there is no
    blowup at the interpolation threshold (N*n_fit ~ k). Properties:
      (i)  >= L_stein always (PSD excess),
      (ii) monotone-decreasing to L_stein as n_fit -> inf,
      (iii) variance ~ O(1/(N*n_fit)) from the fit only.
    Averages over n_reps fits. Returns (R_mean, L_stein).
    """
    if rel_grid is None:
        rel_grid = np.logspace(-6, 2, 20)
    Cov, Sig_phi, trace_p0 = stein_covariances(x0, U, Theta, Gamma, sigma, lam_eval, conditional)
    try:
        L_stein = trace_p0 - float(np.trace(Cov @ np.linalg.solve(Sig_phi, Cov.T)))
    except np.linalg.LinAlgError:
        L_stein = trace_p0 - float(np.trace(Cov @ np.linalg.lstsq(Sig_phi, Cov.T, rcond=None)[0]))

    N, d = x0.shape
    k = Theta.shape[0]
    mu_x0 = x0.mean(0)
    Gc = Gamma if conditional else None
    if n_fit > 1:
        x0_fit = np.repeat(x0, n_fit, axis=0)
        U_fit = np.repeat(U, n_fit, axis=0) if conditional else U
    else:
        x0_fit, U_fit = x0, U
    x0_fit_c = x0_fit - mu_x0
    Mrows = x0_fit.shape[0]

    total = 0.0
    for _ in range(n_reps):
        z = rng.standard_normal((Mrows, d))
        phi = relu_features(x0_fit + sigma * z, Theta, U_fit, Gc)
        phi_c = phi - phi.mean(0)
        Cov_hat = x0_fit_c.T @ phi_c / Mrows                       # (d, k)
        Sig_hat = phi_c.T @ phi_c / Mrows                          # (k, k)
        Lam, Q = np.linalg.eigh(Sig_hat)                           # (k,), (k, k)
        CQ = Cov_hat @ Q                                           # (d, k)
        lam_values = rel_grid * max(float(Lam.mean()), 1e-12)
        best = np.inf
        for lam in lam_values:
            W = (CQ * (1.0 / (Lam + lam))[None, :]) @ Q.T          # (d, k) ridge slope
            R = (trace_p0 - 2.0 * np.trace(W @ Cov.T)
                 + np.trace(W @ Sig_phi @ W.T))
            if R < best:
                best = R
        total += float(best)
    return total / n_reps, max(0.0, L_stein)
