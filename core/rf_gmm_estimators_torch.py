"""
core/rf_gmm_estimators_torch.py

GPU/torch backend for the compute-heavy RF-denoiser estimators, for scaling to
large k where the (k,k) eigendecomposition / solves dominate. Mirrors the numpy
implementations in core.rf_gmm_estimators and core.gmm exactly (same math), but
runs the big linear algebra on a chosen device (default 'cuda').

Ported (the (k,k)-heavy ones):
    stein_covariances_t, stein_finiteN_mmse_t   (empirical-Stein, non-Gaussian)
    rf_fit_analytic_risk_t                       (preferred stable estimator)
    mmse_theory_joint_gaussian_t                 (JG closed form, pop or emp moments)

Cheap estimators (mmse_nw, wiener_emp, rf_optridge on small designs) stay on numpy.

dtype: float64 by default (parity with numpy). Pass dtype=torch.float32 for a large
extra GPU speedup when the ridge keeps conditioning benign — validate before trusting.
"""

import numpy as np
import torch

_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _to(x, device, dtype):
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _ndtr(z):        # standard normal CDF
    return torch.special.ndtr(z)


def _npdf(z):        # standard normal PDF
    return torch.exp(-0.5 * z * z) / _SQRT2PI


def _c0_t(M, s):     # Gaussian-smoothed ReLU: c0(M,s) = M Phi(z) + s phi(z), z=M/s
    z = M / torch.clamp(s, min=1e-12)
    return M * _ndtr(z) + s * _npdf(z)


def relu_features_t(y, Theta, U=None, Gamma=None):
    pre = y @ Theta.T
    if U is not None and Gamma is not None:
        pre = pre + U @ Gamma.T
    return torch.clamp(pre, min=0.0)


# ---------------------------------------------------------------------------
# Empirical-Stein covariances + MMSE
# ---------------------------------------------------------------------------

def stein_covariances_t(x0, U, Theta, Gamma, sigma, lam, conditional,
                        device='cuda', dtype=torch.float64):
    x0 = _to(x0, device, dtype); Theta = _to(Theta, device, dtype)
    Gamma = _to(Gamma, device, dtype); U = _to(U, device, dtype)
    N, d = x0.shape; k = Theta.shape[0]
    mu = x0.mean(0); X0_c = x0 - mu
    s = sigma * torch.linalg.norm(Theta, dim=1)                      # (k,)
    M = x0 @ Theta.T + (U @ Gamma.T if conditional else 0.0)         # (N, k)
    G = _c0_t(M, s[None, :])
    Cov = X0_c.T @ G / N
    z = M / torch.clamp(s[None, :], min=1e-12)
    Phi_z = _ndtr(z); phi_z = _npdf(z)
    G_c = G - G.mean(0)
    Sig_data = G_c.T @ G_c / N
    C1 = s[None, :] * Phi_z
    C2 = s[None, :] * phi_z / 2.0
    C3 = M * phi_z / 6.0
    nn = torch.linalg.norm(Theta, dim=1); Tn = Theta / nn[:, None]
    rho = torch.clamp(Tn @ Tn.T, -1 + 1e-6, 1 - 1e-6)
    Sig_noise = (rho * (C1.T @ C1 / N)
                 + 2.0 * rho ** 2 * (C2.T @ C2 / N)
                 + 6.0 * rho ** 3 * (C3.T @ C3 / N))
    Sig = Sig_data + Sig_noise + lam * torch.eye(k, device=device, dtype=dtype)
    # /N to match Cov / Sig normalization (see core.rf_gmm_estimators.stein_covariances);
    # /(N-1) here injects a spurious ~Tr(Σp0)/N offset, catastrophic at small N + low σ.
    trace_p0 = float((X0_c ** 2).sum() / max(N, 1))
    return Cov, Sig, trace_p0


def stein_finiteN_mmse_t(x0, U, Theta, Gamma, sigma, lam, conditional=True,
                         device='cuda', dtype=torch.float64):
    Cov, Sig, trace_p0 = stein_covariances_t(x0, U, Theta, Gamma, sigma, lam,
                                             conditional, device, dtype)
    expl = float(torch.trace(Cov @ torch.linalg.solve(Sig, Cov.T)))
    return max(0.0, trace_p0 - expl)


# ---------------------------------------------------------------------------
# Preferred stable estimator: empirical fit + analytic (Stein) evaluation
# ---------------------------------------------------------------------------

def rf_fit_analytic_risk_t(x0, U, Theta, Gamma, sigma, n_fit, conditional=True,
                           lam_eval=1e-4, rel_grid=None, n_reps=1,
                           device='cuda', dtype=torch.float64, seed=0):
    if rel_grid is None:
        rel_grid = np.logspace(-8, 2, 30)
    Cov, Sig_phi, trace_p0 = stein_covariances_t(x0, U, Theta, Gamma, sigma, lam_eval,
                                                 conditional, device, dtype)
    L_stein = trace_p0 - float(torch.trace(Cov @ torch.linalg.solve(Sig_phi, Cov.T)))

    x0t = _to(x0, device, dtype); Theta_t = _to(Theta, device, dtype)
    Gamma_t = _to(Gamma, device, dtype); Ut = _to(U, device, dtype)
    N, d = x0t.shape; k = Theta_t.shape[0]; mu_x0 = x0t.mean(0)
    Gc = Gamma_t if conditional else None
    if n_fit > 1:
        x0_fit = x0t.repeat_interleave(n_fit, dim=0)
        U_fit = Ut.repeat_interleave(n_fit, dim=0) if conditional else Ut
    else:
        x0_fit, U_fit = x0t, Ut
    x0_fit_c = x0_fit - mu_x0
    Mrows = x0_fit.shape[0]
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))

    total = 0.0
    for _ in range(n_reps):
        z = torch.randn(Mrows, d, device=device, dtype=dtype, generator=gen)
        phi = relu_features_t(x0_fit + sigma * z, Theta_t, U_fit, Gc)
        phi_c = phi - phi.mean(0)
        Cov_hat = x0_fit_c.T @ phi_c / Mrows
        Sig_hat = phi_c.T @ phi_c / Mrows
        Lam, Q = torch.linalg.eigh(Sig_hat)                          # the GPU win
        CQ = Cov_hat @ Q
        lam_values = rel_grid * max(float(Lam.mean()), 1e-12)
        SphiQ = Sig_phi @ Q                                          # reused across lambdas
        best = float('inf')
        for lam in lam_values:
            inv = (1.0 / (Lam + lam))
            W = (CQ * inv[None, :]) @ Q.T                            # (d, k)
            # R = trace_p0 - 2 Tr(W Cov^T) + Tr(W Sig_phi W^T)
            R = (trace_p0 - 2.0 * float((W * Cov).sum())
                 + float(((W @ Sig_phi) * W).sum()))
            if R < best:
                best = R
        total += best
    return total / n_reps, max(0.0, L_stein)


# ---------------------------------------------------------------------------
# Jointly-Gaussian closed form (population OR empirical moments)
# ---------------------------------------------------------------------------

def mmse_theory_joint_gaussian_t(Sigma_p0, mu_x0, Theta, Gamma, sigma,
                                 C_xU=None, Sigma_U=None, mu_U=None, lam=1e-4,
                                 n_terms=3, device='cuda', dtype=torch.float64):
    Sigma_p0 = _to(Sigma_p0, device, dtype); mu_x0 = _to(mu_x0, device, dtype)
    Theta = _to(Theta, device, dtype); Gamma = _to(Gamma, device, dtype)
    k = Theta.shape[0]
    trace_p0 = float(torch.trace(Sigma_p0))
    conditional = (C_xU is not None) and (mu_U is not None)
    if conditional:
        C_xU = _to(C_xU, device, dtype); Sigma_U = _to(Sigma_U, device, dtype)
        mu_U = _to(mu_U, device, dtype)

    M_tilde = Theta @ mu_x0
    if conditional:
        M_tilde = M_tilde + Gamma @ mu_U
    theta_norms_sq = (Theta ** 2).sum(1)
    ThSigTh = ((Theta @ Sigma_p0) * Theta).sum(1)
    S_sq = ThSigTh + sigma ** 2 * theta_norms_sq
    if conditional:
        ThCG = ((Theta @ C_xU) * Gamma).sum(1)
        GaSuG = ((Gamma @ Sigma_U) * Gamma).sum(1)
        S_sq = S_sq + 2.0 * ThCG + GaSuG
    S_tilde = torch.sqrt(torch.clamp(S_sq, min=1e-24))

    def _c1(m, s):
        z = m / torch.clamp(s, min=1e-12); return s * _ndtr(z)
    def _c2(m, s):
        z = m / torch.clamp(s, min=1e-12); return s * _npdf(z) / 2.0
    def _c3(m, s):
        z = m / torch.clamp(s, min=1e-12); return m * _npdf(z) / 6.0

    alpha = _c1(M_tilde, S_tilde) / torch.clamp(S_tilde, min=1e-12)
    Cov = Sigma_p0 @ Theta.T
    if conditional:
        Cov = Cov + C_xU @ Gamma.T
    Cov = Cov * alpha[None, :]

    Num = Theta @ Sigma_p0 @ Theta.T + sigma ** 2 * (Theta @ Theta.T)
    if conditional:
        ThCGam = Theta @ C_xU @ Gamma.T
        Num = Num + ThCGam + ThCGam.T + Gamma @ Sigma_U @ Gamma.T
    r_tilde = Num / (S_tilde[:, None] * S_tilde[None, :]).clamp(min=1e-24)
    r_tilde = torch.clamp(r_tilde, -1 + 1e-6, 1 - 1e-6)

    C1 = _c1(M_tilde, S_tilde); C2 = _c2(M_tilde, S_tilde)
    Sigma_phi = 1.0 * r_tilde * torch.outer(C1, C1) + 2.0 * r_tilde ** 2 * torch.outer(C2, C2)
    if n_terms >= 3:
        C3 = _c3(M_tilde, S_tilde)
        Sigma_phi = Sigma_phi + 6.0 * r_tilde ** 3 * torch.outer(C3, C3)
    Sigma_phi = Sigma_phi + lam * torch.eye(k, device=device, dtype=dtype)

    expl = float(torch.trace(Cov @ torch.linalg.solve(Sigma_phi, Cov.T)))
    return max(0.0, trace_p0 - expl)
