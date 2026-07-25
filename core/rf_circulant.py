"""
core/rf_circulant.py

Circulant-constrained random-feature (RF) denoiser — the L^circ estimator of
newfile5.tex, Sections 3-4.

Setup (writeup Sec 3).  k = c * d.  The feature projection Theta in R^{k x d} is
constrained to c blocks, each a d x d circulant matrix:

    Theta = [Theta_1; Theta_2; ...; Theta_c],   Theta_a = circ(h_a),  h_a ~ N(0, I/d)

so row (a, tau) is the cyclic shift  Theta_{(a,tau)} = S^tau h_a. Every entry is
marginally N(0, 1/d) — identical to the dense RF — only the JOINT law of the rows
differs (dependent within a block vs. i.i.d. dense). The readout W in R^{d x k} is
likewise block-circulant, W = [W_1 ... W_c], each W_a d x d circulant.

Because both Theta and W are block-circulant, the loss decouples per DFT frequency
(writeup eq. 3.?):

    L^circ = Tr(Sigma_p0) - sum_{f=1}^{d} q_f^H P_f^{-1} q_f
      (P_f)_{ab} = f_f^H Sigma^{(a,b)}_phi f_f ,   (q_f)_a = f_f^H Sigma_{phi_a, x0} f_f

where Sigma_phi (k x k) and Cov(x0, phi) (d x k) are the SAME noise-marginalised
Stein covariances used by the dense estimator (core.rf_gmm_estimators), f_f is the
f-th DFT basis column, and Sigma^{(a,b)}_phi / Sigma_{phi_a, x0} are the d x d
blocks of those covariances. This is the exact minimiser of the block-circulant-W
denoiser loss on block-circulant features, hence directly comparable to the dense
L = Tr(Sigma_p0) - Tr(Cov Sigma_phi^{-1} Cov^T).

Only k divisible by d is meaningful (a whole number of d x d blocks).
"""

import numpy as np

from .rf_gmm_estimators import stein_covariances


# ---------------------------------------------------------------------------
# Block-circulant projection
# ---------------------------------------------------------------------------

def build_circulant_theta(k, d, rng, w=None):
    """Block-circulant Theta (k=c*d rows, d cols). c blocks, each circ(h_a) with
    row (a,tau) = S^tau h_a = np.roll(h_a, tau).

    w : filter bandwidth (support width of the generating kernel h_a).
        w=None or w>=d -> full-width kernel h_a ~ N(0, I/d): every entry marginally
        N(0,1/d), matching the dense RF; only the row-joint differs. This is
        shift-equivariant but NOT local — all d coordinates are weighted equally.
        w<d -> LOCAL (banded) filter: h_a is supported on its first w entries with
        h_a[:w] ~ N(0, I/w), zeros elsewhere. This is the true conv-layer analogue:
        each feature reads a length-w window around one coordinate rather than the
        whole vector. Variance 1/w (not 1/d) keeps E||h_a||^2 = 1, so row norms —
        and hence the pre-activation scale — match the dense RF for every w.

    WARNING — w=1 is DEGENERATE. Then h_a = [g_a, 0, ...], so Theta_a = g_a * I (a
    scaled identity): the feature map does no spatial mixing at all. Worse, since
    relu(g_a * y_i) = |g_a| * relu(+-y_i), the feature span collapses to
    span{relu(y_i), relu(-y_i)} = span{y_i, |y_i|} — only 2 functions per coordinate
    NO MATTER HOW LARGE k IS. L^circ(w=1) therefore saturates at tiny k and is later
    overtaken by wider filters. Any small-k "win" at w=1 reflects fast saturation, not
    expressivity. Use w>=2 for meaningful locality results.
    """
    if k % d != 0:
        raise ValueError(f"k={k} not divisible by d={d}; circulant needs whole blocks")
    c = k // d
    ww = d if (w is None or w >= d) else int(w)
    if ww < 1:
        raise ValueError(f"bandwidth w={w} must be >= 1")
    Theta = np.zeros((k, d))
    for a in range(c):
        h = np.zeros(d)
        h[:ww] = rng.standard_normal(ww) / np.sqrt(ww)   # E||h||^2 = 1 for any w
        for tau in range(d):
            Theta[a * d + tau] = np.roll(h, tau)     # (S^tau h)_i = h_{i-tau mod d}
    return Theta


def build_circulant_gamma(k, d, n_classes, rng):
    """Block-circulant-compatible label bias: shared across the d rows of a block
    (shift-equivariant), i.e. Gamma[(a,tau), :] = gamma_a for all tau. Drawn
    gamma_a ~ N(0, I/n_classes) once per block. Returns (k, n_classes)."""
    if k % d != 0:
        raise ValueError(f"k={k} not divisible by d={d}")
    c = k // d
    Gamma = np.empty((k, n_classes))
    for a in range(c):
        g = rng.standard_normal(n_classes) / np.sqrt(n_classes)
        Gamma[a * d:(a + 1) * d] = g[None, :]
    return Gamma


# ---------------------------------------------------------------------------
# L^circ from Stein covariances
# ---------------------------------------------------------------------------

def _dft_matrix(d):
    """Unitary DFT F, F[j,l] = exp(-2 pi i j l / d)/sqrt(d) (symmetric, F F^H = I)."""
    j = np.arange(d)
    return np.exp(-2j * np.pi * np.outer(j, j) / d) / np.sqrt(d)


def circulant_rf_mmse(x0, U, Theta, Gamma, sigma, lam, conditional=True):
    """L^circ for the empirical distribution: block-circulant W on block-circulant
    features. Theta must already be block-circulant (build_circulant_theta). Uses the
    Stein noise-marginalised Cov(x0,phi) and Sigma_phi, then the per-frequency c x c
    solve. Returns Tr(Sigma_p0) - sum_f Re(q_f^H P_f^{-1} q_f)."""
    Cov, Sig, trace_p0 = stein_covariances(x0, U, Theta, Gamma, sigma, lam, conditional)
    d = x0.shape[1]
    k = Theta.shape[0]
    c = k // d
    F = _dft_matrix(d)
    Fc = F.conj()

    # P^{(a,b)}_f = f_f^H Sigma^{(a,b)}_phi f_f  ->  Pblocks (d, c, c)
    Sig4 = Sig.reshape(c, d, c, d)                       # [a,p,b,q] = Sigma^{(a,b)}_{pq}
    Pblocks = np.einsum('fp,apbq,fq->fab', F, Sig4, Fc, optimize=True)   # (d, c, c)

    # Sigma_{phi_a, x0} = Cov(phi_a, x0) = (Cov[:, block a])^T  ->  CovT4 (c, d, d)
    CovT4 = Cov.T.reshape(c, d, d)                       # [a,p,q] = Cov(phi_{a,p}, x0_q)
    Qblocks = np.einsum('fp,apq,fq->fa', F, CovT4, Fc, optimize=True)    # (d, c)

    expl = 0.0
    for f in range(d):
        Pf = Pblocks[f]                                  # (c, c) Hermitian PD (lam on diag)
        qf = Qblocks[f]                                  # (c,)
        try:
            sol = np.linalg.solve(Pf, qf)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(Pf, qf, rcond=None)[0]
        expl += float(np.real(np.vdot(qf, sol)))         # q^H P^{-1} q (real)
    return max(0.0, trace_p0 - expl)


# ---------------------------------------------------------------------------
# Self-test: frequency-domain L^circ == brute-force block-circulant-W optimum
# ---------------------------------------------------------------------------

def _circulant_from_row(w):
    """d x d circulant with first row w: C[tau] = np.roll(w, tau)."""
    d = w.shape[0]
    return np.stack([np.roll(w, tau) for tau in range(d)])


def _bruteforce_circ_loss(Cov, Sig, trace_p0, c, d):
    """Directly minimise L(W)=Tr(Sp0)-2Tr(W Cov^T)+Tr(W Sig W^T) over block-circulant
    W (k = c*d real DOF: the c first-rows). Quadratic in the DOF -> linear solve.
    Basis: e_{a,r} = block-circulant W whose block a has first row = unit vector r."""
    k = c * d
    # Build basis matrices B[m] (d,k) for m = a*d + r
    B = np.zeros((k, d, k))
    for a in range(c):
        for r in range(d):
            w = np.zeros(d); w[r] = 1.0
            Wa = _circulant_from_row(w)                  # (d,d)
            B[a * d + r][:, a * d:(a + 1) * d] = Wa
    # L(sum_m t_m B_m) = trace_p0 - 2 sum_m t_m Tr(B_m Cov^T)
    #                    + sum_{m,n} t_m t_n Tr(B_m Sig B_n^T)
    lin = np.array([np.trace(B[m] @ Cov.T) for m in range(k)])          # (k,)
    Hess = np.array([[np.trace(B[m] @ Sig @ B[n].T) for n in range(k)]
                     for m in range(k)])                                 # (k,k)
    t = np.linalg.solve(Hess, lin)                                       # minimiser
    return trace_p0 - float(lin @ t)


def _selftest():
    rng = np.random.default_rng(0)
    d, c = 4, 3
    k = c * d
    N = 200
    sigma, lam = 1.0, 1e-6
    x0 = rng.standard_normal((N, d)) * np.array([1.5, 1.0, 0.6, 0.3])
    labels = rng.integers(0, 2, N)
    U = np.eye(2)[labels]
    for cond in (False, True):
        Theta = build_circulant_theta(k, d, np.random.default_rng(1))
        Gamma = build_circulant_gamma(k, d, 2, np.random.default_rng(2)) if cond \
            else np.zeros((k, 2))
        Cov, Sig, trace_p0 = stein_covariances(x0, U, Theta, Gamma, sigma, lam, cond)
        freq = circulant_rf_mmse(x0, U, Theta, Gamma, sigma, lam, cond)
        brute = _bruteforce_circ_loss(Cov, Sig, trace_p0, c, d)
        print(f"cond={cond}: freq-domain L^circ={freq:.8f}  brute-force={brute:.8f}  "
              f"|diff|={abs(freq - brute):.2e}")
        assert abs(freq - brute) < 1e-6, "frequency-domain L^circ != brute-force optimum"
    print("rf_circulant self-test PASSED")


if __name__ == '__main__':
    _selftest()
