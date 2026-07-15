# Methods: RF Denoiser on Finite Training Sets

Script: `scripts/rf_gmm_finite_ntrain.py`
Theory function: `core/gmm.mmse_theory_joint_gaussian`
Reference: `docs/newfile2_article.pdf` Section 3 (Jointly Gaussian x0, U)

---

## Setup

For each N_train in {8, 64, 128, 256, 1024, 50000}:

- Sample fixed training set: `(x0_train, U_train)` with `x0_train` shape `(N_train, d)`, `U_train` shape `(N_train, C)` one-hot
- Compute empirical second-order statistics from the N_train samples:

```
mu_x0    = (1/N) sum_i x0_i                          (d,)
mu_U     = (1/N) sum_i U_i                            (C,)
Sigma_p0 = (1/(N-1)) X0_c^T X0_c                     (d, d)
C_xU     = (1/(N-1)) X0_c^T U_c                      (d, C)   [cross-covariance]
Sigma_U  = (1/(N-1)) U_c^T U_c                        (C, C)
```

where `X0_c = x0_train - mu_x0`, `U_c = U_train - mu_U`.

Random feature projections `Theta` (k, d) and `Gamma` (k, C) are shared across all N_train values.

---

## Method 1: RF Empirical Closed-Form (CF)

For N_noise independent noise draws `Z ~ N(0, sigma^2 I)`:

```
Y = x0_train + Z                             (N, d)
phi_uncond = relu(Y @ Theta^T)               (N, k)
phi_cond   = relu(Y @ Theta^T + U @ Gamma^T) (N, k)
```

Accumulate empirical covariances:
```
Cov_emp   = (1/N) X0_c^T phi    (d, k)
Sigma_phi = (1/N) phi^T phi - mu_phi mu_phi^T + lam I    (k, k)
```

MMSE = Tr(Sigma_p0) - Tr(Cov_emp @ Sigma_phi^{-1} @ Cov_emp^T)

This is the **empirical closed-form** — no analytical formula, directly from data.

---

## Method 2: RF Theory — Jointly Gaussian (x0, U) [Section 3]

Uses ONLY the second-order empirical statistics above (no individual GMM component knowledge).
Assumes (x0, U) is jointly Gaussian with the estimated moments.

### Conditional case (with U):

**Pre-activation mean and variance** for each feature j:
```
m_tilde_j = theta_j^T mu_x0 + gamma_j^T mu_U

S_tilde_j^2 = theta_j^T Sigma_p0 theta_j
             + 2 theta_j^T C_xU gamma_j
             + gamma_j^T Sigma_U gamma_j
             + sigma^2 ||theta_j||^2
```

**Stein coefficient:**
```
alpha_j = c1(m_tilde_j, S_tilde_j) / S_tilde_j

where c1(m, s) = s * Phi(m/s)    [Phi = standard normal CDF]
```

**Covariance Cov(x0, phi^U):**
```
Cov(x0, phi^U)_{:,j} = (Sigma_p0 theta_j + C_xU gamma_j) * alpha_j

Matrix form:  Cov = (Sigma_p0 @ Theta^T + C_xU @ Gamma^T) * alpha[None, :]   (d, k)
```

**Joint correlation r_tilde_ij:**
```
Numerator_ij = theta_i^T Sigma_p0 theta_j
             + theta_i^T C_xU gamma_j
             + gamma_i^T C_Ux theta_j
             + gamma_i^T Sigma_U gamma_j
             + sigma^2 theta_i^T theta_j

r_tilde_ij = Numerator_ij / (S_tilde_i * S_tilde_j)

Matrix form:  Num = Theta Sigma_p0 Theta^T + sigma^2 Theta Theta^T
                  + Theta C_xU Gamma^T + (Theta C_xU Gamma^T)^T
                  + Gamma Sigma_U Gamma^T
              r_tilde = Num / outer(S_tilde, S_tilde)
```

**Feature covariance via Hermite series (n=1,2,3):**
```
Sigma^U_{phi,ij} = sum_{n>=1} n! * r_tilde_ij^n * c_n(m_tilde_i, S_tilde_i) * c_n(m_tilde_j, S_tilde_j)

where:
  c1(m, s) = s Phi(m/s)
  c2(m, s) = s phi(m/s) / 2          [phi = standard normal PDF]
  c3(m, s) = m phi(m/s) / 6

Matrix form:  Sigma_phi = 1! r_tilde * outer(C1,C1)
                        + 2! r_tilde^2 * outer(C2,C2)
                        + 3! r_tilde^3 * outer(C3,C3)
                        + lam * I
```

**MMSE:**
```
MMSE = Tr(Sigma_p0) - Tr(Cov @ Sigma_phi^{-1} @ Cov^T)
```

### Unconditional case:
Same formulas with `Gamma = 0`, `C_xU = None`, `mu_U = None`.
Reduces to:  `m_j = theta_j^T mu_x0`,  `S_j^2 = theta_j^T Sigma_p0 theta_j + sigma^2 ||theta_j||^2`,  `r_ij = (theta_i^T Sigma_p0 theta_j + sigma^2 theta_i^T theta_j) / (S_i S_j)`.

---

## Method 3: Nadaraya-Watson MMSE (Optimal for Empirical Distribution, N_train <= 1024)

For the empirical distribution (uniform mixture of N_train Dirac deltas at x0_i),
the Bayes-optimal denoiser at y is the Nadaraya-Watson kernel smoother:

```
D*(y) = sum_j x0_j * K(y, x0_j) / sum_j K(y, x0_j)

K(y, x) = exp(-||y - x||^2 / (2 sigma^2))    [Gaussian kernel]
```

MMSE evaluated on training points with fresh noise:
```
y_i = x0_i + sigma * z_i,   z_i ~ N(0, I)
NW-MMSE = (1/N) sum_i ||D*(y_i) - x0_i||^2
```
averaged over N_noise=30 noise draws.

**Interpretation:** This is the lowest MMSE achievable by any denoiser that is optimal for EXACTLY these N_train points (not the population GMM). It measures how well the training set can in principle be memorized.

For N_train=8 at sigma=0.5, NW-MMSE ≈ 0.027 — near-perfect memorization of 8 points.

---

## Method 4: Empirical Linear Wiener (Horizontal Baseline)

From eigenvalues of the empirical Sigma_p0:
```
MMSE_linear = sum_i sigma^2 * lambda_i / (lambda_i + sigma^2)
```
where `lambda_i = eigvalsh(Sigma_p0_emp)`.

---

## Method 5: GMM Population Bayes Optimal (Reference, Dotted Line)

The Bayes-optimal denoiser for the infinite GMM population, computed by Monte Carlo (N=200,000 samples). Shown only for reference — this is the target MMSE as N_train -> infinity.

---

## Known Issues / Questions

- **Does the jointly Gaussian approximation hold for one-hot U?** U is discrete (one-hot), so (x0, U) is NOT jointly Gaussian. The Section 3 formula is an approximation that uses only the first two moments. For large N_train this approximates the GMM Stein formula; for small N_train the empirical C_xU/Sigma_U may be poor estimates.

- **r_tilde clipping:** r_tilde is clipped to (-1+eps, 1-eps) to ensure valid Hermite expansion. Near-degenerate cases (very small S_tilde) may cause issues.

- **The Hermite series truncation at n=3** may underestimate Sigma_phi for highly nonlinear regimes (large sigma, small k).
