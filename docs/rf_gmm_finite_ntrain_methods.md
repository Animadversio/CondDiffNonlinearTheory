# Methods: RF Denoiser on Finite Training Sets

Script: `scripts/rf_gmm_finite_sample.py`
Theory functions: `core/gmm.mmse_theory_gmm_pop`, `core/rf_gmm_estimators_torch.mmse_theory_gmm_pop_t`
Reference: `docs/newfile2_article.pdf` Sections 2–3

---

## Setup

For each N_train in {16, 64, 256, 1024}:

- Sample fixed training set: `(x0_train, U_train)` with `x0_train` shape `(N_train, d)`, `U_train` shape `(N_train, C)` one-hot
- The target distribution is the empirical distribution: uniform mixture of N_train Dirac deltas.
- Compute empirical second-order statistics from the N_train samples (using `/N` for consistency with the empirical distribution's MMSE):

```
mu_x0    = (1/N) sum_i x0_i                       (d,)
Sigma_p0 = (1/N) X0_c^T X0_c                      (d, d)   [empirical covariance, /N not /N-1]
```

Random feature projections `Theta` (k, d) ~ N(0,1)/sqrt(d) and `Gamma` (k, C) ~ N(0,1)/sqrt(C)
are shared across all N_train values.

**Population GMM** (d=8 or d=32, C=3 classes):
- Component means: mu_0=(2,0,...), mu_1=(-1,1.5,0,...), mu_2=(-1,-1,1.2,0,...) in R^d
- Covariances: S0 = diag(1.2, 0.4, ...), S1 = diag(0.4, 1.0, 0.8, 0.4,...), S2 = random (AA^T + 0.5I)
- Equal weights w_c = 1/3

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

## Method 2: RF Theory — Per-Component GMM Population (N→∞) [Corrected]

**Key fix:** U = e_c is a one-hot vector (constant given component c), NOT Gaussian.
Using a single jointly-Gaussian (x0, U) model was incorrect. The correct approach
treats each GMM component separately and sums over components.

Theory function: `mmse_theory_gmm_pop` / `mmse_theory_gmm_pop_t`

### Per-component pre-activation statistics

For each component c and feature j:
```
mbar_{cj} = theta_j^T mu_c + Gamma[j,c]           (scalar, conditional)
           = theta_j^T mu_c                         (scalar, unconditional)

S_{cj}^2  = theta_j^T Sigma_c theta_j + sigma^2 ||theta_j||^2
```

where `mu_c`, `Sigma_c` are the GMM component mean and covariance.

### Hermite coefficients of ReLU (definition: c_n = (1/n!) <relu, He_n>_Gaussian)

```
c0(m, s) = s * phi(m/s) + m * Phi(m/s)             [E[relu(m + sz)]]
c1(m, s) = s * Phi(m/s)                             [first Hermite coeff]
c2(m, s) = s * phi(m/s) / 2
c3(m, s) = -m * phi(m/s) / 6                        [NEGATIVE sign — sign matters!]
```

where `phi` = standard normal PDF, `Phi` = standard normal CDF, `z ~ N(0,1)`.

### Stein covariance (per component, conditional)

```
alpha_{cj} = c1(mbar_{cj}, S_{cj}) / S_{cj}

Cov(x0, phi_j | c) = Sigma_c theta_j * alpha_{cj} + (mu_c - mu) * c0_{cj}

where mu = sum_c w_c mu_c  (population mean)
```

Aggregate over components:
```
Cov(x0, phi_j) = sum_c w_c [Sigma_c theta_j * alpha_{cj} + (mu_c - mu) * c0_{cj}]
```

Matrix form `(d, k)`:
```
Cov = sum_c w_c [Sigma_c @ Theta^T * alpha_c[None,:] + outer(mu_c - mu, c0_c)]
```

### Feature covariance Sigma_phi

**Between-component term:**
```
Sigma_phi^between = sum_c w_c outer(c0_c - mu_phi, c0_c - mu_phi)

where mu_phi = sum_c w_c c0_c    (mean of phi across population)
      c0_c   = [c0(mbar_{c1}, S_{c1}), ..., c0(mbar_{ck}, S_{ck})]   (k,)
```

**Within-component term** (Mehler / Hermite expansion):

For features i,j within component c, the full data+noise correlation is:
```
r_{ij,c} = (theta_i^T Sigma_c theta_j + sigma^2 theta_i^T theta_j) / (S_{ci} * S_{cj})
```

Hermite series (n=1,2,3):
```
Sigma_phi^within_c = sum_{n=1}^{3} n! * r_c^{[n]} * outer(Cn_c, Cn_c)

where r_c^{[n]} denotes elementwise power,
      C1_c = [c1(mbar_{c1}, S_{c1}), ...]   (k,)
      C2_c = [c2(mbar_{c2}, S_{c2}), ...]   (k,)
      C3_c = [c3(mbar_{c3}, S_{c3}), ...]   (k,)
```

**Exact diagonal (replaces truncated series at r=1):**

At i=j, r_{ii,c} = 1 and the Hermite series sum_n n!·(c_n)^2 diverges.
Replace diagonal entries with the exact formula:
```
Var[relu(mbar_{cj} + S_{cj} z)] = E[relu^2] - c0^2
  = (mbar_{cj}^2 + S_{cj}^2) * Phi(mbar_{cj}/S_{cj})
    + mbar_{cj} * S_{cj} * phi(mbar_{cj}/S_{cj})
    - c0(mbar_{cj}, S_{cj})^2
```

**Total:**
```
Sigma_phi = Sigma_phi^between + sum_c w_c Sigma_phi^within_c + lam * I
```

**MMSE:**
```
MMSE = Tr(Sigma_p0^pop) - Tr(Cov @ Sigma_phi^{-1} @ Cov^T)
```

Note: this uses the **population** GMM covariance `Tr(Sigma_p0^pop)` as the baseline
(N→∞ curve), not the empirical N_train sample covariance.

### Unconditional case
Set `Gamma = 0`: `mbar_{cj} = theta_j^T mu_c`, `Cov = sum_c w_c Sigma_c theta_j * alpha_{cj}`.

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

## Implementation Notes

- **Precompute cache:** `precompute_gmm_pop(gmm, Theta)` pre-computes `ThTh = Theta @ Theta^T`
  and `ThSigTh_c[c] = Theta @ Sigma_c @ Theta^T` once per (gmm, Theta) pair, then reuses
  across sigma values. This avoids redundant (k,k) matrix products (~3s saved per k-step on CPU).

- **CUDA dispatch:** `mmse_theory_gmm_pop_t` is a torch implementation fully vectorized over C
  components using einsum. Called automatically when `DEVICE=cuda`.

- **Normalization:** All empirical quantities use `/N` (not `/(N-1)`) so that the plug-in
  estimator is consistent with the empirical distribution's MMSE.

- **Hermite series truncation at n=3:** The series `sum_{n>=1} n! r^n c_n^2` converges for |r|<1
  but diverges at r=1. Truncating at n=3 underestimates the diagonal, making Sigma_phi
  ill-conditioned and causing the Stein MMSE to dip below the NW floor. The exact diagonal
  formula fixes this.

- **r clipping:** Within-component r_{ij,c} is clipped to (-1+1e-6, 1-1e-6) for off-diagonal
  entries only; diagonal entries use the exact formula above.
