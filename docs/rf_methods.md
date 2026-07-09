# Random-Feature Denoiser: Methods for Computing $\mathcal{L}_\sigma$

This document explains the methods used in `scripts/rf_theory_vs_empirical.py`. Full derivation: [`newfile2.tex`](newfile2.tex) / [`newfile2_article.pdf`](newfile2_article.pdf).

---

## Setup

**Feature map.** $\phi(y) = \omega(\Theta y + \epsilon)$, $\Theta \in \mathbb{R}^{k \times d}$ fixed random Gaussian matrix, $\epsilon$ fixed random bias (both unlearned).

**Denoiser.** Linear readout on top of fixed features:
$$D_\theta(\phi(y)) = W_\sigma \phi(y) + b_\sigma$$

**Optimal weights** (newfile2.tex §1):
$$W^*_\sigma = \mathrm{Cov}(x_0, \phi)\,\Sigma_\phi^{-1}, \qquad b^*_\sigma = \mu_{p_0} - W^*_\sigma \mu_\phi$$

**Minimal loss:**
$$\mathcal{L}_\sigma = \mathrm{Tr}\!\bigl(\Sigma_{p_0} - \mathrm{Cov}(x_0,\phi)\,\Sigma_\phi^{-1}\,\mathrm{Cov}(\phi, x_0)\bigr)$$

All methods below compute this quantity — they differ only in how $\mathrm{Cov}(x_0,\phi)$ and $\Sigma_\phi$ are obtained.

---

## Method 1 — Empirical Closed-Form (`rf_*_empirical_cf`)

**What.** Sample covariances from noisy observations plugged directly into the trace formula. $W^*$ is never formed explicitly.

**How.** For each $\sigma$, draw $n_\text{noise}=5$ noise realizations per training image ($N=10{,}000$):
$$\hat\Sigma_\phi = \frac{1}{N\,n_\text{noise}-1}\sum_i (\phi_i - \bar\phi)(\phi_i - \bar\phi)^\top + \lambda I$$
$$\widehat{\mathrm{Cov}}(x_0,\phi) = \frac{1}{N\,n_\text{noise}-1}\sum_i (x_{0,i} - \bar x_0)(\phi_i - \bar\phi)^\top$$

where $\phi_i = \omega(\Theta(x_0^{(i)} + \sigma Z^{(i)}) + \epsilon)$ is the noisy feature.

**Difference vs Method 3.** Both estimating the same population quantities, but Method 1 averages over noise draws (MC over $Z$), while Method 3 does the $Z$-integral analytically. Method 1 is in-sample (train covariances); slightly optimistic.

---

## Method 2 — Empirical Direct (`rf_*_empirical_dir`)

**What.** Explicitly computes $W^*$ from training data, evaluates MSE on a held-out test set.

**How.**
1. $W^* = \widehat{\mathrm{Cov}}(x_0,\phi)\,\hat\Sigma_\phi^{-1}$, $b^* = \bar x_0 - W^*\bar\phi$ (from train)
2. Draw one fresh noise realization for each test image
3. $L = \frac{1}{N_\text{test}}\sum_n \|W^* \phi(y_n) + b^* - x_{0,n}\|^2$

**Key property.** Out-of-sample — avoids in-sample optimism of Method 1. Both converge to $\mathcal{L}_\sigma$ as $N \to \infty$.

---

## Method 3 — Theory: Exact $p_0$ + Hermite for Noise (`rf_*_theory`)

This is the main contribution, following newfile2.tex §1–2. **No assumption on $p_0$.**

### Key reduction to 1D

Conditional on $x_0$, each feature $j$ depends on $Z$ only through the scalar $\xi_j = \theta_j^\top Z / \|\theta_j\| \sim \mathcal{N}(0,1)$:

$$\phi_j(x_0 + \sigma Z) = \omega\!\bigl(\underbrace{\theta_j^\top x_0 + \epsilon_j}_{M_j(x_0)} + \underbrace{\sigma\|\theta_j\|}_{s_j}\,\xi_j\bigr)$$

Define $f_j(\xi) = \omega(M_j(x_0) + s_j\,\xi)$. The $d$-dimensional noise integral reduces to a 1D expectation over $\xi_j$.

### Hermite expansion (newfile2.tex §1)

Expand $f_j(\xi)$ in the Hermite basis (orthogonal under $\mathcal{N}(0,1)$):
$$f_j(\xi) = \sum_{n=0}^\infty c_n(M_j(x_0), s_j)\,\mathrm{He}_n(\xi)$$

where:
$$c_n(M_j, s_j) = \frac{1}{n!}\langle f_j, \mathrm{He}_n\rangle_\varphi = \frac{1}{n!}\mathbb{E}_{\xi\sim\mathcal{N}(0,1)}[f_j(\xi)\,\mathrm{He}_n(\xi)]$$

Note: $\langle \mathrm{He}_m, \mathrm{He}_n\rangle_\varphi = n!\,\delta_{mn}$. The first Hermite polynomials (from $\mathrm{He}_n = (-1)^n e^{x^2/2}\frac{d^n}{dx^n}e^{-x^2/2}$):
$$\mathrm{He}_0 = 1,\quad \mathrm{He}_1 = x,\quad \mathrm{He}_2 = x^2-1,\quad \mathrm{He}_3 = x^3-3x$$

For **ReLU** $\omega = \max(0,\cdot)$, using Stein's identity ($\mathbb{E}[f'(\xi)] = \mathbb{E}[f(\xi)\xi]$):

| coefficient | analytic formula | code variable |
|---|---|---|
| $c_0(M,s) = g_j(x_0)$ | $M\Phi(z) + s\varphi(z)$ | `G` |
| $c_1(M,s) = \frac{1}{1!}\mathbb{E}[f\,\xi]$ | $s\,\Phi(z)$ | `C1` |
| $c_2(M,s) = \frac{1}{2!}\mathbb{E}[f\,(\xi^2-1)]$ | $\frac{s\,\varphi(z)}{2}$ | `C2` |

where $z = M/s$.

### Cross-covariance $\mathrm{Cov}(x_0, \phi)$ (newfile2.tex §1, eq. after eq. 15)

$$\mathrm{Cov}(x_0, \phi)_{ij} = \mathbb{E}_{x_0}[(x_{0,i} - \mu_{p_0,i})\,g_j(x_0)] = \mathbb{E}_{u_j}[g_j(u_j)\,r_i(u_j)]$$

where $u_j = \theta_j^\top x_0 + \epsilon_j$ and $r_i(u_j) = \mathbb{E}_{x_0}[x_{0,i} - \mu_i \mid u_j]$. The two expressions are equal by the tower property: the multi-dimensional integral over $p_0$ reduces to a 1D integral over $u_j$.

**In code** ($G_{nj} = g_j(x_0^{(n)})$, $X_0^c = x_0 - \bar x_0$):
```python
Cov_x0_phi = X0_c.T @ G / N   # (d, k) — sample average over actual data
```

### Feature covariance $\Sigma_\phi$ (newfile2.tex §1, final eq.)

By the law of total variance (newfile2.tex §1):
$$\Sigma_{\phi,ij} = \underbrace{\mathrm{Cov}_{x_0}(g_i(x_0), g_j(x_0))}_{\text{between-}x_0\text{ variance}} + \underbrace{\mathbb{E}_{x_0}\!\left[\sum_{n\geq 1} n!\,\rho_{ij}^n\,c_n(M_i,s_i)\,c_n(M_j,s_j)\right]}_{\text{within-}x_0\text{ variance (noise), Mehler identity}}$$

The noise-only pre-activation correlation (newfile2.tex §1, eq. for $\rho_{ij}$):
$$\rho_{ij} = \mathrm{Corr}(\xi_i, \xi_j) = \frac{\theta_i^\top\theta_j}{\|\theta_i\|\|\theta_j\|}$$

Truncated at $n=2$ (validated — see Gaussian Equivalence Conjecture discussion):

**In code** (with $C_1, C_2 \in \mathbb{R}^{N\times k}$ sample matrices and $\rho \in \mathbb{R}^{k\times k}$):
```python
rho = (Theta @ Theta.T) / (norm_i * norm_j)    # sigma cancels: s_i s_j = sigma^2 ||theta_i|| ||theta_j||
Sig_data  = G_c.T @ G_c / N                     # Cov_{x0}(g_i, g_j)
Sig_noise = rho * (C1.T @ C1 / N) \
          + 2 * rho**2 * (C2.T @ C2 / N)        # n=1 and n=2 Hermite terms
Sigma_phi = Sig_data + Sig_noise + lam * I
```

**Key property.** The Hermite expansion applies only to the noise $\xi_j$ (Gaussian — exact). The data variation of $g_j$ is captured exactly by sampling from the actual distribution. No Gaussian assumption on $p_0$.

---

## Conditional Version (`rf_cond_*`)

$\phi^U(y,U) = \omega(\Theta y + \Gamma U + \epsilon)$, $U$ = one-hot class label.

Methods 1 & 2 apply unchanged (replace $\phi(y)$ with $\phi^U(y,U)$).

Method 3: the pre-activation mean per sample becomes $M^U_j(x_0,U) = \theta_j^\top x_0 + \gamma_j^\top U + \epsilon_j$, with noise std $s_j = \sigma\|\theta_j\|$ unchanged (noise enters only through $\theta_j^\top \sigma Z$). The cross-correlation $\rho_{ij} = \theta_i^\top\theta_j / (\|\theta_i\|\|\theta_j\|)$ is also unchanged. Formulas for $\mathrm{Cov}(x_0,\phi^U)$ and $\Sigma^U_\phi$ are identical, with $G_{nj} = g^U_j(x_0^{(n)}, U^{(n)})$ computed over the joint $(x_0, U)$ training distribution.

---

## Results Figure

![Theory vs Empirical](../figures/rf_theory_vs_empirical_cifar10.png)

*CIFAR-10, $N_\text{train}=10{,}000$, $k=512$ ReLU features, $\sigma \in [0.01, 100]$.*

**Main finding:** Theory (Method 3) closely tracks both empirical methods across all $\sigma$, confirming that the Hermite expansion truncated at $n=2$ is accurate (Gaussian Equivalence Conjecture). Random features relu($\Theta y$) are substantially worse than linear Wiener — e.g. uncond RF MSE $\approx 86$ vs Wiener $\approx 41$ at $\sigma = 0.71$ — because the random projection $d=3072 \to k=512$ discards most pixel information.
