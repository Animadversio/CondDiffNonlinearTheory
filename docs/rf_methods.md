# Random-Feature Denoiser: Methods for Computing $\mathcal{L}_\sigma$

This document explains the three methods used in `scripts/rf_theory_vs_empirical.py` to compute the MMSE loss for the random-feature denoiser.

**Full derivation by Michimin:** see [`newfile2.tex`](newfile2.tex) / [`newfile2_article.pdf`](newfile2_article.pdf).

---

## Setup

**Feature map.** $\phi(y) = \mathrm{relu}(\Theta y)$, where $\Theta \in \mathbb{R}^{k \times d}$ is a fixed random matrix (Gaussian entries, scaled by $1/\sqrt{d}$). No learning of $\Theta$.

**Denoiser.** Linear readout on top of the fixed features:
$$D_\theta(\phi(y)) = W_\sigma \phi(y) + b_\sigma$$

**Loss.** Expected MSE over $x_0 \sim p_0$ (CIFAR-10) and $Z \sim \mathcal{N}(0,I)$:
$$\mathcal{L}_\sigma = \mathbb{E}\bigl\|W_\sigma \phi(x_0 + \sigma Z) + b_\sigma - x_0\bigr\|^2$$

**Optimal weights** (derived in newfile2.tex, Section 1):
$$W^*_\sigma = \mathrm{Cov}(x_0,\, \phi)\; \Sigma_\phi^{-1}, \qquad b^*_\sigma = \mu_{p_0} - W^*_\sigma \mu_\phi$$

**Minimal loss** (plugging $W^*$ back in):
$$\mathcal{L}_\sigma = \mathrm{Tr}(\Sigma_{p_0}) - \mathrm{Tr}\!\bigl(\mathrm{Cov}(x_0,\phi)\;\Sigma_\phi^{-1}\;\mathrm{Cov}(\phi, x_0)\bigr)$$

All three methods below compute this same quantity — they differ in how $\mathrm{Cov}(x_0,\phi)$ and $\Sigma_\phi$ are obtained.

---

## Method 1 — Empirical Closed-Form (`rf_*_empirical_cf`)

**What it does.** Estimates $\mathrm{Cov}(x_0,\phi)$ and $\Sigma_\phi$ directly from data, plugs into the trace formula above. $W^*$ is never explicitly formed.

**How.** For each $\sigma$, draw $n_\text{noise}=5$ noise realizations per training image ($N=10{,}000$), giving $N \times n_\text{noise} = 50{,}000$ pairs $(y_i, x_{0,i})$:

$$\hat\Sigma_\phi = \frac{1}{N-1}\sum_i (\phi_i - \bar\phi)(\phi_i - \bar\phi)^\top + \lambda I$$
$$\widehat{\mathrm{Cov}}(x_0,\phi) = \frac{1}{N-1}\sum_i (x_{0,i} - \bar x_0)(\phi_i - \bar\phi)^\top$$
$$L = \mathrm{Tr}(\hat\Sigma_{p_0}) - \mathrm{Tr}\!\bigl(\widehat{\mathrm{Cov}}(x_0,\phi)\;\hat\Sigma_\phi^{-1}\;\widehat{\mathrm{Cov}}(\phi, x_0)\bigr)$$

**Key property.** Evaluation uses the same dataset used to estimate the covariances (in-sample). This tends to slightly *underestimate* the true loss (optimistic bias from finite samples).

---

## Method 2 — Empirical Direct (`rf_*_empirical_dir`)

**What it does.** Explicitly computes $W^*$ and $b^*$ from training data, then evaluates the MSE on a *held-out test set* (5,000 CIFAR-10 test images).

**How.**

1. From training data: $W^* = \widehat{\mathrm{Cov}}(x_0,\phi)\;\hat\Sigma_\phi^{-1}$, $b^* = \bar x_0 - W^* \bar\phi$
2. Draw one fresh noise realization for each test image: $y_\text{test} = x_{0,\text{test}} + \sigma Z$
3. Evaluate: $L = \frac{1}{N_\text{test}}\sum_n \|W^* \phi(y_n) + b^* - x_{0,n}\|^2$

**Key property.** Out-of-sample evaluation — avoids the in-sample optimism of Method 1. In practice slightly *higher* than Method 1 (train/test gap). As $N \to \infty$ both converge to the true $\mathcal{L}_\sigma$.

---

## Method 3 — Gaussian/Stein Theory (`rf_*_theory`)

**What it does.** Treats $p_0$ as a Gaussian with the empirical mean $\hat\mu_{x_0}$ and covariance $\hat\Sigma_{p_0}$, then computes $\mathrm{Cov}(x_0,\phi)$ and $\Sigma_\phi$ *analytically* using Stein's lemma and the Hermite expansion of ReLU. See newfile2.tex, Sections 1–2 for the full derivation.

**Key formulas.** For each feature $j$ with $\ell_j = \theta_j^\top y \mid x_0 \sim \mathcal{N}(M_j, s_j^2)$ where:
$$M_j = \theta_j^\top x_0, \quad s_j^2 = \sigma^2\|\theta_j\|^2 + \theta_j^\top \Sigma_{p_0} \theta_j$$

Hermite coefficients for ReLU (closed-form via Stein's identity):
$$c_1(M_j, s_j) = s_j\,\Phi(z_j), \quad c_2(M_j, s_j) = \frac{s_j\,\varphi(z_j)}{2}, \qquad z_j = \frac{M_j}{s_j}$$

where $\Phi,\varphi$ are the standard normal CDF/PDF. Then:

$$\mathrm{Cov}(x_0,\phi)_{:j} = \Sigma_{p_0}\,\theta_j\,\Phi(z_j) \quad \text{(Stein's lemma)}$$

$$\Sigma_{\phi,ij} \approx \tilde r_{ij}\,c_{1,i}\,c_{1,j} + 2\tilde r_{ij}^2\,c_{2,i}\,c_{2,j} \quad \text{(Hermite truncated at }n=2\text{)}$$

where $\tilde r_{ij} = \mathrm{Cov}(\ell_i, \ell_j)/(s_i s_j)$ is the pre-activation correlation.

**Key property.** Fully analytic — no data needed beyond $(\hat\mu_{x_0}, \hat\Sigma_{p_0})$. Approximates $p_0$ as Gaussian and truncates the Hermite series at $n=2$. Validated to closely match empirical estimates (~85.7 vs 86.3 at $\sigma=0.71$).

---

## Conditional Version (`rf_cond_*`)

The conditional feature map is $\phi^U(y, U) = \mathrm{relu}(\Theta y + \Gamma U)$ where $U$ is the one-hot class label and $\Gamma \in \mathbb{R}^{k \times C}$ is fixed random.

The empirical methods (1 & 2) apply without change — just replace $\phi(y)$ with $\phi^U(y,U)$.

For the theory (Method 3), the derivation extends by treating $(x_0, U)$ as jointly Gaussian (approximation). The modified pre-activation variance becomes:
$$\tilde s_j^2 = \theta_j^\top \Sigma_{p_0} \theta_j + 2\,\theta_j^\top C_{xU}\,\gamma_j + \gamma_j^\top \Sigma_U \gamma_j + \sigma^2\|\theta_j\|^2$$

and $\mathrm{Cov}(x_0,\phi^U)_{:j} = (\Sigma_{p_0}\theta_j + C_{xU}\gamma_j)\,\Phi(\tilde z_j)$ where $C_{xU} = \mathrm{Cov}(x_0, U)$.

---

## Results Figure

![Theory vs Empirical](../figures/rf_theory_vs_empirical_cifar10.png)

*CIFAR-10, $N_\text{train}=10{,}000$, $k=512$ ReLU features, $\sigma \in [0.01, 50]$.*

**Main finding:** Theory closely tracks empirical at all $\sigma$, validating the Stein/Hermite approximation. Random features relu($\Theta y$) are substantially worse than linear Wiener (e.g. 86 vs 41 at $\sigma=0.71$) because random projection $d=3072 \to k=512$ discards most pixel information.
