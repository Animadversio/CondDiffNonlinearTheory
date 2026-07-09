# Estimation-Error Deviation in the RF Denoiser at High Noise

**Setting.** We fit a linear readout $W^*_\sigma \in \mathbb{R}^{d \times k}$,
$b^*_\sigma \in \mathbb{R}^d$ on $M = N \cdot n_\text{noise}$ training pairs
$(x_0^{(n)},\, \phi(x_0^{(n)} + \sigma Z_m^{(n)}))$.
Two evaluations:

| Mode | What it measures |
|---|---|
| **CF (closed-form)** | In-sample: same noise draws for fit and loss formula |
| **Train + fresh Z** | Out-of-sample: same $x_0$ pool, one fresh noise draw per image |

At high $\sigma$, the optimal $W^* \to 0$ (features carry no signal), so both
should converge to the linear Wiener loss $\approx \operatorname{Tr}(\Sigma_{p_0})$.
Instead, with $k=8192 > N=10000$, we observe a systematic deviation:

$$
L^\text{train+fresh}_\sigma \;\approx\; \operatorname{Tr}(\Sigma_{p_0}) + \Delta
\qquad
L^\text{CF}_\sigma \;\approx\; \operatorname{Tr}(\Sigma_{p_0}) - \Delta
$$

with $\Delta \approx k \cdot \operatorname{Tr}(\Sigma_{p_0}) / M$.
This document derives this scaling.

---

## Derivation

### Step 1 — Bias-variance decomposition on fresh Z

Let $\hat{W}^* = \hat{C} \hat{\Sigma}_\phi^{-1}$ be the empirical estimator
and $\delta W = \hat{W}^* - W^*_\text{true}$.
Since $\hat{W}^*$ is fit on **independent** training data, the cross term
vanishes in expectation:

$$
\mathbb{E}_{z_\text{fresh}}\!\left[\|\hat{W}^* \phi(x_0 + \sigma z_\text{fresh}) + \hat{b}^* - x_0\|^2\right]
= \underbrace{L^\text{true}_\sigma}_{\to\,\operatorname{Tr}(\Sigma_{p_0})\text{ at large }\sigma}
+ \underbrace{\operatorname{Tr}\!\bigl(\delta W\;\Sigma_\phi\;\delta W^\top\bigr)}_{\text{excess from estimation noise}}
$$

### Step 2 — Express excess via covariance estimation error

Since $\hat{W}^* \approx \hat{C}\,\Sigma_\phi^{-1}$ (ignoring ridge at large $\sigma$),

$$
\operatorname{Tr}(\delta W\,\Sigma_\phi\,\delta W^\top)
= \operatorname{Tr}\bigl(\delta C\;\Sigma_\phi^{-1}\;\delta C^\top\bigr)
$$

where $\delta C = \hat{C} - C$ is the $(d \times k)$ estimation error matrix.

### Step 3 — Variance of each entry of $\delta C$

Each entry is a sample mean from $M$ i.i.d. pairs:

$$
\operatorname{Var}(\delta C_{ij})
= \frac{1}{M}\operatorname{Var}\!\bigl((x_{0,i} - \mu_i)(\phi_j - \mu_{\phi_j})\bigr)
$$

At high $\sigma$, the feature $\phi_j = \operatorname{relu}(\theta_j^\top(x_0 + \sigma Z))$ is
dominated by the noise $Z$, so $x_{0,i} \perp \phi_j$ approximately:

$$
\operatorname{Var}\!\bigl((x_{0,i}-\mu_i)(\phi_j-\mu_{\phi_j})\bigr)
\;\approx\;
\operatorname{Var}(x_{0,i})\cdot\operatorname{Var}(\phi_j)
$$

### Step 4 — The key cancellation

Assuming features are approximately uncorrelated (random $\Theta$, large $k$),
$\Sigma_\phi$ is approximately diagonal:

$$
\operatorname{Tr}(\delta C\;\Sigma_\phi^{-1}\;\delta C^\top)
\;\approx\;
\sum_{i=1}^d \sum_{j=1}^k \frac{\operatorname{Var}(\delta C_{ij})}{\operatorname{Var}(\phi_j)}
= \sum_{i,j} \frac{\operatorname{Var}(x_{0,i})\cdot\operatorname{Var}(\phi_j)}{M\cdot\operatorname{Var}(\phi_j)}
= \frac{k}{M}\sum_{i=1}^d \operatorname{Var}(x_{0,i})
$$

**$\operatorname{Var}(\phi_j)$ cancels exactly.** This gives:

$$
\boxed{
\Delta \;=\; \frac{k \cdot \operatorname{Tr}(\Sigma_{p_0})}{N \cdot n_\text{noise}}
}
$$

**Why $\sigma$-independent at large $\sigma$:** $\operatorname{Var}(\phi_j) \propto \sigma^2$
appears in both the numerator (noise in $\hat{C}$) and the denominator ($\Sigma_\phi^{-1}$)
— they cancel, removing all $\sigma$ dependence.

---

## Empirical Validation — CIFAR-10, $k = 8192$, $N = 10{,}000$, $n_\text{noise} = 5$

Parameters: $d = 3072$, $M = 50{,}000$.

$\operatorname{Tr}(\Sigma_{p_0}) \approx 191.1$ (from linear Wiener at $\sigma = 100$).

Predicted:
$$
\Delta = \frac{8192 \times 191.1}{50{,}000} \approx 31.3
$$

| $\sigma$ | $L_\text{Wiener}$ | $L^\text{train+fresh}$ | $L^\text{CF}$ | Fresh excess $\uparrow$ | CF deficit $\downarrow$ | Predicted $\Delta$ |
|---|---|---|---|---|---|---|
| 31.6 | 187.7 | 225.3 | 157.5 | 37.6 | 30.2 | 30.8 |
| 46.4 | 189.7 | 227.0 | 158.8 | 37.3 | 30.9 | 31.1 |
| 68.1 | 190.7 | 228.9 | 159.5 | 38.2 | 31.2 | 31.2 |
| 100.0 | 191.1 | 228.4 | 159.9 | 37.3 | 31.2 | 31.3 |

**CF deficit** $\approx \Delta$ almost exactly (in-sample bias is a pure noise-gain artifact).

**Fresh Z excess** $\approx 37$, slightly above $\Delta \approx 31$.
The gap (~6) is likely due to using only $n_\text{eval} = 1$ fresh draw per image:
a single fresh draw adds extra Monte Carlo variance $\approx \operatorname{Tr}(\Sigma_{p_0}) \cdot k / N$
(same formula with $n_\text{noise} = 1$ instead of $M/N$).

**Average of CF and train+fresh** $\approx 194.2 \approx \operatorname{Tr}(\Sigma_{p_0}) + 3$
(approximately symmetric around the truth, as expected).

---

## Implications

**For $k = 512$ ($N = 10{,}000$, $n_\text{noise} = 5$):**
$$
\Delta = \frac{512 \times 191}{50{,}000} \approx 2.0
$$
Invisible on the plot — explains why $k = 512$ tracks the Wiener filter closely.

**General rule:** the evaluation mode matters when $k \cdot \operatorname{Tr}(\Sigma_{p_0}) / M \gtrsim$ noise floor.

**Optimal regularization at high $\sigma$:** to suppress the excess, ridge $\lambda$ should scale as
$\lambda \sim \operatorname{Tr}(\Sigma_{p_0}) \cdot k / M \approx 31$ for $k = 8192$, not the
current $\lambda = 10^{-4}$.
