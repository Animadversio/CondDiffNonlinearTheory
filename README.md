# CondDiffNonlinearTheory

Numerical validation of the random-feature conditional denoiser theory
from "Conditional Non-linear Denoiser" (theory notes, 2026).

## Theory summary

The denoiser uses a random-feature map:

```
phi^U(y, U) = omega(Theta @ y + Gamma @ U + epsilon)
```

The Bayes-optimal linear-head loss is:

```
L_{sigma,U} = Tr(Sigma_p0) - Tr(Cov(x0,phi^U) @ inv(Sigma_phi^U) @ Cov(phi^U,x0))
```

Under jointly Gaussian (x0, U), Stein's lemma gives closed-form expressions
for all quantities via Hermite expansion coefficients and the Mehler product identity.

## Structure

```
core/           # reusable library
  hermite.py    # Hermite polynomials He_n, expansion coefficients c_n(m,s)
  denoiser.py   # RandomFeatureMap, empirical/theoretical loss
  gaussian.py   # JointGaussian, Stein's lemma, gaussian_theoretical_loss
  metrics.py    # MI approximation, sigma sweep, R^2 gain
scripts/        # experiment scripts
notebooks/      # Jupyter analysis notebooks
bash/           # Slurm job scripts
figures/        # output figures
tables/         # output tables
```

## Quick start

```python
import numpy as np
from core import JointGaussian, RandomFeatureMap, gaussian_theoretical_loss

# Define distribution
d, d_u, k = 10, 5, 64
dist = JointGaussian(
    mu_x=np.zeros(d), mu_U=np.zeros(d_u),
    Sigma_p0=np.eye(d), Sigma_U=np.eye(d_u),
    C_xU=0.3 * np.random.randn(d, d_u),
)

# Random feature map
Theta   = np.random.randn(k, d) / np.sqrt(d)
Gamma   = np.random.randn(k, d_u) / np.sqrt(d_u)
epsilon = np.random.randn(k)
omega   = np.tanh

# Theoretical loss under Gaussian assumption
results = gaussian_theoretical_loss(omega, Theta, Gamma, epsilon, sigma=1.0, dist=dist)
print(f"L_sigma={results['L_sigma']:.4f}, L_sigma_U={results['L_sigma_U']:.4f}, "
      f"gain={results['gain']:.4f}")
```

## Experiments

| Script | What it validates |
|--------|------------------|
| `scripts/verify_unconditional.py` | Empirical vs theoretical L_sigma |
| `scripts/verify_conditional.py`   | Empirical vs theoretical L_{sigma,U} |
| `scripts/verify_jointly_gaussian.py` | Stein's lemma simplification |
| `scripts/sweep_sigma.py`          | MI approximation across sigma |
