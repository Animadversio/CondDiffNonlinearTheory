# The Euclidean defect norm is badly matched to image covariances

Empirical note arising from computing `k_*` for CIFAR-10 pixels.

## The observation

For the theorem's error term we need `hat_eps_d^2 = E||Delta||^2` with the **Euclidean**
norm, and the bound is

    L^RF >= L^lin - 2 ||Sigma||_op^2 * k * check_eps_d^2 / delta.

Decomposing the measured defect in the eigenbasis of `Sigma` (CIFAR-10 raw pixels,
d=3072, sigma=0.5) shows the defect norm and the denoising loss are supported on
**opposite ends of the spectrum**:

| top-r directions | eig_r / eig_0 | cum. % of `\|\|Delta\|\|^2` | cum. % of linear loss |
|---|---|---|---|
| 10   | 4.95e-02 |   0.00% |   7.66% |
| 100  | 2.56e-03 |   0.01% |  49.66% |
| 500  | 1.89e-04 |   0.26% |  90.20% |
| 1000 | 3.65e-05 |   1.73% |  97.83% |
| 2000 | 1.88e-06 |  24.58% |  99.90% |
| 3072 | 2.75e-08 | 100.00% | 100.00% |

98% of `||Delta||^2` sits in directions that carry ~2% of the loss. Those are exactly the
directions where `Sigma` is at numerical-noise level (865 of 3072 eigenvalues are below
1e-6 x max; condition number 3.6e7), i.e. where `Sigma^{-1}` in
`Delta = Sigma^{-1}Cov(x0, psi) - alphabar theta` is meaningless.

## Consequences

1. **This is why `k_*` for images is shrinkage-sensitive.** The reported `k_*` moves 2-10x
   across ridge levels 1e-4..1e-2 because the quantity being measured is dominated by the
   near-null subspace, not by the signal.
2. **It is not a preprocessing problem.** Ledoit-Wolf only chooses the ridge level
   optimally (in Frobenius error); it does not change which directions dominate
   `||Delta||^2`. PCA truncation does remove them, but it changes `d` and the
   representation, so it computes `k_*` for a different problem (see below).
3. **The looseness is enormous.** Comparing the Euclidean term actually used against a
   Sigma-weighted defect `Delta^T Sigma Delta` (CIFAR, sigma=0.5):

   | shrink | `E\|\|Delta\|\|^2` | `E Delta^T Sigma Delta` | `\|\|Sigma\|\|_op^2 E\|\|Delta\|\|^2` | ratio |
   |---|---|---|---|---|
   | 1e-4 | 4.73e-01 | 6.10e-05 | 1.45e+03 | 2.4e7 |
   | 1e-3 | 1.39e-01 | 5.24e-05 | 4.26e+02 | 8.1e6 |
   | 1e-2 | 1.49e-01 | 5.97e-05 | 4.57e+02 | 7.7e6 |

   The Sigma-weighted defect is 6-7 orders of magnitude smaller, and is also far more
   stable in the shrinkage (5.2e-5 to 6.1e-5 across two decades of ridge, vs a 3.4x swing
   for the Euclidean one).

## Where this enters the proof

`||Sigma||_op` is introduced at a single step of the master theorem: step (d), justified by
"`Cov(r) <= Sigma` gives `||Cov(r,x)||_op <= ||Sigma||_op`". That is precisely where the
`Sigma`-weighting is discarded in favour of its operator-norm envelope. Keeping it would
give an error term of the shape

    sum_j  Delta_j^T Sigma Delta_j / gamma_j        instead of      ||Sigma||_op^2 sum_j ||Delta_j||^2 / gamma_j,

which automatically de-weights the near-null directions. For a covariance with a flat
spectrum (the synthetic GMM: eigenvalues in [0.42, 3.66], condition 8.7) the two agree up
to a constant, which is why this never showed up in the GMM experiments. For natural images
(7 decades of dynamic range) it is the difference between a usable and a vacuous bound.

**Suggestion:** if a `Sigma`-weighted defect can be carried through the assembly, `k_*` for
image data becomes both far larger and essentially free of the shrinkage nuisance. Whether
step (d) survives the change is a question for the proof, not for the experiment.

## Why PCA truncation is *not* the fix for `k_*`

Restricting to the top-r subspace makes Assumption 1 (`Sigma` uniformly PD, bounded)
genuinely true, but:

* `k_* = d / hat_eps_d^2` is explicit in `d`; truncating gives `k_*` for the r-dimensional
  problem, not for denoising the image;
* `hat_eps` is representation-dependent (`Delta -> A^{-T}Delta` under `x -> Ax`), so
  `hat_eps_r` is not an approximation to `hat_eps_d`;
* the RF denoiser under study acts on the full image, so a truncated `k_*` answers a
  different question.

So PCA is the right move only if the object of study is redefined to be the r-dimensional
representation. As a fix for the reported `k_*` it is a category error.
