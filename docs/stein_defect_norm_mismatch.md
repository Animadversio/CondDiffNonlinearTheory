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

## CORRECTION: the sharp quantity is already in the proof, and it is not `Delta^T Sigma Delta`

An earlier version of this note suggested replacing `||Sigma||_op^2 ||Delta||^2` by
`Delta^T Sigma Delta`. That was the right instinct but the wrong object. Tracing
Lemma (calculus of T), step (d), the chain is

    EV(r; zeta2~)  <=  Tr( Cov(r,x) P_2 C_2^{-1} P_2^T Cov(x,r) )        <- SHARP
                   <=  ||Cov(r,x)||_op^2 Tr( P_2 C_2^{-1} P_2^T )        <- weighting discarded HERE
                   <=  ||Sigma||_op^2   Tr( P_2 C_2^{-1} P_2^T )         <- Cov(r) <= Sigma

so the sharp per-feature error term is

    sum_j  || Cov(r) Delta_j ||^2 / (rho_* gamma_j),

weighted by `Cov(r)` -- the residual covariance of the matched LINEAR model -- not by
`Sigma`. In the wide-feature limit `Cov(r)` is the Wiener residual

    Cov(r) = sigma^2 Sigma (Sigma + sigma^2 I)^{-1},   eigenvalues  sigma^2 lam/(lam + sigma^2).

This is better than a `Sigma` weighting in BOTH tails:
  * small lam (near-null directions): weight -> lam, so they are suppressed, as `Sigma`
    weighting would also do;
  * large lam: weight SATURATES at `sigma^2` rather than growing to `||Sigma||_op`.
The second is what the operator-norm step throws away, and it is the larger effect:
`||Sigma||_op^2 = 3070` at CIFAR scale versus a cap of `sigma^4 = 0.0039` at sigma = 0.25.

Measured on CIFAR-10 pixels:

| sigma | shrink | `\|\|Sigma\|\|_op^2 E\|\|Delta\|\|^2` | `E\|\|Cov(r)Delta\|\|^2` | loose / sharp |
|---|---|---|---|---|
| 0.25 | 1e-4 | 2.56e+03 | 3.77e-07 | 6.8e9 |
| 0.25 | 1e-2 | 5.40e+02 | 3.81e-07 | 1.4e9 |
| 0.5  | 1e-4 | 1.45e+03 | 7.88e-07 | 1.8e9 |
| 0.5  | 1e-2 | 4.57e+02 | 8.01e-07 | 5.7e8 |
| 1.0  | 1e-4 | 6.44e+02 | 1.45e-06 | 4.4e8 |
| 1.0  | 1e-2 | 3.87e+02 | 1.47e-06 | 2.6e8 |

Two things to note. The gap is 8-10 orders of magnitude, not the 6-7 estimated from the
`Sigma`-weighted proxy. And the sharp quantity is **essentially shrinkage-free**
(3.77e-07 vs 3.81e-07 across two decades of ridge, i.e. 1%, versus a 4.7x swing for the
loose one) -- because `Cov(r)` annihilates exactly the near-null subspace that made
`Sigma^{-1}` ill-posed in the first place.

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


## Is `Cov(r) <= Sigma` used anywhere else?  No -- the change is local

Traced every occurrence in the note.

**`Cov(r) <= Sigma` appears exactly once**: line 238, inside the proof of Lemma
(calculus of T) step (d), and only to conclude `||Cov(r,x)||_op <= bar c`. Lines 293 and
580 merely restate it in commentary. There is no independent use.

**Every appearance of `||Sigma||_op` descends from that single step**:

| line | where | status |
|---|---|---|
| 216 | statement of Lemma calc(d) | the claim to be modified |
| 240 | proof of (d) | the discarding step itself |
| 271 | boxed master theorem | downstream of (d) |
| 290, 291 | assembly chain | downstream |
| 547, 572, 573 | Theorem (prob) | downstream |
| 580 | commentary | -- |

So replacing the step-(d) conclusion propagates mechanically; nothing else has to be
re-proved on that account.

**`bar c` IS used elsewhere, but for an unrelated purpose.** Lines 339 (Assumption: data,
`c I <= Sigma <= bar c I`), 353, 356, 375, 377 and 513 use `bar c` to bound
`Var(u) = theta^T Sigma theta <= 4 bar c` for the pre-activation `u = theta^T x_0 +
epsilon`, which fixes the level `R_0 = 2 tau_eps + 4 sqrt(bar c)` at which the activation
modulus `m(R)` must be positive. That is a statement about the ACTIVATION, not about the
signal coupling, and is untouched by reweighting step (d). (Line 443 is inside Lemma 9,
the product-measure route, which is not used here anyway.)

## The refined weight is Theta-independent for k >= d

One might worry that `Cov(r)` depends on the design, since `r` is the residual of
predicting `x` from `zeta_1 = bar B^T x + nu_1`. It does not, in the regime of interest.
By construction `nu_1 ~ N(0, sigma^2 bar B^T bar B)`, so

    zeta_1  =d  bar B^T (x + sigma Z)  =  bar B^T y,

i.e. `zeta_1` is exactly the matched linear read of the noisy observation. If `bar B^T` has
full column rank `d` -- generic once `k >= d` -- the map `y -> bar B^T y` is injective, so
`sigma(zeta_1) = sigma(y)` and the best predictor is the Wiener filter itself. Hence

    Cov(r) = Sigma - Sigma Sigma_y^{-1} Sigma = sigma^2 Sigma (Sigma + sigma^2 I)^{-1},

deterministic and independent of `Theta`. For `k < d` the read is a compression and
`Cov(r) >=` this, so the Wiener residual is the right (and favourable) surrogate throughout.

Since the whole point of `k_*` is the regime `k ~ 10^4 >> d`, the refined error term

    (1/rho_*) sum_j || sigma^2 Sigma (Sigma + sigma^2 I)^{-1} Delta_j ||^2 / gamma_j

is fully explicit, needs no new probabilistic input, and is what the existing proof
already establishes one line above the operator-norm step.
