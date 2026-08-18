# Structured block-circulant `L^circ`: method, estimator choice, and validation

Companion to `core/rf_circulant_struct.py`. Written after a exchange with michimin that
corrected two things I had stated loosely; both corrections are recorded here rather than
quietly fixed.

## 1. What is being computed

Exactly the closed form of `docs/circulant linear and nonlinear.tex` §4, unchanged:

```
L^circ = Tr(Sigma_p0) - sum_f q_f^H P_f^{-1} q_f
(P_f)_{ab} = f_f^H Sigma_{phi_a,phi_b} f_f        P_f in C^{c x c}, c = #blocks
(q_f)_a    = f_f^H Sigma_{phi_a,x0}   f_f         f = 1..d frequencies
```

Nothing about the formula is modified. The only question this module answers is *how to
obtain `Sigma_{phi_a,phi_b}` at scale*.

## 2. Why a new code path was needed

`core/rf_circulant_torch.circulant_rf_mmse_t` builds `Sigma_phi` as a dense `K x K` matrix
(`K = c*d`) and then slices out the per-frequency `c x c` blocks. That is fine at
`K ~ 10^4`. The free-parameter-matched setting is not:

matching free parameters means `Theta` and `W` should have `k*d` entries on both sides.
Dense width `k` has exactly that; block-circulant with `c` blocks has `c*d`. So `c = k`
blocks, hence `c*d = k*d` **total features** — 262,144 already at `k/d = 1`, `d = 512`,
where `K x K` is 4.4 TB.

This module computes the same `P_f`, `q_f` in `O(c^2 d)` memory instead of `O(c^2 d^2)`.

**Cost scaling (michimin's, and correct).** The structured form needs a `c x c` Gram at each
of `d` frequencies, `N c^2 d`, against dense's `N k^2`; with `c = k` that is exactly a factor
`d`, not `d^2`. I initially claimed ~10 h at `k/d = 16` from a microbenchmark that used
128x128 output tiles — far below peak, whereas the dense case runs one large `k x k` GEMM
near peak. Measured properly, one dense evaluation at `k = 8192` takes 0.21 s, so `512x` is
~107 s. **The `d`-fold rule is right and the inflated estimate was a benchmarking artifact.**

## 3. Estimator choice — the part that is *not* forced by the writeup

michimin asked why the Stein/Mehler machinery appears at all, since the closed form does not
use it. It doesn't, and the question is fair.

`Sigma_{phi_a,phi_b}` can simply be the **empirical** covariance of the features. Via
DFT-transformed features that is

```
P_f[a,b] = (1/N) sum_n uhat_a[n,f] conj(uhat_b[n,f])
```

a `c x c` Gram per frequency at the same `c^2 d` cost — **no rho, no Hadamard product, no
frequency convolution, and neither bug in §5 would have existed.** That route is strictly
simpler.

The reason it was not taken: our existing `L^dense` curves compute `Sigma_phi`
*analytically*, marginalising the noise `z` in `y = x0 + sigma z` in closed form
(`stein_finiteN_mmse_t` -> `stein_covariances_t`). That marginalisation is what produces

```
Sigma_noise = rho o (C1^T C1/N) + 2 rho^2 o (C2^T C2/N) + 6 rho^3 o (C3^T C3/N)
```
(`core/rf_gmm_estimators_torch.py:77`), with the diagonal subsequently *replaced* by the
exact conditional variance. Using the same estimator for `L^circ` makes the two curves
differ **only** in the law of `Theta` and the constraint on `W`. Mixing an analytic-noise
curve with a sampled-noise curve would leave part of the measured gap attributable to the
estimator, and a confound of exactly that kind already forced one retraction in this project
(see the finite-sample `Tr(Sigma)` deficit note in `docs/rf_circulant_methods.md`).

So: **the ρ-structure is a property of our estimator, not of the writeup — but it was also
not forced.** It is a deliberate trade, buying estimator-consistency at the price of a harder
derivation. The honest framing is "I chose a harder route than the formula asks for", not
"the noise structure isn't in the writeup".

Grepping `docs/circulant linear and nonlinear.tex`: zero occurrences of `rho`, `Hadamard`,
`\circ`, `Mehler`, or `DFT`.

## 4. The frequency-domain identity

`rho` is block-circulant: `rho^{(a,b)}[p,q] = rho0_{ab}(p-q)/(n_a n_b)` with
`rho0_{ab}(m) = sum_j h_a[j] h_b[j+m]`. Expanding in the DFT basis and using
`F[f,p] F[g,p] = F[f+g,p]/sqrt(d)`,

```
f_f^H (rho^{o n} o M) f_f = (1/d) sum_g w_g D_{ab}(f+g),   D_{ab}(f') = f_{f'}^H M^{(a,b)} f_{f'}
```

a cyclic **correlation** over the frequency index. This is the step that couples all
frequencies, and the one that had to be validated rather than trusted.

`w` is rank-one in `(a,b)` for each `g` (it factorises through `hhat_a`, `hhat_b`), which is
what keeps the whole thing tractable; the Hadamard *powers* `rho^{o2}`, `rho^{o3}` are not
rank-one, but are still only `c^2 d` to form per block-pair chunk.

## 5. The two bugs, both mine

1. **Spectrum convention.** The expansion `rho[p,q] = sum_g w_g F[g,p] conj(F[g,q])` holds
   with `w_g = sum_m psi(m) e^{+2 pi i m g/d} = d * ifft(psi)`. I first used a contraction
   with `F`, which is wrong by a sign in the exponent *and* a factor `sqrt(d)`.
2. **Conjugate in the correlation.** The correlation consumes `conj(w) = fft(psi)`, not `w`.

Fixing only (1) still left 16–45% relative error, so neither is cosmetic. They were isolated
by checking each term separately against the reference `K x K` build: `q` and `P^data`
matched to 1e-15 immediately, which localised the fault to the noise term.

Neither bug came from the writeup, and neither was a misreading of it — they were in a
derivation that only exists because of the §3 estimator choice.

## 6. Validation

`python -m core.rf_circulant_struct` checks against the reference `K x K` implementation,
which has itself matched a brute-force optimisation over block-circulant `W` to ~1e-16 since
July.

```
    d    c    w      reference        structured       rel err
    8    2   full    5.369045515446   5.369045515446   6.6e-16
    8    3   3       9.081062373972   9.081062373972   7.8e-16
   12    2   full    9.790288620006   9.790288620006   2.2e-15
   16    4   5      15.803913155461  15.803913155461   9.0e-16
   16    3   2      38.842960063747  38.842960063747   7.3e-16
```

and in the regime actually used, `c = c0*d` blocks with `c > d`:

```
    8   16 (c0=2) full   1.0e-15
    8   32 (c0=4) w=3    2.4e-15
   12   24 (c0=2) w=4    5.0e-15
   10   30 (c0=3) full   0.0
```

plus forced-small `block_chunk` cases so the streaming path is exercised. Coverage: full-width
and banded filters, `c < d` and `c > d`, chunked and unchunked.

## 7. Resource limits — the ones that actually bind

Real-scale check: `k/d = 1` (`c = 512` blocks, `K = 262,144` features, `N = 2000`) evaluates
in **46 s** at 27 GB peak GPU.

Two limits, both discovered the hard way:

* **GPU.** The per-frequency Grams are `(d, c, c)`: 8.6 GB each at `c = 1024`, 34 GB at
  `c = 2048`, and there are four of them (`P^data` plus three `D`). Past `c ~ 1024` they must
  live host-side and stream. The ρ-convolution is elementwise in `(a,b)` and so streams over
  block-pair chunks cleanly.
* **Host RAM is capped by the Slurm cgroup, not by the node.** Caching every chunk's
  transform costs `4 c N d * 16` bytes = **156 GB** at `c = 512, N = 10^4, d = 512`. The node
  has 1.5 TB, but the allocation is `mem=150G`, and the first run was OOM-killed at ~156 GB
  RSS. I had asserted "1.3 TB free, so it fits" without checking the cgroup. The module now
  takes `host_budget_gb` (default 60), caches only what fits, and recomputes the rest —
  a few seconds per uncached chunk per outer iteration, versus fatal.

## 8. Status

`k/d in {1, 2}` is reachable as written. `{4, 8}` additionally need the ρ-convolution
assembly streamed over block-pair chunks (contained change, same identity). `k/d = 16` needs
that plus host-side `P_f` and is the point at which the `512x` rule stops being free.
