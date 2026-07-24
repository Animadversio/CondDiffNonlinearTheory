# Circulant-constrained RF denoiser — methods & the L^circ-vs-L^dense experiment

Implements newfile5.tex Sections 3–4: the **block-circulant** nonlinear random-feature
(RF) denoiser and the experiment testing whether there is a window of widths `k` where
the circulant constraint beats the dense one.

Code:
- `core/rf_circulant.py` — numpy: `build_circulant_theta`, `build_circulant_gamma`,
  `circulant_rf_mmse`, plus a machine-precision self-test (`python -m core.rf_circulant`).
- `core/rf_circulant_torch.py` — CUDA port `circulant_rf_mmse_t` (batched per-frequency solve).
- `scripts/rf_circulant_win.py` — Section-4 win-region sweep (GMM d=32).
- `scripts/rf_gmm_finite_sample.py` — the per-`N_train` figures now carry an `L^circ` curve.

---

## 1. Setup

`k = c·d`. The projection `Θ ∈ R^{k×d}` is constrained to `c` blocks, each a `d×d`
**circulant** matrix:

```
Θ = [Θ_1; Θ_2; …; Θ_c],   Θ_a = circ(h_a),   h_a ~ N(0, I/d)
```

so row `(a,τ)` is the cyclic shift `Θ_{(a,τ)} = S^τ h_a` (`np.roll(h_a, τ)`). Every entry
is marginally `N(0,1/d)` — **identical to the dense RF**; only the *joint* law of the rows
differs (dependent within a block vs. i.i.d. dense). The readout `W ∈ R^{d×k}` is likewise
block-circulant, `W = [W_1 … W_c]`, each `W_a` `d×d` circulant. No additive unconditional
bias `ε` is used (matching the dense `stein` curve exactly), so the *only* changed variable
is the row-joint of `Θ`. For the conditional curve the label bias `Γ U` is shared across a
block's `d` rows (shift-equivariant): `Γ_{(a,τ)} = γ_a`.

## 2. Per-frequency closed form

Both `Θ` and `W` block-circulant ⇒ the loss decouples across the `d` DFT frequencies.
With `F` the unitary DFT (`F[j,l]=e^{-2πi jl/d}/√d`), `f_f` its `f`-th basis column, and the
`d×d` blocks of the **noise-marginalised Stein covariances** `Σ_φ` (`k×k`) and
`Cov(x0,φ)` (`d×k`):

```
P_f ∈ C^{c×c},  (P_f)_{ab} = f_f^H Σ^{(a,b)}_φ f_f          (Hermitian, +λI on the diag)
q_f ∈ C^c,      (q_f)_a    = f_f^H Σ_{φ_a, x0} f_f
```

the optimal block-circulant readout gives

```
L^circ = Tr(Σ_p0) − Σ_{f=1}^{d} q_f^H P_f^{-1} q_f .
```

This is the exact minimiser of the block-circulant-W loss on block-circulant features. In
code, `P` and `q` are built with a single `einsum('fp,apbq,fq->fab', F, Σ_φ.reshape(c,d,c,d), conj(F))`
(and the analogous `Q`), then a batched `solve` over the `d` frequencies.

**Validation.** `core/rf_circulant._selftest()` reconstructs the *actual* block-circulant
`W` from real degrees of freedom (the `c` first-rows), minimises the quadratic loss
`Tr(Σ_p0) − 2Tr(W Cov^T) + Tr(W Σ_φ W^T)` by a direct linear solve, and checks it equals
the frequency-domain `L^circ`. Match is machine precision (`|diff| ~ 1e-16`, both
unconditional and conditional). CPU vs. CUDA parity is also `~1e-16`.

The covariances `Σ_φ`, `Cov(x0,φ)` are the *same* Stein estimator used for the dense curve
(`core/rf_gmm_estimators.stein_covariances`): real samples for the data expectation, exact
Hermite/Mehler series for the noise (with the exact `E[relu²]` diagonal). So `L^circ(k)` and
`L^dense(k)` differ only by (i) the circulant vs. dense projection and (ii) circulant vs.
unconstrained `W` — nothing else.

Only `k` divisible by `d` (whole `d×d` blocks) is meaningful; the driver emits `NaN` (a gap
in the curve) for other `k`.

## 3. Section 4 — when does L^circ beat L^dense?

`scripts/rf_circulant_win.py` sweeps `k` (divisible by `d`) on a large "population" sample
of the GMM d=32, averaging both losses over `N_REP` independent projection draws, and forms
the writeup's decomposition

```
L^circ(k) − L^dense(k) = A_circ(k) + Δ_stat(σ) − A_dense(k)
  A_dense(k) = L^dense(k) − MMSE(p0)
  A_circ(k)  = L^circ(k)  − MMSE(p̄0)
  Δ_stat(σ)  = MMSE(p̄0) − MMSE(p0)          (stationarisation penalty, k-independent)
```

`p̄0` is the **cyclic-shift symmetrisation** of the data (`x̄0 = S^T x0, T~Unif{0..d-1}`),
realised exactly as a `C·d`-component GMM `{(P_τ μ_c, P_τ Σ_c P_τ^T, w_c/d)}`; both
`MMSE(p0)` and `MMSE(p̄0)` are the population Bayes MMSE via `GaussianMixture.mmse_uncond_exact`.

The **win set** is `{k : L^circ(k) < L^dense(k)}`, equivalently where
`A_dense(k) − A_circ(k) > Δ_stat(σ)`. Asymptotics: `k→∞` sends both `A→0`, so the difference
`→ Δ_stat > 0` (dense wins); a win can only open at small/intermediate `k` where the dense
excess risk `A_dense` blows up faster than `A_circ`.

**Result for the current GMM (d=32).** This GMM is strongly **non-stationary** (means and
covariances concentrated on the first few coordinates), so `Δ_stat` is sizeable and no win
window appears — `L^circ` sits above `L^dense` at every `k` (see `figures/rf_circulant_win.png`
and the printed win set). The machinery is general: a win region is expected to open for
data that is closer to shift-stationary (smaller `Δ_stat`). The figure's bottom row plots
`A_dense − A_circ` against the `Δ_stat` threshold so the (non-)crossing is explicit.

## 4. N_train figures

`scripts/rf_gmm_finite_sample.py` adds a `RF circulant L^circ` curve (dark-violet squares,
plotted only at `k` divisible by `d`) alongside the dense `RF (Stein)` curve, in both the
unconditional and conditional rows, for each `N_train`.
