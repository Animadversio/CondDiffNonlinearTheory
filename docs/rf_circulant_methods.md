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
window appears — `L^circ` sits above `L^dense` at every `k`. The machinery is general: a win
region is expected to open for data that is closer to shift-stationary (smaller `Δ_stat`).
The figure's bottom row plots `A_dense − A_circ` against the `Δ_stat` threshold so the
(non-)crossing is explicit.

**Effective-dimension sweep (`M_ACTIVE`).** To test whether spreading the structure over
more coordinates — *without* stationarising (which would trivially favour circulant) — opens
a win, `make_gmm_active(m_active)` builds a non-stationary GMM whose mean + anisotropy
structure is axis-pinned to the first `m_active` coords, **normalised so the total
mean-separation and anisotropic-variance budgets are `m`-independent** (isolating effective
dimension from raw signal power). Set `M_ACTIVE=3,8,14,20,26` to sweep; the driver emits a
per-`m` detailed figure plus a trend figure `figures/rf_circulant_win_trend.png`
(`min_k(L^circ−L^dense)` and `Δ_stat` vs `m_active`). Note `m_active` is **not** the number
of components: `C=3` fixes the number of modes, and the component means span only ≤`C−1=2`
dims, but the covariances are unconstrained and carry the `m_active`-dim anisotropy that sets
the RF approximation cost.

**Filter bandwidth sweep (`W_BAND`) — locality.** The full-width kernel `h_a ~ N(0,I/d)` is
shift-equivariant but **not local**: every one of the `d` coordinates is weighted equally, so
there is no "emphasise a neighbourhood" inductive bias. `build_circulant_theta(..., w=)`
instead supports a **banded** kernel — `h_a` supported on its first `w` entries with
`h_a[:w] ~ N(0, I/w)`, zeros elsewhere — the true conv-layer analogue (each feature reads a
length-`w` window). Variance `1/w` (not `1/d`) keeps `E‖h_a‖² = 1`, so row norms and the
pre-activation scale match the dense RF at every `w`. Only `L^circ` depends on `w`;
`L^dense`, `MMSE(p0)`, `MMSE(p̄0)` and `Δ_stat` are all `w`-independent.

Finding — **locality opens genuine win regions**. Best bandwidth is `w=1` in every
(σ, m_active) cell, and the gap grows monotonically with `w` (full width `w=d` is the worst
case). At σ=0.5, `min_k(L^circ − L^dense)`:

| m_active | w=1 | w=2 | w=3 | w=4 | w=8 | w=32 |
|---|---|---|---|---|---|---|
| 3  | **−0.317** | +0.386 | +0.332 | +0.298 | +0.600 | +0.668 |
| 20 | **−1.030** | **−0.617** | **−0.066** | **−0.097** | +0.366 | +0.456 |

Wins sit at small `k/d ≈ 1–4` — exactly the §4 prediction (circulant near its `MMSE(p̄0)`
floor while dense is still far from `MMSE(p0)`); by `k/d ≳ 10` dense overtakes and the
difference → `+Δ_stat`. Since `Δ_stat` is `w`-independent, the entire locality gain is
`A_circ` shrinking: a banded filter reaches its stationary floor with far fewer features,
each having to resolve only a length-`w` window instead of all `d` coordinates. Locality and
effective dimension **compound** — `m=20, w=1` (−1.03) is far deeper into the win than either
`m=3, w=1` (−0.32) or `m=20, w=32` (+0.46). Wins remain confined to low σ: at σ≥1 even `w=1`
stays positive because `Δ_stat` grows with σ faster than the locality gain.

Finding (full-width kernel): raising `m_active` 3→26 **monotonically narrows** the circulant gap
`min_k(L^circ−L^dense)` (e.g. σ=1: 2.28→1.42; σ=2: 1.78→1.17) — the dense model loses
relative ground as its target denoiser function becomes higher-dimensional and needs more
features — but it does **not** cross zero: under fixed signal power `Δ_stat` stays sizeable
(~0.3/1.7/2.3/0.9 at σ=0.5/1/2/5) and the circulant floor `MMSE(p̄0)` remains above
`MMSE(p0)`. So effective dimension pushes toward a win but is not sufficient alone; the
stationarisation toll `Δ_stat` (a covariance-geometry property) must also be small.

## 4. N_train figures

`scripts/rf_gmm_finite_sample.py` adds a `RF circulant L^circ` curve (dark-violet squares,
plotted only at `k` divisible by `d`) alongside the dense `RF (Stein)` curve, in both the
unconditional and conditional rows, for each `N_train`.
