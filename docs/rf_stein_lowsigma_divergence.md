# Divergence Investigation: RF-linear MMSE estimators disagree at low σ

**Setting.** Finite dataset of `N=16` atoms treated as the target distribution; ReLU
random features `φ(y)=relu(Θy)`, `k=2048`, `d=8`. We estimate the best-linear-readout
("RF-linear") denoiser MMSE
`L = min_{W,b} E_{x0~emp, z}||W φ(x0+σz)+b−x0||²`.

**Symptom.** At low σ the estimators disagreed by >10× — Stein/`rf_fit_analytic_risk`
reported ~0.55 while the MC estimator reported ~0.06 (σ=0.2). "MC below Stein" looks
impossible if Stein is the true minimum-over-W risk.

## TL;DR (the resolution)

The divergence is **primarily a normalization bug**, not the spectrum:
`stein_covariances` computed `trace_p0` with `/(N-1)` while its covariances `Cov, Σφ`
use `/N`. The mismatch adds a spurious, **σ-independent** offset
`≈ Tr(Σp0)·(1/(N-1) − 1/N) = 0.5169` to the MMSE. It is catastrophic where the true MMSE
is tiny (low σ, small N) and minor where it is large. **Fix:** use `/N` throughout.
After the fix Stein reports the true population RF-linear MMSE (0.033 at σ=0.2), which is
≤ the brute-force MC of a fitted head, as it should be.

The super-singular spectrum of Σφ is **real but was a red herring** for the 0.55 gap — it
causes only a *secondary*, smaller inaccuracy (see §5).

> **Honesty note.** I got the mechanism wrong three times before the brute-force test:
> (i) "fixed ridge over-regularizes" — wrong (Stein flat in λ);
> (ii) "add Hermite terms" — wrong (series already converged at n≤3);
> (iii) "ill-conditioned Σφ makes the collapsed formula diverge from the achieved risk" —
> wrong as the *primary* cause. The actual primary cause is the `/(N-1)` vs `/N`
> mismatch, which I had flagged generically in the first code review and then lost track of.
> Only brute-force MC + the side-by-side below is trustworthy.

---

## 1. What was ruled out (with data)

- **Not the ridge λ.** Sweeping Stein's ridge 1e-4 → 1e-12 barely moves it
  (σ=0.2: 0.554 → 0.550).
- **Not Hermite truncation.** Numerical Gauss–Hermite coefficients up to n=30 do not close
  the gap; the series is converged by n≤3 (`n3:0.550 … n30:0.553`).
- **Not `solve` vs `eigh`.** For identical covariances and λ, both give the same value.

## 2. Decisive evidence #1 — the risk formula is exact

For a **fixed** ridge head `W_λ = Cov(Σφ+λI)^{-1}` from exact empirical covariances,
the covariance-formula risk `R(W)=Tr(Σp0)−2Tr(WCovᵀ)+Tr(WΣφWᵀ)` matches a formula-free
brute-force MC (independent noise) to ≤0.003 at every λ. So the risk *definition* is not
in question. The minimum achievable risk (honest opt-λ, independent val/test, large
samples) is **0.037** at σ=0.2.

## 3. Decisive evidence #2 — the smoking gun (normalization)

Holding the covariances fixed and only varying the `trace_p0` normalization:

```
trace_p0:  /(N-1)=8.2702   /N=7.7533   difference = 0.5169
σ=0.2:  explained(/N covs)=7.7200
        Stein as-is (tr=/(N-1)) = 8.2702−7.7200 = 0.5502   == stein_finiteN_mmse
        Stein fixed (tr=/N)     = 7.7533−7.7200 = 0.0333   ≈ brute-MC 0.037
```

The entire 0.55 is `0.517 (offset) + 0.033 (true residual)`. The offset is exactly
`Tr(Σp0)/(N(N-1))·N = trace_p0_{N-1} − trace_p0_N`.

**Why σ-dependent in appearance:** the offset is constant (0.517), but the *true* MMSE
grows with σ, so the relative error shrinks: 15× at σ=0.2 (true 0.033), ~1.06× at σ=2
(true ~4.8). This matches the observed pattern exactly.

## 4. Fix + re-validation

`trace_p0 = sum(X0_c²)/N` (was `/(N-1)`). `/N` is correct for the empirical-distribution
MMSE (uniform weight over the N atoms), and matches the `/N` covariances and the brute-MC.
After the fix:

| σ | Stein (fixed) | rf_analytic | brute-MC (fitted head) |
|---|---|---|---|
| 0.2 | 0.033 | 0.053 | 0.042 |
| 0.5 | 0.287 | 0.556 | 0.461 |
| 1.0 | 1.733 | 2.383 | 2.204 |
| 2.0 | 4.811 | 5.176 | 5.001 |

Stein (the population-optimal-W MMSE) is now **≤ brute-MC (a fitted head)** at every σ,
which is the correct ordering. The gap = finite-sample estimation error of the fitted head
(small at low σ where interpolation is easy, larger at high σ).

## 5. Secondary effect — the spectrum really is singular

`Σφ` is genuinely super-singular at low σ (validated; see
`figures/rf_stein_spectrum_diagnosis.png`):

| σ | cond(Σφ) | effective rank (participation ratio) |
|---|---|---|
| 0.2 | 1.6e8 | 6.1 |
| 0.5 | 4.2e6 | 7.6 |
| 1.0 | 1.6e6 | 10.8 |
| 2.0 | 9.2e5 | 13.5 |

- **Not "dead features."** 0% of features are always-off or always-on; all 100% are mixed.
  The singularity comes from having only `N=16` atoms (noise-averaged features span ≤ N−1
  directions, energy concentrated in ~6), plus a σ²-scaled noise tail filling the rest.
- The cross-covariance `Cov(x0,φ)` energy sits mostly on the *large*-λ directions
  (a_i ∝ λ_i), so the explained variance comes from the top ~15 directions; the tiny-λ tail
  is essentially signal-free noise.
- **Consequence (after the /N fix): the collapsed `L = Tr(Σp0) − Tr(Cov Σφ⁻¹ Covᵀ)` can
  UNDERestimate and dip below the Bayes floor** when Σφ is ill-conditioned (small N / low σ /
  conditional). The /N fix removed the *over*estimate offset, exposing this *under*estimate.
  Worst in the conditional case:

  | (N=16, k=2048, cond) | stein (collapsed) | brute-MC RF-cond (truth) | NW-cond floor |
  |---|---|---|---|
  | σ=1 | 0.968 (below floor) | 1.303 | 0.983 |
  | σ=2 | 2.531 (below floor) | 3.058 | 2.840 |

  Empirically this is in-sample (plug-in) optimism of the collapsed form: fully inverting the
  ill-conditioned Σφ over-fits its small-eigenvalue tail. Rank-truncating the inverse (top
  ~100 of 2048 eigenvalues) recovers the out-of-sample value (L ≈ 1.38 ≈ brute-MC 1.30);
  full inversion drives it to 1.08 (Hermite: 0.968, below the floor). No hand-wavy mechanism
  beyond that is asserted here.

- **A closed-form fix does not exist** — the ridgeless MMSE of a singular Σφ is ill-posed.
  In particular, `min_λ [Tr(Σp0) − 2Tr(W_λ Covᵀ) + Tr(W_λ Σφ W_λᵀ)]` with the *optimal*
  `W_λ = Cov(Σφ+λI)⁻¹` (which I proposed in an earlier draft) is **also unstable** — the
  ridgeless optimum has huge norm and the estimate diverged to −47 at σ=2. **The reliable
  estimator is `rf_fit_analytic_risk`**, which evaluates the achieved risk of a *bounded,
  finite-sample-fitted* head (naturally regularized): it matches brute-MC and stays ≥ the
  Bayes floor at every σ tested (table above: rf_analytic = 1.41, 3.23 ≈ brute 1.30, 3.06).

## 6. What I am sure of / not sure of

**Sure (brute-force verified):**
- The risk formula `R(W)` is exact (matches MC for all W).
- The primary bug is the `trace_p0` `/(N-1)` vs `/N` mismatch; it accounts for the 0.517
  offset exactly and is fixed by `/N`.
- After the fix, Stein ≤ brute-MC (correct ordering) at all σ tested.
- The true RF-linear MMSE at σ=0.2, N=16, k=2048 is ~0.03–0.04 — **below** linear Wiener
  (0.29). So the RF *does* beat linear Wiener at low σ (the MC/interpolation win was real).
- Σφ is super-singular at low σ (cond 1.6e8, eff-rank ~6), but this is a *secondary* effect.

**Not sure / open:**
- Whether `/N` or `/(N-1)` should be standardized codebase-wide (JG theory and
  `wiener_emp` still use `/(N-1)`; they should match — a 7% cross-estimator inconsistency
  remains).
- The exact size of the secondary collapsed-formula inaccuracy vs σ and cond(Σφ); a stable
  achieved-risk estimator (§5) is proposed but only spot-checked, not swept.
- The conditional (U=class) case was not re-tested after the fix.
- The high-σ finite-`n_fit` bias of the fitted-head estimators (`rf_optridge`,
  `rf_analytic`) was not characterized.

## 7. Recommendations / impact

1. **Done:** `trace_p0 = /N` in `stein_covariances` (fixes `stein_finiteN_mmse` and, via it,
   `rf_fit_analytic_risk`).
2. **Standardize normalization to `/N`** across `mmse_theory_joint_gaussian` (driver's
   `Sig_p0_emp`) and `wiener_emp` so all estimators are mutually consistent.
3. **For the ill-conditioned regime, use `rf_fit_analytic_risk` (a bounded fitted head), not
   the collapsed-L `stein`.** The collapsed form is an idealized lower bound that is
   numerically unreliable when Σφ is ill-conditioned (small N / low σ / conditional) and can
   dip *below* the achievable Bayes floor. `rf_analytic` matches brute-MC and respects the
   floor. (An earlier draft recommended `min_λ R(W_λ)` with the optimal ridgeless head —
   that is also unstable; do not use it.) Treat the stein↔rf_analytic gap as a *diagnostic*
   of Σφ's conditioning rather than a trustworthy MMSE at low σ.
4. **Impact on committed results:** the tables/figures in
   `tables/rf_gmm_finite_sample_N*.npz` and the corresponding figures used the buggy
   `/(N-1)` Stein/analytic — the Stein/analytic curves are overestimated at small N + low σ
   by ~Tr(Σp0)/N. They should be regenerated after the fix (and the JG/wiener normalization
   aligned).

## Appendix: reproduction

Throwaway scripts on an H100 node (numpy/CPU suffices at k=2048): (i) exact empirical
`Cov, Σφ` from ~80k noisy samples; (ii) `R(W)` formula vs chunked brute-force MC on
independent noise; (iii) the `trace_p0` side-by-side; (iv) spectrum/eff-rank via `eigvalsh`.
Estimators: `core.rf_gmm_estimators.{stein_finiteN_mmse, rf_optridge_mmse,
rf_fit_analytic_risk}`.
