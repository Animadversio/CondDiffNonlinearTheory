# The readout relaxation we priced on the wrong axis

`docs/rf_blockdiag_derivation_and_toll.md` §5 concluded: *"There is no cheap interior point
on CIFAR: to remove the toll you have to break equivariance almost completely."*

That conclusion is about the **axis that was priced**, not about the toll. Period-`P` is the
only relaxation that was tried, and it is close to the worst one available.

---

## 1. Why period-`P` is the wrong dial

`A` commutes with `S^P` ⇒ the kernel may depend on position **mod P** ⇒ in frequency it
couples output `f` to inputs `f + j·(d/P)` — a **comb** with teeth `d/P` apart. Spelled out:
period-`P` buys the filter the right to vary *rapidly* in space, at period `P` pixels.

CIFAR's non-stationarity is the opposite kind. Objects are centred, borders are unlike
interiors, sky sits above ground — **smooth, whole-image scale.** §5 states this and then
prices the comb anyway.

The tell is already in the §5 table. `period2d p=16` frees the kernel at *every* position
except a 2×2 quadrant tie — 2,359,296 readout parameters — and still pays `+3.341` of the
`+6.469` toll at σ=1.61. **More than half the toll lives in the single coarsest spatial
mode**, which is exactly the mode a comb reaches last and a band reaches first.

The search was over **subgroups of `Z_d`**. The relaxation that matches smooth
non-stationarity is not a subgroup, so it was never a candidate.

## 2. The band

```
A = Σ_{s: |s_r|,|s_c| ≤ B}  diag(e_s) · BCCB(a_s),      e_s[p] = exp(2πi⟨s,p⟩/32)
```

"a convolution whose kernel varies smoothly across the image, band-limited to `B` spatial
modes." In frequency, output `f` reads inputs `f+Δ` for `|Δ| ≤ B`.

**The fast path survives untouched.** Each parameter `a_s[g]` enters exactly one output
frequency `f = g+s`, so the normal equations still decouple into 1024 independent blocks —
of size `3(2B+1)²` instead of `3`. This is a *third* route to decoupling, alongside the two
already in the repo: equivariance gives it by symmetry, block-diagonality by direct sum,
and this gives it by **partition of the parameters**, with no group at all.

`B=0` is exactly `period2d(p=1)`; `scripts/rf_band_relaxation.py` asserts the match.

### The implementation is smaller than it looks

The identical class is obtained by **modulating the features and changing nothing else**:

```
ψ_{a,r} = m_r ⊙ φ_a ,      m_r = the R = (2B+1)² lowest 2-D Fourier/DCT modes
```

then keep the existing block-circulant readout verbatim. Output `f` of
`Σ_{a,r} W_{a,r} ψ_{a,r}` reads `φ̂[g]` for `g` in a band around `f`, and `ŵ_{a,r}[f]`
belongs to output `f` alone — so `P_f` is just the `cR × cR` Gram of the modulated features.
Per-frequency block goes `c → cR`; no new derivation, no reindexing of the Stein assembly
(the modulation is a fixed diagonal, so the empirical estimator of §3 of
`rf_circulant_struct_methods.md` takes it without changes).

## 3. Measured

Not CIFAR — no cluster access from where this was written. Ensemble: 32×32×3 crops from
real photographs (genuine `1/f` spectrum and chromatic correlation), composited as
*centred object over background* with a per-image gain and a vertical tilt, i.e. CIFAR's
kind of smooth non-stationarity. `Tr(Σ)` normalised to 191.52 and the same σ grid, so the
units are comparable. `N=4000`. Reproduce with `band_vs_comb_surrogate.py`.

**Control — exactly shift-stationary ensemble** (same crops, random cyclic shift applied).
Toll `+0.154` at σ=1.61 and neither dial does much: the diagnostic reads ~0 when there is
no non-stationarity to buy back, as it must.

**Non-stationary ensemble**, excess over free Wiener:

| model | W params | σ=0.127 | σ=0.452 | σ=1.61 | σ=5.0 |
|---|---|---|---|---|---|
| free Wiener | 9,437,184 | 0 | 0 | 0 | 0 |
| **equivariant** (`p=1` = `B=0`) | 9,216 | +1.517 | +4.869 | +13.515 | +20.190 |
| comb `p=2` | 36,864 | +1.257 | +4.404 | +13.342 | +20.139 |
| comb `p=4` | 147,456 | +1.009 | +3.652 | +12.760 | +19.940 |
| comb `p=8` | 589,824 | +0.767 | +2.843 | +11.344 | +19.151 |
| comb `p=16` | 2,359,296 | +0.450 | +1.507 | +7.016 | +13.671 |
| **band `B=1`** | **82,944** | **+0.952** | **+2.738** | **+5.567** | **+7.214** |
| **band `B=2`** | 230,400 | +0.705 | +1.937 | +3.391 | +2.840 |
| **band `B=3`** | 451,584 | +0.567 | +1.464 | +2.458 | +1.966 |

Band `B=1` removes **more** toll than comb `p=16` with **28× fewer** parameters
(7.95 vs 6.50 at σ=1.61; 12.98 vs 6.52 at σ=5.0). Per million readout parameters at σ=1.61:
band `B=1` = 95.8, comb `p=16` = 2.8 — **34×**. Band `B=2` removes 86% of the toll at σ=5.0
for one tenth of comb `p=16`'s budget.

The flatness §5 reported is a property of the comb, not of the problem.

## 4. Prediction for CIFAR

The surrogate's `p=16` signature matches CIFAR's (over half the toll surviving the
quadrant tie), so the mechanism should transfer. At σ=1.61, toll `+6.469`:

- `B=1` (82,944 params) should land near **+2.5 … +3.5**, i.e. below `p=16` at 1/28 the cost.
- `B=2` (230,400 params) should land near **+1.5 … +2.0**.

If `B=1` comes in above `p=8`'s `+4.827`, the band hypothesis is wrong and CIFAR's
non-stationarity is not low-order — worth knowing either way, and it costs 90 seconds.

`python scripts/rf_band_relaxation.py`

## 5. What this does and does not license

It is still a **linear** pricing, so it bounds nothing about the nonlinear RF directly. What
it changes is the arithmetic of §5's verdict. The margin the model must close is
`toll − nonlinear gain`. If `B=1` takes the toll from `+8.47` to `≈+3` (1-D) or `+6.47` to
`≈+3` (2-D), the required improvement in the nonlinear gain at σ=1.61 drops from ~50× to
~20×, and at σ=0.127 the margin (`+1.273 − 1.144 = +0.13`) goes **negative** — the circulant
RF would beat free Wiener at low σ on the readout relaxation alone, with no change to `Θ`.

That is the first configuration in this project where the circulant model is predicted to
win outright.
