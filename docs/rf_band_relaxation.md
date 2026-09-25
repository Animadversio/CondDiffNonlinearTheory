# The readout relaxation we priced on the wrong axis

`docs/rf_blockdiag_derivation_and_toll.md` §5 concluded: *"There is no cheap interior point
on CIFAR: to remove the toll you have to break equivariance almost completely."*

That conclusion is about the **axis that was priced**, not about the toll. Period-`P` is the
only relaxation that was tried, and it is close to the worst one available.

**Status 2026-09-22 (evening):** measured on CIFAR by
`scripts/rf_band_relaxation.py` → `tables/rf_band_relaxation_atlas.npz`. The mechanism
holds. Two claims in the first version of this doc were wrong and are corrected in §3 and
§5 rather than quietly deleted.

---

## 1. Why period-`P` is the wrong dial

`A` commutes with `S^P` ⇒ the kernel may depend on position **mod P** ⇒ in frequency it
couples output `f` to inputs `f + j·(d/P)` — a **comb** with teeth `d/P` apart. Spelled out:
period-`P` buys the filter the right to vary *rapidly* in space, at period `P` pixels.

CIFAR's non-stationarity is the opposite kind: objects centred, borders unlike interiors,
sky above ground — **smooth, whole-image scale.** §5 of the toll doc states this and then
prices the comb anyway.

The tell is in that table. `period2d p=16` frees the kernel at *every* position except a
2×2 quadrant tie — 2,359,296 readout parameters — and still pays `+3.341` of the `+6.469`
toll at σ=1.61. **More than half the toll lives in the single coarsest spatial mode**, which
is exactly the mode a comb reaches last and a band reaches first.

The search was over **subgroups of `Z_d`**. The relaxation that matches smooth
non-stationarity is not a subgroup, so it was never a candidate.

## 2. The band

```
A = Σ_{s: |s_r|,|s_c| ≤ B}  diag(e_s) · BCCB(a_s),      e_s[p] = exp(2πi⟨s,p⟩/32)
```

"a convolution whose kernel varies smoothly across the image, band-limited to `B` spatial
modes." In frequency, output `f` reads inputs `f+Δ` for `|Δ| ≤ B`.

Every model in this project is a truncation of the **discrete Weyl expansion**
`A = Σ_{s,m} α[s,m] diag(e_s) S^m`, keeping all lags `m`:

| model | `s` kept | `W` params |
|---|---|---|
| circulant (current) | `s = 0` | `cd` |
| period-`P` | `s ∈ {0, d/P, 2d/P, …}` — a **subgroup** | `cdP` |
| band-`B` | `|s| ≤ B` — a **neighbourhood of 0** | `cd(2B+1)²` |
| free `W` | all `s` | `cd²` |

That table is the argument in one line: the mass sits near `s=0` because the
non-stationarity is smooth, and a subgroup is forced to jump straight to `s = d/P`.

**The fast path survives.** Each parameter `a_s[g]` enters exactly one output frequency
`f = g+s`, so the normal equations still decouple — into 1024 blocks of size `3(2B+1)²`
instead of 3. A *third* route to decoupling: equivariance gives it by symmetry,
block-diagonality by direct sum, this by **partition of the parameters**, with no group.

`B=0` is exactly `period2d(p=1)`; the script asserts it (7e-14). A stronger assert is
available and should be folded in: the general-`G` solver also reproduces `period2d` at
`p = 1,2,4,8`, which exercises non-trivial `G` against independently written polyphase code.

### Implementation

The identical class is obtained by **modulating the features and changing nothing else**:

```
ψ_{a,r} = m_r ⊙ φ_a ,      m_r = the R = (2B+1)² lowest 2-D Fourier modes
```

then keep the existing block-circulant readout verbatim; `P_f` is the `cR × cR` Gram of the
modulated features. Use this ordering (`Σ_r C_r diag(m_r)`), not `Σ_r diag(m_r) C_r`: only
this one decouples for an arbitrary mask set.

**Storage note.** `P_f[(a,s),(b,s')] = E[φ̂_a[f−s] conj(φ̂_b[f−s'])]` depends on `(s,s')`
only through the **difference** `Δ = s−s'`, so the Stein assembly needs `(4B+1)²` frequency-
offset tensors of shape `(c,c,|G|)`, **not** `R² = (2B+1)⁴`. At `B=1` that is 25, not 81 —
3.2× less than a naive count. 1.7 GB at `c=64`, 27 GB at `c=256`, streamable beyond.

## 3. Measured on CIFAR — and where the first version of this doc was wrong

`tables/rf_band_relaxation_atlas.npz`, excess over free Wiener, `d=3072`, `N=10⁴`:

| model | W params | σ=0.127 | σ=0.452 | σ=1.61 | σ=5.0 |
|---|---|---|---|---|---|
| equivariant (`p=1` = `B=0`) | 9,216 | +0.889 | +3.082 | +6.469 | +7.782 |
| comb `p=2` | 36,864 | +0.838 | +2.802 | +6.262 | +7.724 |
| comb `p=8` | 589,824 | +0.611 | +1.975 | +4.827 | +6.864 |
| comb `p=16` | 2,359,296 | +0.443 | +1.376 | +3.341 | +5.372 |
| band `B=1` | 82,944 | +0.643 | +2.094 | +4.065 | +3.592 |
| band `B=2` | 230,400 | +0.526 | +1.615 | +3.050 | +2.513 |
| **band `B=3`** | **451,584** | +0.451 | **+1.303** | **+2.343** | **+1.824** |
| band `B=4` | 746,496 | +0.396 | +1.081 | +1.882 | +1.433 |

**Correction 1 — the B=1 headline was wrong.** The first version claimed "band `B=1` beats
comb `p=16` with 28× fewer parameters." On CIFAR that holds **only at σ=5.0**; at σ ≤ 1.61
comb `p=16` is absolutely lower, at 28× the parameter cost. The claim that survives at every
σ is per-parameter, and the clean absolute cell is `B=3`:

- **Per million readout parameters**, toll removed: `B=1` gives 3.0 / 11.9 / 29.0 / 50.5
  against `p=16`'s 0.2 / 0.7 / 1.3 / 1.0 — a factor **16× / 16× / 22× / 49×**.
- **Band `B=3` matches or beats comb `p=16` at every σ with 5.2× fewer parameters**
  (+0.451 vs +0.443 — a tie; then −0.073 / −0.998 / −3.548).

So: *band dominates comb per parameter everywhere, and dominates in absolute toll from
`B ≈ 3` on.* Not "`B=1` dominates".

**Correction 2 — the surrogate was uncalibrated, and optimistic.** §4 of the first version
predicted `B=1 → +2.5…+3.5` at σ=1.61; the measurement is `+4.065`. The surrogate's own
equivariant toll is 1.517/4.869/13.515/20.190 against CIFAR's 0.889/3.082/6.469/7.782 —
**1.7× too non-stationary at σ=0.127 rising to 2.6× at σ=5.0.** The band's payoff scales
with how much smooth non-stationarity there is to buy, so the prediction was pre-inflated by
construction. The methodological error was reading effect sizes off a surrogate whose
*equivariant toll had not been matched to CIFAR's first*. Any future surrogate in this repo
should be calibrated on that one number before anything is read off it.

The stated falsification bar was not triggered (`B=1` above `p=8`'s `+4.827` ⇒ hypothesis
dead); `+4.065 < +4.827`, so the mechanism survives — but it survives at roughly 60% of the
predicted effect size.

## 4. What it does to the dense crossover

`circ_floor = Wiener + toll − gain`, with the measured nonlinear gain
(1.144 / 1.143 / 0.157 / 0.031) **assumed to transfer** (see §5). Crossover `k/d` at which
dense overtakes, read off `tables/rf_pixel_dense_sweep.npz`:

| | σ=0.127 | σ=0.452 | σ=1.61 | σ=5.0 |
|---|---|---|---|---|
| current (Z_3072) | 2.3 | <2 | <2 | <2 |
| 2-D re-index (`B=0`) | 3.0 | 2.5 | <2 | <2 |
| `B=1` | 3.4 | 3.4 | 2.6 | 3.5 |
| `B=2` | 3.6 | 4.0 | 3.4 | 4.5 |
| `B=3` | 3.7 | 4.7 | 4.1 | 6.0 |
| `B=4` | 3.8 | 5.2 | 4.8 | 7.7 |

At `B=4` — 746,496 readout parameters, still 12.6× fewer than a free `W` — the crossover
roughly doubles to quadruples. It does not go to infinity and never can: `Δ_stat ≥ 0` means
that at `k→∞` the constrained model can at best tie.

Against the free Wiener denoiser rather than dense, the picture is narrower: beating linear
needs `toll < gain`, which the 2-D re-index alone achieves at σ=0.127 (0.889 < 1.144) and
`B=4` achieves at σ=0.452 (1.081 < 1.143, margin 0.062). At σ ≥ 1.61 the gain is 0.157 and
no reachable `B` gets there.

## 5. The load-bearing assumption, and the experiment that settles it

**Correction 3 — the first version mixed two groups**, combining the `Z_3072` toll `+1.273`
with band numbers computed in `Z_32×Z_32` + free channel mixing, whose `B=0` is `+0.889`.
Redone consistently, the conclusion holds and is *cheaper* than claimed: `B=0` alone gives
`7.900 + 0.889 − 1.144 = 7.645` against Wiener's 7.900 (margin −0.26), and `B=1` gives
`7.399` (margin −0.50). **The re-indexing alone predicts the first outright win; the band
widens it.**

Every number in §4 and the paragraph above rides on one untested assumption: that the
nonlinear gain of 1.144 — measured against the `Z_3072` linear class — **transfers to a
larger linear class.** It generally should not, in full: enlarging the linear class lets the
linear model absorb part of what relu was buying, so the residual gain shrinks. How much it
shrinks is exactly what decides §4.

The experiment: run the band-modulated nonlinear RF and measure
`gain(B) = linear_floor(B) − L_RF(c, B)` directly.

Design requirements:

1. **`B=0` at the same `c` is mandatory as a control.** Without it, a shrinking gain cannot
   be separated from a width effect, since the pixel flatness evidence bottoms out at
   `c=1536` and says nothing about `c=64`. The estimand is the *ratio* `gain(1)/gain(0)` at
   matched `c`, which is insensitive to whether that `c` is on the floor.
2. **Run the 2-D `B=0` re-index at `c=1536` first**, against the existing 1-D numbers
   (8.077 at `c=1536`, 8.029 at `c=6144`). That tests the *other* transfer — across groups —
   needs no mask code at all, and is the larger half of the predicted win. If the gain does
   not survive re-indexing, the band question is moot.
3. **σ = 0.127 and 0.452, not 1.61.** At σ=1.61 the gain is 0.157, so "did it shrink" is a
   question about a quantity already near seed noise. At σ ≤ 0.452 the gain is ≈1.14 — 7×
   the signal. σ=1.61 is worth doing afterwards, not first.
4. Report `L_RF`, the band linear floor, and the gain, each at every `(c, B)` cell, so the
   ratio is readable without re-deriving it.
