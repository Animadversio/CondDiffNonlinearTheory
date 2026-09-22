# Block-diagonal Θ and W: the loss formula, and why the model is worse than what we have

Answers the 2026-09-22 request: *"re-derive the circulant loss formula for where Θ is 8×8
(or whatever tap number) circulant along the diagonal and the rest is 0, and readout W is
also this structure"*, plus *"what in this is equivariance?"* and the objection that a
circulant `W` cannot be what is holding the model back because at matched free rows it has
as many non-repeated parameters as a dense free `W`.

Three results, in order of how much they change the plan:

1. **I was wrong in `rf_circulant_relaxations.md` §2(b).** This model *does* have a fast
   path, and it is *cheaper* than the one we run today. §1 derives it.
2. **But the model is excluded by measurement.** Even a *perfect nonlinear* denoiser with
   this structure loses to the plain linear denoiser by `+8.6 … +97.6`, where the circulant
   RF we already have loses by only `+0.13 … +8.97`. §3.
3. **The parameter-count objection is right, and the readout is still the binding
   constraint.** Both can be true because the constraint is on the *function class*, not
   the parameter count — and it is now measured exactly on raw pixels, with no RF model and
   no GMM proxy: forcing the readout to be equivariant costs `+1.27 / +4.07 / +8.47 / +9.01`
   at `σ = 0.127 / 0.452 / 1.610 / 5.0`, which is **the entire deficit of the circulant RF**.
   §4–5.

Everything numeric here is reproduced by `scripts/rf_equivariance_toll.py` (~90 s, one GPU)
into `tables/rf_equivariance_toll.npz`.

---

## 1. The derivation

Let `d = 3072`, block size `b` (`b = 8` in the request), `M = d/b = 384` blocks. Write
`y_j ∈ R^b` for the `j`-th contiguous chunk of `y`.

**Design.** For channel `a ∈ {1..c}`, `Θ^{(a)}` is block-diagonal with `b×b` circulant
blocks `C_{a,j} = circ(h_{a,j})`, `h_{a,j} ∈ R^b` independent across `(a,j)`. Features

```
φ_{a,j} = relu( C_{a,j} y_j ) ∈ R^b
```

**Readout.** `W^{(a)}` has the same structure, blocks `D_{a,j} = circ(g_{a,j})`, plus the
free per-position bias we already use. The prediction restricted to chunk `j` is

```
x̂_j = Σ_a D_{a,j} φ_{a,j} + β_j
```

**The key step.** `W^{(a)}` is block-diagonal, so chunk `j` of the output depends only on
chunk `j` of the features, and `‖v‖² = Σ_j ‖v_j‖²` splits the loss with no cross terms:

```
L = Σ_{j=0}^{M-1}  E ‖ x0_j − Σ_a D_{a,j} φ_{a,j} − β_j ‖²
```

The `j`-th summand involves only `{D_{a,j}}_a` — **no parameter is shared between chunks**,
so the minimisation separates into `M` independent problems. Cross-chunk feature moments
`E[φ_{a,j} φ_{b,j'}ᵀ]`, `j ≠ j'`, are large on CIFAR and simply never enter, for the same
reason cross-frequency moments never enter today: *the constrained readout cannot use them.*

Now diagonalise within a chunk. Let `F_b` be the unitary `b`-point DFT and
`φ̂_{a,j} = F_b φ_{a,j}`. A `b×b` circulant acts as multiplication by its symbol
`ĝ_{a,j}[f]`, so by Parseval

```
L = Σ_j Σ_{f=0}^{b-1}  E | x̂0_j[f] − Σ_a ĝ_{a,j}[f] φ̂_{a,j}[f] |²
```

(the bias absorbs the means; take all moments centred). Each `ĝ_{a,j}[f]` appears in
exactly one summand, so with

```
P_{j,f} = Cov( φ̂_j[f] ) ∈ C^{c×c},      q_{j,f} = Cov( x̂0_j[f], φ̂_j[f] ) ∈ C^c
```

the optimum is `ĝ_{j,f} = P_{j,f}^{-1} q_{j,f}` and

```
  L^blockdiag  =  Tr(Σ_{p0})  −  Σ_{j=0}^{M-1} Σ_{f=0}^{b-1}  q_{j,f}^H P_{j,f}^{-1} q_{j,f}
```

Identical in shape to the model we run; the single index `f ∈ Z_d` is replaced by the pair
`(j, f) ∈ Z_M × Z_b`. There are `M·b = d` solves of size `c×c` — **exactly the same count
as today.** The Hermitian fold applies within each chunk (`P_{j,b-f} = conj(P_{j,f})`),
leaving `M·(b/2+1) = 1920` solves at `b = 8` versus `d/2+1 = 1537` now.

**It is also strictly easier to assemble.** The expensive part of the current code is the
noise term: the Gaussian noise seen by two features is `σ² ⟨S^r h_a, S^{r'} h_b⟩`, which is
nonzero for every pair and couples all frequencies — that is the `ρⁿ∘(CₙᵀCₙ)` convolution
and the `(c,c,d)` tensor it needs. Here the blocks have **disjoint support**, so that inner
product vanishes whenever `j ≠ j'`, and within a chunk it depends only on the lag `r − r'`.
The noise covariance is block-diagonal across chunks and circulant within one, so it is
diagonalised by the same `F_b`. No cross-frequency convolution at all. Memory is
`(c, c, b)` per chunk — 151 MB at `c = 1536` — and chunks can be streamed in batches, so
the `(nf, c, c)` ceiling that blocks `c = 12288` today does not arise.

So: feasible, same cost, simpler code. My "no group ⇒ 2.6 PB" claim was wrong, and the
reason is worth stating because it generalises: **I looked for a symmetry and missed a
direct sum.** The normal equations decouple whenever the readout's parameter blocks touch
disjoint output coordinates — equivariance is one way to get that, separability is another.

### Validation, and two different "cross-covariances" that must not be conflated

Raised 2026-09-22: *"the cross-variance of noise shouldn't be 0."* There are two distinct
objects here and they behave oppositely.

| | across chunks `j ≠ j'` | enters the optimum? |
|---|---|---|
| **feature** covariance `Cov(φ_{a,j}, φ_{b,j'})` | **nonzero** — `x0` is correlated across chunks | **no** |
| **noise** covariance `σ²(Θ_a Θ_bᵀ)_{[j,j']}` | **exactly 0** — the blocks have disjoint support | n/a |

The noise one is zero as an identity, not numerically: feature `(a,j,r)` sees only
`Z[jb .. jb+b)` and feature `(b,j',r')` only `Z[j'b .. j'b+b)`, and those index sets are
disjoint, so `σ² E[Z_j Z_{j'}ᵀ] = 0`. Verified to `0.000e+00` analytically.

The feature one is genuinely nonzero and it is right to object to any claim otherwise — but
it does not enter, because a block-diagonal `W` cannot use it. That is the same phenomenon
as in the current model, where cross-frequency moments `E[φ̂_a[f] φ̂_b[g]*]`, `f≠g`, are
large on CIFAR and never appear in `L^circ`. It is a property of the model, not of `p(x0)`.

`SELFTEST=1 python scripts/rf_equivariance_toll.py` checks exactly this, on data built to
have strong cross-chunk correlation and non-Gaussian marginals (`d=32, b=4, c=2`): it
builds `Θ` explicitly, solves the structure-constrained least squares by brute force over
all `c·M·b` readout taps plus a free per-coordinate bias, and compares.

```
max |cross-chunk FEATURE covariance|         = 0.101286      (nonzero -- real)
max |cross-chunk NOISE covariance|, analytic = 0.000e+00     (exactly 0)
brute-force constrained optimum              = 8.26177742778661
formula Tr(Sig) - sum_{j,f} q^H P^-1 q       = 8.26177742778656
abs diff                                     = 4.8e-14
```

So the formula is the exact optimum *with* those nonzero cross-chunk correlations present.

If the intent is a model whose noise *does* couple across blocks, then the blocks must not
have disjoint support — either overlapping windows (see the corollary below, which keeps
the fast path), or the `b×b` blocks arranged **circulantly instead of only on the
diagonal**. That second option is block-circulant-with-circulant-blocks, i.e. exactly a 2-D
convolution, i.e. the group `Z_M × Z_b` — the same re-indexing recommended in §6.

### What is equivariance here, concretely

`Θ` and `W` are *equivariant* for a shift `S` when `Θ S = S̃ Θ` and `W S̃ = S W` (`S̃` the
induced shift on features): **translate the input, and the output translates the same
way.** Because `relu` acts pointwise it commutes with `S̃`, so the whole denoiser satisfies
`f(S y) = S f(y)`. That is exactly the property that makes a convolution layer a
convolution layer — weight sharing *is* equivariance, stated as an identity rather than as
a storage trick. So equivariance is not opposed to the CNN structure you want to keep; it
is the name for it.

The two models differ in *which group*:

| | group | acts how |
|---|---|---|
| today | `Z_d`, `d = 3072` shifts | one filter slides over the whole image, wrapping |
| proposed | `Z_b` **per chunk**, `M` independent copies | a filter rotates inside its own 8-pixel chunk |

The proposed model is equivariant only under rotations *within* a chunk, and chunks are
independent. What buys the fast path there is not that group — it is the direct-sum
structure above. Both statements in §1 of the earlier doc ("equivariance is what pays,
sparsity is not") should be read as: *equivariance is one sufficient condition, and
block-separability is another; plain sparsity in `Θ` alone is still worth nothing.*

### A useful corollary

The decoupling used only that **`W`** is block-diagonal. `Θ` is unconstrained in the
argument. So one can keep stride-1 overlapping windows in `Θ` — full feature diversity,
receptive field spilling past the chunk edge — and still get `M` independent solves, as
long as features are *assigned* to one output chunk each. If a tiled model is built at all,
build that variant; it dominates the strict version at identical cost.

---

## 2. What the proposed model gives up

Parameter counts at `c` channels, against dense width `k` (matched at `c = k`):

| | features | `Θ` random params | `W` trained params | readout receptive field |
|---|---|---|---|---|
| dense | `k` | `k·d` = 18.9M | `k·d` = 18.9M | global |
| banded block-circulant (today) | `c·d` | `c·t` = **49k** | `c·d` = 18.9M | **global** (`W` is a full-width `d`-tap circulant) |
| proposed 8×8 block-diagonal | `c·d` | `c·d` = **18.9M** | `c·d` = 18.9M | **8 pixels** |

The diagnosis behind the proposal is correct and the fix works: `Θ`'s random parameters go
from `c·t` to `c·(d/b)·b = c·d`, a 384× increase that lands exactly on dense's count. That
was the real asymmetry and this removes it.

The cost is in the last column, and it is easy to miss because it is invisible in the
parameter counts. **Our current `W` is a full-width circulant, not a banded one** — `c·d`
parameters spent as `c` filters of `d` taps each, so the prediction at position `p` already
depends on every pixel. Making `W` block-diagonal spends the same `c·d` parameters as
`c·M` filters of `b` taps, and the prediction at position `p` now depends on 8 pixels. Only
`Θ` was ever local in our model; this makes the *readout* local too.

In the channel-major raster layout (`index = ch·1024 + row·32 + col`), 8 contiguous
coordinates are **8 horizontally adjacent pixels of a single colour channel**. Such a chunk
contains none of the vertical correlation (offset 32) and none of the cross-channel
correlation (offset 1024), which are the two strongest non-local structures in CIFAR.

---

## 3. The measurement that rules it out

Best *linear* denoiser under each pattern, exact, CIFAR raw pixels `d = 3072`, `N = 10⁴`
(excess over the free Wiener denoiser in parentheses):

```
                          sg=0.127         sg=0.452         sg=1.610          sg=5.0
free Wiener                 7.900           28.552           72.196          130.302
tiled b=8                  19.381 (+11.5)   82.173 (+53.6)  169.169 (+97.0)  188.858 (+58.6)
tiled b=128                13.349  (+5.4)   45.763 (+17.2)  112.852 (+40.7)  172.582 (+42.3)
sliding w=8                17.841  (+9.9)   78.059 (+49.5)  168.332 (+96.1)  188.764 (+58.5)
2-D 5x5, 3 channels         8.538  (+0.6)   34.081  (+5.5)  105.874 (+33.7)  173.993 (+43.7)
2-D 9x9, 3 channels         8.188  (+0.3)   30.526  (+2.0)   84.188 (+12.0)  158.015 (+27.7)
---
block-circulant RF (ours)   8.029  (+0.13)  31.479  (+2.93)  80.511  (+8.32) 139.276  (+8.97)
```

A nonlinear model is not bounded below by the linear one, so the decisive number is the
**Bayes floor of the tiled class** — `Σ_j MMSE(x0_j | y_j)`, the best *any* nonlinear tiled
denoiser can achieve. Computed exactly against the empirical measure (`b = 8`, 2000 noisy
draws per block):

```
 sigma   Wiener   tiled linear   tiled BAYES   BAYES − Wiener   median N_eff
 0.127    7.900        19.381        16.512          +8.612            821
 0.452   28.552        82.173        80.107         +51.555          5,183
 1.610   72.196       169.169       169.832         +97.636          9,400
 5.000  130.302       188.858       190.066         +59.764          9,939
```

`N_eff` is 821–9,939, so this is **not** the `N_eff = 1` memorisation artifact that makes
the `d = 3072` oracle-Bayes curves meaningless — an 8-dimensional marginal is well sampled
by 10⁴ atoms. (At `σ ≥ 1.61` the Bayes estimate sits ~1 above the exact linear value; that
is Monte-Carlo error on 2000 draws, at a noise level where both are within 1% of `Tr Σ`.
It does not affect the conclusion by two orders of magnitude.)

**A perfect nonlinear tiled denoiser loses to plain linear regression by 8.6 to 97.6.** The
model we already have loses by 0.13 to 8.97. The proposal is between 7× and 66× worse in
the quantity we are trying to drive negative, and no feature map or width fixes it, because
the number quoted is the floor of the class.

Two secondary readings, both relevant to design:

- **Overlap is a minor effect; receptive-field size is the whole story.** Sliding beats
  tiling by only `0.1–4.1` at equal width, while going from width 8 to width 128 buys
  `6–56`. The 2026-08-17 permutation test found diversity mattered more than locality *at
  fixed width*; this says width dominates both.
- **If a local model is built, it must be 2-D and must mix colour channels.** At `σ=0.452`,
  a `5×5` patch costs `+17.2` on one channel and `+5.5` across three; `9×9` across three
  costs `+2.0`. The 1-D contiguous window is a poor proxy for a CNN, and our `Θ`'s band is
  currently 1-D contiguous too — see §6.

---

## 4. The parameter-count objection, granted and then sharpened

> *"I don't think circulant W instead of free W is holding circulant model back because in
> the case of matched free rows the circulant W has as much meaningful (not repeated)
> parameters as the dense free W."*

The count is exactly right: at `c = k` both readouts have `c·d = k·d = 18,874,368` genuinely
free, non-repeated parameters. I established that number myself and it is not in dispute.

The inference does not follow, because the two readouts are attached to feature spaces of
very different sizes:

| | features | readout params | **params per feature** | reachable readouts |
|---|---|---|---|---|
| dense | `k` | `k·d` | `d` = 3072 — a free `d`-vector each | all of `R^{d×k}` |
| circulant | `c·d` | `c·d` | **1** | a `c·d`-dim subspace of `R^{d×cd}`, i.e. `1/d` of it |

Equal totals, but the circulant model has `d` times more features and gives each of them
`d` times less freedom. Its readout is confined to a measure-zero slice of its *own* natural
readout space, and the slice is not an arbitrary one: a circulant `W` cannot couple DFT
frequency `f` to frequency `g ≠ f`. Adding features does not enlarge the set of functions
reachable in that direction, which is why the toll in the GMM ablation did not vanish as
`k → ∞`.

Note the proposal does not change this: `c·d` features with `c·d` readout parameters is
still 1 per feature.

---

## 5. The toll, measured exactly — this settles which hypothesis is right

The cleanest version of the question needs no RF model at all. Constrain the **linear**
denoiser to be equivariant and compare to the free one. This is an exact minimum over the
whole equivariant class, computed per DFT character, so it has **no taps to starve** and no
width to run out of:

```
 sigma   free Wiener   equivariant linear   toll     circulant RF   RF − equiv-linear
 0.127         7.900                9.173  +1.273          8.029            −1.144
 0.452        28.552               32.622  +4.070         31.479            −1.143
 1.610        72.196               80.668  +8.473         80.511            −0.157
 5.000       130.302              139.307  +9.005        139.276            −0.031
```

Read the last two columns together:

- **The toll alone accounts for the circulant RF's entire deficit.** At `σ = 1.61` the RF
  loses `8.32` to linear and the equivariance constraint by itself costs `8.47`. At
  `σ = 5.0`: `8.97` versus `9.01`.
- **The RF's nonlinearity is worth `1.14` at low `σ` and `0.03–0.16` at high `σ`** — that is
  how far it gets below the best *linear* equivariant denoiser. At `σ ≥ 1.61` the circulant
  RF, with 18.9M features, is within `0.16` of what a single equivariant linear filter
  achieves.

So hypothesis (B) is confirmed and (A) is not the binding constraint. Tap renewal acts on
`Θ`; the toll is a property of `W`'s symmetry and is present at the exact optimum over
every equivariant map, whatever `Θ` is.

**How much does relaxing the readout buy?** Let `A` commute with `S^P` (`P = 1` today,
`P = d` free). This prices period-`P` sharing before any RF code is written:

```
     P    W params   sg=0.127   sg=0.452   sg=1.610    sg=5.0        (excess over Wiener)
     1       3,072     +1.273     +4.070     +8.473    +9.005
     8      24,576     +1.141     +3.743     +8.220    +8.932
    32      98,304     +0.884     +2.956     +6.671    +7.263
   128     393,216     +0.676     +2.279     +5.922    +6.993
   512   1,572,864     +0.464     +1.260     +3.458    +4.854
  1024   3,145,728     +0.222     +0.379     +0.782    +0.820
  3072   9,437,184      0          0          0         0
```

**This is a correction to my own recommendation.** I proposed period-`P` as the dial to run,
expecting a useful interior point. The curve is nearly flat through `P = 8` (`8.47 → 8.22`
at `σ=1.61`, for 8× the readout parameters) and only collapses at `P = 1024` — which is
`d/3`, i.e. the point where the filter may depend on spatial position and is tied only
across the three colour planes. The non-stationarity that matters is at the scale of the
whole `32×32` image (objects centred, borders unlike interiors), not at any short period.
The 2-D version behaves the same way (`p=1: +6.47`, `p=8: +4.83`, `p=16: +3.34`, `p=32: 0`
at `σ=1.61`). **There is no cheap interior point on CIFAR: to remove the toll you have to
break equivariance almost completely.**

---

## 6. Where this leaves the plan

- **Do not build the tiled 8×8 model.** §3 — its floor is 7–66× worse than what we have.
- **Do not build period-`P` expecting a win.** §5 — priced, and it is flat where it is
  affordable. `P = 1024` at `c = 32` is the only cell that would move the needle
  (block size `cP = 32768`, 3 coarse frequencies, ~52 GB) and it is barely a constraint
  any more.
- **Do not build the free-`W` ablation.** It was my recommendation, and §5 supersedes it:
  it was a proxy for exactly this question, on a GMM, at `c ≤ 8`, and the direct
  measurement on raw pixels is better in every respect.
- **The one thing still worth running is the `t`-sweep — but at low `σ`, not high.** This
  reverses what I offered on 2026-09-21. The margin the model must close is the toll minus
  the nonlinear gain: `1.27 − 1.14 = 0.13` at `σ=0.127`, versus `8.47 − 0.16 = 8.31` at
  `σ=1.61`. Only the low-`σ` end is within reach of any improvement to `Θ`, and it needs a
  ~1.1× improvement in the nonlinear gain rather than a ~50× one. Run `t ∈ {8,32,128}` at
  `c=1536`, `σ ∈ {0.127, 0.452}`. Standing evidence is still against the premise (the July
  `w`-sweep had the gap growing monotonically in `t`, full width worst), so this is a
  falsification test, not an expected win.
- **Cheap and likely worth more than any of the above: change the group from `Z_3072` to
  `Z_32 × Z_32` with free 3-channel mixing.** Our 1-D raster circulant treats the image as
  a ring, so a `t=8` band is 8 horizontal pixels of one colour plane. The 2-D group is the
  honest CNN and the toll is lower for free: `+0.889 / +3.082 / +6.469 / +7.782` versus
  `+1.273 / +4.070 / +8.473 / +9.005`, i.e. `0.4 / 1.0 / 2.0 / 1.2` recovered by
  re-indexing. The machinery is unchanged — every finite abelian group has a DFT; it is
  `rfft2` over a `(c, 3, 32, 32)` layout instead of `rfft` over `(c, 3072)`, with per-
  frequency blocks of size `3c` over `1024` frequencies. Much smaller change than any
  redesign discussed so far, and it strictly improves the class.

### Honest limits of §5

The equivariant *linear* denoiser is not a lower bound for the equivariant *nonlinear* RF —
the RF beats it by `1.14` at `σ=0.127`. A better `Θ` could in principle widen that gain, and
the rigorous floor would be the equivariant *Bayes* denoiser, which we cannot estimate on
CIFAR at `d=3072` (`N_eff = 1`). So §5 is strong evidence, not a proof, that tap renewal
cannot close the gap. The argument it does license is quantitative: the required improvement
in the nonlinear gain is ~1.1× at `σ=0.127` and ~50× at `σ=1.61`, and that is what makes the
low-`σ` `t`-sweep the only version of the test worth running.
