# What the fast path actually needs, and how far W can be relaxed

Answers three questions raised 2026-09-21 about relaxing the block-circulant RF denoiser:

1. does a **free readout** `W` destroy the fast path?
2. if `Θ` becomes **8×8 blocks along the diagonal**, must `W` match that structure?
3. is the equivariant `W` **too constrained**, given we claim to be faithful to a CNN?

Short answers: **yes**; **no — it must match Θ's *symmetry group*, not its sparsity
pattern**; and **yes, measurably so — and there is a dial (period-`P` sharing) that moves
along exactly that axis without leaving the fast path.**

No new experiments here; this is the algebra needed before choosing which one to run.

---

## 0. The property the fast path rests on

Not the prefix sum (that is a memory trick over *samples* and survives any redesign). The
load-bearing fact is:

> `Θ` and `W` both commute with the cyclic shift `S` ⇒ the normal equations for `W`
> block-diagonalise over the characters of `Z_d`.

Concretely, with `W = [W_1 … W_c]`, each `W_a` a `d×d` circulant with DFT symbol
`ŵ_a[f]`, Parseval gives

```
L = (1/d) Σ_f  E | x̂0[f] − Σ_a ŵ_a[f] φ̂_a[f] |²
```

and `ŵ_a[f]` appears in **one** summand only. So the `cd` unknowns split into `d`
independent least-squares problems of size `c`:

```
P_f = E[ φ̂[f] φ̂[f]^H ]   (c×c),    q_f = E[ x̂0[f]* φ̂[f] ]   (c),
L^circ = Tr(Σ_p0) − Σ_f q_f^H P_f^{-1} q_f.
```

Worth stating explicitly because it is easy to assume otherwise: **this needs no
stationarity of the data.** Cross-frequency moments `E[φ̂_a[f] φ̂_b[g]^*]`, `f≠g`, are
nonzero for CIFAR — they simply never enter, because a circulant `W` cannot couple `f`
to `g`. The decoupling is a property of the *model*, not of `p(x0)`.

---

## 1. Free `W`: yes, it destroys everything

A free `W ∈ R^{d×K}`, `K = cd`, has optimum `W = Cov(x0,φ) Σ_φ^{-1}`, and `Σ_φ` is a dense
`K×K` matrix with no exploitable structure. Storage `8K²`:

| c | K = cd | Σ_φ |
|---|---|---|
| 8 | 24,576 | 4.8 GB |
| 16 | 49,152 | 19.3 GB |
| 32 | 98,304 | 77 GB |
| 1536 | 4.72M | 177 TB |
| 6144 | 18.9M | **2.6 PB** |

So free `W` is not "slower" at the widths we run — it is impossible. But note the top of
the table: **the free-`W` ablation is feasible on raw pixels for `c ≤ 8`, and awkward but
possible at `c = 16`.**

> ⚠ **Correction (2026-09-22).** This paragraph used to continue *"since the circulant is
> already on its floor by `c ≈ 8–32`, that is not a toy regime"*. That is a high-`σ`
> statement and I wrote it as a general one. The sweep below settles it: the floor claim is
> true at `σ ≥ 1.61` and false at `σ = 0.127`, which is the one place it was being used.

**Where the block-circulant RF actually floors in `c`** — `scripts/run_rf_pixel_csweep.sh`
→ `tables/rf_pixel_circ_csweep.npz`, job 47855859, 1 h 15 m, one code path
(`circulant_rf_mmse_lag2`), 8 seeds at `c ≤ 256`, 4 at `c = 512`, 2 at `c = 1536`. Entries
are the excess over the free Wiener denoiser, `mean ± sd` over seeds; `c = 3072, 6144` are
the existing single-seed cells from `tables/rf_pixel_featmatch2.npz`:

```
  sigma     c=32        96         256        512       1536      3072    6144
  0.127  +0.6213   +0.3891    +0.2907    +0.2338    +0.1772   +0.1504  +0.1294
          0.0417    0.0152     0.0146     0.0057     0.0017
  0.452  +3.5012   +3.2396    +3.0879    +3.0207    +2.9679   +2.9452  +2.9266
          0.0342    0.0207     0.0175     0.0057     0.0016
  1.610  +8.4030   +8.3627    +8.3404    +8.3292    +8.3199   +8.3170  +8.3151
          0.0194    0.0088     0.0041     0.0008     0.0001
  5.000  +9.0210   +8.9880    +8.9791    +8.9764    +8.9748   +8.9743  +8.9741
          0.0093    0.0019     0.0006     0.0002     0.0000
```

Read it as a fraction of the quantity being explained, not in absolute loss. At `σ = 1.61`
and `σ = 5.0` the `c = 32` value already sits within `1.1%` and `0.5%` of the `c = 6144`
value — floored, and the original claim stands. At `σ = 0.127` it does not: `c = 32`
overstates the final margin by `4.8×` (`+0.6213` against `+0.1294`), and even `c = 1536`
overstates it by `37%`. A power-law tail fit `L(c) = L_∞ + A c^{-α}` on `c ≥ 256` (residuals
at the seed-noise level) extrapolates to

```
  sigma    alpha    L_inf - Wiener    L(6144) - Wiener
  0.127    0.411        +0.071            +0.129
  0.452    0.590        +2.900            +2.927
  1.610    0.739        +8.313            +8.315
  5.000    1.032        +8.974            +8.974
```

so the margin at `σ = 0.127` is still roughly halving beyond the widest cell we have run,
while at `σ ≥ 1.61` `c = 6144` is the asymptote to four decimals. **The circulant still does
not cross the linear denoiser at any `σ`** — `L_∞ − L^lin = +0.071 > 0` — but the margin it
must close at low `σ` is about half of what the `c = 6144` number suggests.

Two consequences:

- The free-`W` ablation at `c ≤ 8` is a toy regime *at low `σ`* after all. At `σ ≥ 1.61` it
  is not. That probe (offered 2026-09-17, still unbuilt) was superseded anyway — see
  `docs/rf_blockdiag_derivation_and_toll.md` §5.
- For the band readout relaxation, which multiplies the per-frequency block by
  `|G| = (2B+1)²`, the useful quantity is how much of the nonlinear gain a given `c` buys.
  At `σ = 0.127`, against the equivariant-linear denoiser `9.173`: `c = 32` captures `57%`
  of the `c = 6144` gain of `1.144`, `c = 96` `77%`, `c = 256` `86%`, `c = 512` `91%`,
  `c = 1536` `96%`. So a band-vs-plain *differential* at `c = 256–512` captures most of the
  effect and costs ~7 and ~25 min per seed; an *absolute* loss number at low `σ` does not
  converge until `c ≳ 1536`, which at `|G| = 9` is ~7 h/seed.

---

## 2. Block-diagonal ≠ block-circulant

> ⚠ **§2(b) below is WRONG — see `docs/rf_blockdiag_derivation_and_toll.md` (2026-09-22).**
> A fresh 8×8 block at each diagonal position *does* have a fast path, and a cheaper one
> than we run today: if `W` carries the same block structure, the normal equations decouple
> across chunks by **direct sum** (disjoint output coordinates), with no group needed, then
> by an 8-point DFT within each chunk. `M·b = d` solves of size `c×c` — the same count as
> now. I looked for a symmetry and missed a separability. The model is still the wrong one
> to build, but for a completely different reason (its readout receptive field collapses to
> 8 pixels); §6's recommendation is superseded there too.

"8×8 circulant blocks along the diagonal" splits into two very different models, and the
distinction is the whole answer:

**(a) the same 8×8 block repeated.** `Θ` commutes with `S^8`, so the group is `Z_{d/8}`
rather than `Z_d`. Still abelian, still diagonalised by a DFT — of size `d/8`, with blocks
of size `8c`. **Fast path survives.** But the taps do *not* renew: it is the same 8 numbers
everywhere, which is the opposite of the stated goal.

**(b) a fresh random 8×8 block at each diagonal position.** This is the version that
renews taps — and it has **no** translation symmetry at all, because block `j` differs from
block `j'`. No group ⇒ no block-diagonalisation ⇒ the full `K×K` Gram ⇒ 2.6 PB at `c=6144`.

Being *block-diagonal* buys a cheap forward map `Θy`; it buys nothing in the readout solve,
which is where the cost lives. **Equivariance is what pays, sparsity is not.**

There is a second, non-computational reason to avoid tiling. Non-overlapping 8×8 blocks
give `d/8 = 384` distinct windows per channel instead of `d = 3072` overlapping ones — an
8× cut in exactly the "feature diversity" that the 2026-08-17 permutation test found to be
a *larger* effect than locality proper (permuting coordinates cost `+23.2` at `k/d=4` on
layer2, while it cost nothing on avgpool where only diversity is available).

---

## 3. Period-`P` sharing: the dial that does both

Let the filter used at position `p` depend on `p mod P`: `P` distinct filters per channel,
cycling. Windows still slide with stride 1 (overlap preserved), taps renew every `P`
positions, and `Θ` commutes with `S^P`, so the group is `Z_{d/P}`.

Decomposing positions as `P` copies of the regular representation of `Z_{d/P}`:

| | circulant `P=1` | period `P` | free / dense `P=d` |
|---|---|---|---|
| group | `Z_d` | `Z_{d/P}` | trivial |
| coarse frequencies | `d` | `d/P` | 1 |
| block size | `c` | `cP` | `cd` |
| `Θ` random params | `c·t` | `c·t·P` | `k·d` |
| `W` trained params | `c·d` | `c·d·P` | `c·d²` |
| solve cost | `c³d` | `c³P²d` | `(cd)³` |

(The `W` count is `dim Hom_{Z_{d/P}}(R[Z_{d/P}]^{cP}, R[Z_{d/P}]^{P}) = P·cP·(d/P) = cdP`,
which correctly returns `cd` at `P=1` and `cd²` at `P=d`.)

So period-`P` is a **genuine interpolation between block-circulant `W` and free `W`**, and
it strictly dominates tiling: it renews taps *and* keeps the overlapping windows *and*
keeps a group. It is the right way to write the 8×8 idea.

`P_Θ` and `P_W` can differ — the fast path only needs a common group, i.e.
`Z_{d/lcm(P_Θ, P_W)}`. Given §4, **spending `P` on `W` rather than on `Θ` is the better
bet.**

### Cost, chunked over frequencies (`nf` blocks live at a time)

Memory per coarse-frequency block is `16(cP)²` bytes; the Hermitian fold halves the count.

| c | P | `cP` | per block | comment |
|---|---|---|---|---|
| 1536 | 8 | 12,288 | 2.4 GB | ~19 GB at `nf=8`; ≈8× the `P=1` accumulation, so ~30 min/seed |
| 32 | 384 | 12,288 | 2.4 GB | `d/P = 8` freqs ⇒ 19 GB total; `t·P = d`, i.e. `Θ` random params matched to dense |
| 64 | 384 | 24,576 | 9.7 GB | 77 GB total — the ceiling |

`c=1536, P=8` is the informative cell: at `σ ≥ 1.61` the circulant is floored to four
decimals by `c = 1536` (§1), so anything that moves is attributable to the relaxation and
not to width. At `σ = 0.127` that is *not* true — `c = 1536` still overstates the final
margin by `37%` and the curve is falling `0.036` per doubling there — so a low-`σ` cell must
be run against a plain-circulant baseline at the *same* `c`, as a differential, never read
as an absolute loss.

---

## 4. Is equivariant `W` too constrained? Yes — and we have measured it

This is not speculative. `scripts/rf_circulant_readout_ablation.py` (GMM d=32, m_active=20,
exact population moments, N_REP=6, `w=4`) compares three models. **Re-run 2026-09-21 to
verify — numbers below are from that run, not from the July notes:**

```
min_k ( L[circulant Θ, free W] − L[dense Θ, free W] )        (negative = circulant wins)
  σ=0.5  −1.5159      σ=1.0  −0.9724      σ=2.0  −0.5206      σ=5.0  −0.1417
  and it wins over a RANGE of k, not one point: win_k = [32 … 1024] at σ=0.5

cost of constraining the readout  ( L[circ Θ, circ W] − L[circ Θ, free W] )
  σ=0.5  +0.426       σ=1.0  +1.598      σ=2.0  +1.913      σ=5.0  +0.670   (k → large)
```

⇒ on that problem the circulant *feature map* was a good inductive bias — it **beat dense
at every σ** — and the entire loss came from the *readout* constraint, which does **not**
vanish as `k→∞`.

(My July note recorded `−1.391 / −0.934 / −0.523 / −0.142`; σ=2 and σ=5 reproduce, σ=0.5
and σ=1 differ by ~0.1 across Θ-draws. The conclusion is unchanged, but quote the
re-verified figures.)

**We have never separated these two terms on raw pixels.** The `+2.5 … +9.6` margin in the
matched-free-parameter tables is a sum of
(i) an impoverished feature map and (ii) the readout toll, and the GMM result says (ii) may
be all of it. A redesign aimed at (i) — renewing taps — is aimed at the term we have *no*
evidence is binding, and there is standing evidence against it: the July `w`-sweep had the
gap growing monotonically with `t`, with full width `t=d` the **worst** case, even though
`t=d` gives `c·d` random taps, exactly matched to dense.

### On CNN faithfulness

Cuts both ways, and it is worth being explicit. A convolutional *denoiser* (U-Net, DDPM)
does end in a conv layer, so an equivariant `W` is defensible there — more so than for a
classifier, which ends in a dense head. But a U-Net is not one conv: downsampling, skip
connections and normalisation all break exact translation equivariance at finite scale.
Full block-circulant `W` is therefore a strictly *stronger* constraint than any CNN
actually imposes, and §4 says that gap is not free. Period-`P` is the honest middle:
"equivariant up to period `P`" is much closer to what a real architecture does.

---

## 5. A cheaper relaxation worth knowing about

Between "circulant `W`" and "free `W`" there is a two-stage, fully convex option:

```
W = W_circ + U V^T,    V ∈ R^{K×r} FIXED,   U ∈ R^{d×r} free
```

With `V` fixed the problem stays jointly linear in `(W_circ, U)`, and the Schur complement
only needs the frequency-decoupled inverse applied to `r` vectors — so the cost is `r`
circulant solves plus an `r×r` system. Natural choices for `V`: the top-`r` principal
directions of the residual `x0 − W_circ φ` from a first pass.

Note we already use the `r=1`, `V=1` case: the **free per-position bias `b ∈ R^d`**. That
single relaxation already moves the model's floor — `core/equivariant_floor.py`, same run
as above, σ=2.0: strictly-equivariant floor `MMSE(p̄0) = 18.6662` vs free-bias floor
`floor_free = 18.5051`, a drop of `0.161` bought by `d` extra parameters out of `cd²`.
A hint that the next few non-equivariant directions are not cheap either.

---

## 6. Recommendation

Ordered by information per hour:

1. **Free-`W` ablation on raw pixels at `c = 2,4,8`** (feasible today, no new math).
   Decomposes the measured margin into feature-map vs readout. If the readout is most of
   it, the tap redesign is aimed at the wrong term.
2. **`t`-sweep at fixed `c=1536`**, `t ∈ {8,32,128}`, σ = 1.61 and 5.0. Directly tests the
   tap-starvation premise. `B` is `(c,c,3(2t−1))` fp64 = 0.85 / 3.6 / 14.4 GB.
3. **Implement period-`P`** and run `c=1536, P ∈ {1,2,4,8}`. Highest value, but it is the
   only item needing new code (polyphase reshape `(c, d/P, P)`, rfft over the middle axis,
   blocks indexed by `(channel, phase)`; the Stein lag/noise assembly needs reindexing,
   which is the risky part). De-risked by a brute-force `K×K` reference at `c=2, P=4`
   (`K = 6144`) where the exact optimum is directly computable.

Items 1 and 2 answer "is the premise true?" for a few hours of GPU and no new derivation.
Item 3 is the redesign itself, and is worth doing only if 1 and 2 point at the feature map.
