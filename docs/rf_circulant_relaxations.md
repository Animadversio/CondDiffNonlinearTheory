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
possible at `c = 16`.** Since the circulant is already on its floor by `c ≈ 8–32`, that is
not a toy regime — it is most of the way to the behaviour we care about. This is the probe
offered on 2026-09-17 and still unbuilt.

---

## 2. Block-diagonal ≠ block-circulant

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

`c=1536, P=8` is the informative cell: the circulant is long since floored at that `c`, so
anything that moves is attributable to the relaxation and not to width.

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
