# The lag trick, and what it does and does not buy

michimin's question: with banded circulant blocks most entries are zero — can the loss be
computed more cheaply, and is the circulant case then a *constant factor* of dense at
feature-match? Answer: the trick is real and large, the constant-factor claim holds for
**memory and build** and **fails for the solve**.

## 1. The obstruction it removes

michimin's closed form is
```
L^circ = Tr(Sigma_p0) - sum_f q_f^H P_f^{-1} q_f
```
The outer sum over frequencies accumulates trivially — build `P_f`, solve, add the scalar,
discard. That was never the problem. The problem is that *building any single* `P_f` needed
the noise Grams at **every** frequency, through

```
f_f^H (rho^{o n} o M) f_f = (1/d) sum_g w_g D_ab(f+g),     D_ab(f') = f_{f'}^H M^(a,b) f_{f'}
```

a cyclic correlation over the frequency index. That forces the full `(d, c, c)` tensor:
1.69 TB at `c = d = 3072`, which is what made feature-match on raw pixels look impossible.

## 2. The trick

Do the same contraction in **lag space** instead. With `m = p - q`,

```
f_f^H (rho^{o n} o M) f_f = (1/d) sum_m e^{-2 pi i f m / d} psi(m)^n R_ab(m)
```

where `R_ab(m) = (1/N) sum_n sum_q C_a[n, q+m] C_b[n, q]` is the lag-`m` cross-correlation of
the two blocks' feature matrices, and `psi_ab(m) = sum_j h_a[j] h_b[(j+m) mod d]` is the
filter autocorrelation.

**`psi` inherits the band support of `h`.** If `h_a`, `h_b` are supported on `t` contiguous
taps then `psi_ab(m) = 0` for all `|m| >= t`, so only `L = 2t - 1` lags contribute — 15 lags
at `t = 8`, against `d = 3072` frequencies. Verified numerically: for `d=3072, t=8` the
nonzero lags are exactly `{-7, ..., +7}`.

The Hadamard powers `rho^{o2}`, `rho^{o3}` are `psi^2`, `psi^3` elementwise and have the
**same** support, so all three noise orders need the same 15 lags.

Storage drops from `4 d c^2` to `4 L c^2`, a factor `d / L = 205x`:

| k/d | c | all frequencies | 2t-1 lags |
|---|---|---|---|
| 1 | 3072 | 1.69 TB | 8.4 GB |
| 2 | 6144 | 6.75 TB | 33.8 GB |
| 4 | 12288 | 27.00 TB | 135.0 GB |

## 3. Fact-check of the constant-factor claim

At feature-match the dense model has `k = j*d` rows and the circulant has `c = j*d` blocks:

**Corrected 2026-08-18.** An earlier version of this table claimed the build was also a
constant factor `L`. That was wrong: it counted only the *noise* term and silently omitted
the *data* term, which is the larger of the two by ~200x. Both are recorded here so the
error is not repeated.

`Sigma_phi` has two parts and only one of them is banded:

* **noise** `rho^{o n} o (C_n^T C_n)` — `psi` is band-supported, so `L = 2t-1` lags. Cost
  `N c^2 L`.
* **data** `Sigma_data^{(a,b)}[p,q] = (1/N) sum_n Gc_a[n,p] Gc_b[n,q]` — **not banded at
  all.** `phi = relu(Theta y)` is dense whatever `Theta` is, so in lag space `R^G_ab(m)` is
  nonzero at *all* `d` lags. Streaming one frequency at a time costs `N c d^2`; holding all
  frequencies costs `N c^2 d`. Either way it is `O(d)` larger than the noise term and it
  dominates.

| | dense | circulant (lag trick) | ratio |
|---|---|---|---|
| build, noise part | — | `N c^2 L` | — |
| build, data part | — | `N c d^2` (streamed) | — |
| **build, total** | `N k^2 = N j^2 d^2` | `N c d^2 + N c^2 L` | **`~d/j`, NOT constant** |
| **memory** | `k^2` | `(L+1) c^2` | **`~L = 15`, constant** |
| **solve** | `k^3 = j^3 d^3` | `d c^3 = j^3 d^4` | **`d = 3072`, NOT constant** |

Measured at `d = 3072`, `N = 10^4`, `t = 8`:

| k/d | dense build | circ noise | circ data | circ total | ratio |
|---|---|---|---|---|---|
| 1 | 9.44e10 | 1.42e12 | 2.90e14 | 2.91e14 | 3087x |
| 2 | 3.77e11 | 5.66e12 | 5.80e14 | 5.85e14 | 1551x |

**Verdict: the constant-factor claim holds for MEMORY only.** Build is `~d` times dense and
solve is `d` times dense. The lag trick removes no flops anywhere — it is purely a memory
result, and calling it a compute result would overstate it by two to three orders of
magnitude.

The stronger statement that *is* true, and is probably the one for the writeup:

> At equal ROW COUNT the circulant is `d^2` cheaper to build and `d^2` cheaper to store than
> dense. Feature-match gives the circulant `d` times more rows than the dense model it is
> matched against, so it pays `d` in build and `d` in solve for a `d^2` structural saving —
> a net `d`-fold win over what those rows would otherwise cost.

## 4. What is still not available

Sparsity does **not** help the feature Grams: `phi = relu(Theta y)` is dense even when
`Theta` is banded, so `R_ab(m)` itself is a dense correlation. The lag trick reduces the
number of *lags* that must be retained, not the cost of each one.

It also does not apply to the full-width circulant (`t = d`), where `psi` has full support
and `L = d`. The trick is specific to the banded construction.
