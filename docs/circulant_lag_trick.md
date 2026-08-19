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

| | dense | circulant (lag trick) | ratio |
|---|---|---|---|
| build | `N k^2 = N j^2 d^2` | `N c^2 L = N j^2 d^2 L` | **`L = 15`, constant** |
| memory | `k^2` | `c^2 L` | **`L = 15`, constant** |
| solve | `k^3 = j^3 d^3` | `d c^3 = j^3 d^4` | **`d = 3072`, NOT constant** |

So the claim is **correct for build and memory** and **wrong for the solve**. The solve
inverts `d` matrices of size `c x c = (jd) x (jd)`, where dense inverts one of size `jd`;
that is an extra factor of `d` no reordering removes, because it is inherent to having `d`
independent per-frequency problems each as large as the whole dense problem.

The claim as stated ("constant O as expensive as dense") should therefore be narrowed to
memory and build, or restated as "constant factor per frequency".

Absolute cost is nonetheless fine, because the constant is small and `d c^3` at these sizes
is not large in wall-clock terms:

| k/d | dense build | dense solve | circ build | circ solve |
|---|---|---|---|---|
| 1 | 7.5e11 | 2.3e11 | 1.1e13 | 7.1e14 (~24 s) |
| 2 | 3.0e12 | 1.9e12 | 4.5e13 | 5.7e15 (~3 min) |

## 4. What is still not available

Sparsity does **not** help the feature Grams: `phi = relu(Theta y)` is dense even when
`Theta` is banded, so `R_ab(m)` itself is a dense correlation. The lag trick reduces the
number of *lags* that must be retained, not the cost of each one.

It also does not apply to the full-width circulant (`t = d`), where `psi` has full support
and `L = d`. The trick is specific to the banded construction.
