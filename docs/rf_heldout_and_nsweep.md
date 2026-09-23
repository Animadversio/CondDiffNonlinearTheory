# Held-out evaluation of the circulant RF denoisers, and the finite-`N` control

**Status.** Complete. Jobs 47897466 (held-out), 47894575 (`N`-sweep) and 47890358 (`c`-sweep)
all finished 2026-09-23. Every table below is regenerated from the stored `.npz` files by

```
python scripts/rf_heldout_report.py
```

and is pasted here verbatim from that script's output rather than transcribed.

---

## 1. What this measures and why it was needed

Every `L^circ` this project had reported was an **in-sample** number: the per-frequency
readout `w_f = (P_f + λI)^{-1} q_f` was solved on the same moments it was scored against.
At `c = 1536` the 2-D model carries 4.7M trained parameters and 1.57M features against
`N = 10⁴` images, and on 2026-09-21 a model in that regime (the dense RF at `k/d = 8`) turned
out to be scoring well by memorising — its advantage over linear collapsed
`0.825 → 0.336 → 0.039` as `N` went `10k → 20k → 40k`, with `λ = 1e-6` doing nothing to stop
it. A steadily-falling-in-`c` curve at fixed `N` is exactly that signature.

Two independent checks were run against it.

* **The `N`-sweep** (indirect): hold `c` fixed, grow `N`, watch the excess over the same-`N`
  Wiener.
* **The held-out evaluation** (direct, michimin's construction): keep `w_f`, build
  `P_f^test` and `q_f^test` from a disjoint split, and evaluate

  ```
  L_test = Tr(Σ_test) + Σ_f wgt_f [ −2 Re⟨q_f^test, w_f⟩ + w_f^H P_f^test w_f ].
  ```

Section 6 checks the two against each other.

---

## 2. The estimator

`x0_test=` on both `core/rf_circulant2d.py::circulant2d_rf_mmse` (the `Z_32 × Z_32` arm) and
`core/rf_circulant_struct.py::circulant_rf_mmse_lag2` (the `Z_3072` arm). With `x0_test=None`
both return a float exactly as before; with a test split they return a dict.

### 2.1 Two in-sample losses, and they are not the same thing

Every stored table in this project holds

```
train = Tr(Σ) − Σ_f wgt_f · Tr( q^H (P + λI)^{-1} q ),
```

which is **not** the achieved residual of `w_f` — it is shifted by exactly `λ‖W‖²`.
michimin's `L_test` formula *is* the true residual. So both are returned:

| key | meaning | use it for |
|---|---|---|
| `train` | project convention, ridge-shifted | differencing against the stored `c`-sweep tables |
| `train_resid` | achieved residual of the same `w_f` | the honest gap, `test − train_resid` |
| `test` | held-out risk of that `w_f` | everything cross-model |

On CIFAR at `λ = 1e-6` the shift is ~0.002 at `c = 1536`. It is small but it is *not* noise,
and it grows with `c` (∝ `‖W‖²`), so mixing the two conventions puts a spurious `c`-dependent
drift into any gap. This is also what made an early brute-force reference disagree at 1e-7
before it was recognised as a feature rather than a bug.

### 2.2 Centring is part of the model

`β = μ_train − W · gmean_train`, so **test moments are centred by the TRAIN means** (both the
image mean `μ` and the feature mean). Re-centring on test would silently refit `d = 3072`
parameters at evaluation time. This is legitimate because

```
E[(φ − m)(φ − m)^T] = E[Cov(φ | x0)] + E[(G − m)(G − m)^T]
```

holds for *any* fixed `m`. Test-side noise terms (the lag Grams `S^n`, the exact-diagonal
Stein correction) **are** rebuilt on test samples; only `h`, `w_f` and the two train means
cross the split.

### 2.3 Validation

| check | result |
|---|---|
| `selftest2d`: 5 configs vs brute-force constrained LS, test split drawn from a *shifted and rescaled* distribution so the moments genuinely differ | ≤ 1.5e-15 on all three columns |
| `selftest_lag2`: `Z_d = Z_d × Z_1` with `Cin=1` makes the 2-D brute force an **independent** reference for the 1-D path | 3 configs, ≤ 5.4e-15 |
| the original 6 `lag2` regression cases | still ≤ 2e-15 ⇒ the refactor is loss-neutral |
| both suites feed one split in twice at `λ=0`, where `L_test` must collapse onto `L_train` | exact; pins sign and normalisation without either brute force |
| the driver **asserts** its `train` column reproduces the stored tables | 0.0 (2-D) / 5.7e-14 (1-D), at every cell of the run |

---

## 3. The linear baseline has to be held out too

This is the single largest effect in the exercise and it is easy to miss.

`Σ` at `d = 3072` carries 4.7M free parameters — *more* than the 2-D RF's 4.7M at `c = 1536`
— and is estimated from the same 10⁴ images. So the Wiener line moves under a split as well.
`linear_split()` in `scripts/rf_pixel_heldout.py` solves `A = Σ_tr(Σ_tr + σ²I)^{-1}` on train
and scores on test in closed form (the noise term `σ² Tr(AA^T)` is split-independent).

Measured at `c = 32`, `σ = 0.127`:

```
                train     test      gap
  Wiener       7.8997   8.3590   +0.4593
  2-D RF c=32  8.4391   8.4620   +0.0241
  1-D RF c=32  8.5382   8.5553   +0.0189
```

**Scoring a held-out RF against an in-sample Wiener invents a gap 19× the real one, in the
direction that flatters linear.** Every comparison in the driver is train-vs-train or
test-vs-test, never mixed.

---

## 4. Results

### 4.1 Held-out, both arms, all σ

```
    === sigma=0.127   Wiener train   7.8997  test   8.3590  (gap +0.4593) ===
      2d c=32    train   8.4391 ( +0.5394)   test   8.4620 ( +0.1031)   own gap +0.0241
      2d c=96    train   7.9528 ( +0.0532)   test   7.9847 ( -0.3743)   own gap +0.0331
      2d c=256   train   7.5518 ( -0.3478)   test   7.5913 ( -0.7677)   own gap +0.0411
      2d c=512   train   7.2271 ( -0.6726)   test   7.2765 ( -1.0825)   own gap +0.0511
      2d c=1536  train   6.8592 ( -1.0404)   test   6.9302 ( -1.4288)   own gap +0.0729
      2d c=3072  train   6.6525 ( -1.2472)   test   6.7469 ( -1.6121)   own gap +0.0965
      1d c=32    train   8.5382 ( +0.6385)   test   8.5553 ( +0.1963)   own gap +0.0189
      1d c=96    train   8.2899 ( +0.3902)   test   8.3101 ( -0.0489)   own gap +0.0215
      1d c=256   train   8.1758 ( +0.2761)   test   8.2010 ( -0.1580)   own gap +0.0263
      1d c=512   train   8.1303 ( +0.2306)   test   8.1603 ( -0.1986)   own gap +0.0311
      1d c=1536  train   8.0768 ( +0.1772)   test   8.1174 ( -0.2416)   own gap +0.0414
    === sigma=0.452   Wiener train  28.5522  test  29.1278  (gap +0.5756) ===
      2d c=32    train  31.5097 ( +2.9575)   test  31.5229 ( +2.3951)   own gap +0.0132
      2d c=96    train  30.8813 ( +2.3290)   test  30.9096 ( +1.7817)   own gap +0.0284
      2d c=256   train  30.2761 ( +1.7238)   test  30.3192 ( +1.1914)   own gap +0.0433
      2d c=512   train  29.5482 ( +0.9960)   test  29.6002 ( +0.4724)   own gap +0.0523
      2d c=1536  train  28.8223 ( +0.2700)   test  28.8831 ( -0.2447)   own gap +0.0612
      2d c=3072  train  28.3311 ( -0.2212)   test  28.3947 ( -0.7331)   own gap +0.0639
      1d c=32    train  32.0693 ( +3.5170)   test  32.0666 ( +2.9388)   own gap -0.0026
      1d c=96    train  31.7838 ( +3.2315)   test  31.7824 ( +2.6546)   own gap -0.0012
      1d c=256   train  31.6550 ( +3.1028)   test  31.6553 ( +2.5275)   own gap +0.0003
      1d c=512   train  31.5754 ( +3.0232)   test  31.5764 ( +2.4486)   own gap +0.0011
      1d c=1536  train  31.5202 ( +2.9679)   test  31.5223 ( +2.3945)   own gap +0.0022
    === sigma=1.61   Wiener train  72.1956  test  72.4610  (gap +0.2654) ===
      2d c=32    train  79.1922 ( +6.9966)   test  79.0162 ( +6.5552)   own gap -0.1760
      2d c=96    train  78.5170 ( +6.3214)   test  78.3569 ( +5.8959)   own gap -0.1601
      2d c=256   train  78.3341 ( +6.1385)   test  78.1809 ( +5.7199)   own gap -0.1532
      2d c=512   train  78.2981 ( +6.1025)   test  78.1470 ( +5.6860)   own gap -0.1511
      2d c=1536  train  78.2023 ( +6.0067)   test  78.0527 ( +5.5917)   own gap -0.1496
      2d c=3072  train  78.1359 ( +5.9403)   test  77.9877 ( +5.5267)   own gap -0.1483
      1d c=32    train  80.5994 ( +8.4039)   test  80.4574 ( +7.9964)   own gap -0.1420
      1d c=96    train  80.5604 ( +8.3649)   test  80.4191 ( +7.9580)   own gap -0.1414
      1d c=256   train  80.5415 ( +8.3459)   test  80.4002 ( +7.9392)   own gap -0.1413
      1d c=512   train  80.5253 ( +8.3297)   test  80.3841 ( +7.9231)   own gap -0.1412
      1d c=1536  train  80.5155 ( +8.3199)   test  80.3744 ( +7.9134)   own gap -0.1411
    === sigma=5.0   Wiener train 130.3022  test 129.9680  (gap -0.3342) ===
      2d c=32    train 138.8257 ( +8.5235)   test 138.1118 ( +8.1438)   own gap -0.7139
      2d c=96    train 138.1212 ( +7.8190)   test 137.4182 ( +7.4502)   own gap -0.7030
      2d c=256   train 137.9662 ( +7.6640)   test 137.2657 ( +7.2977)   own gap -0.7005
      2d c=512   train 137.9495 ( +7.6473)   test 137.2493 ( +7.2813)   own gap -0.7002
      2d c=1536  train 137.9402 ( +7.6379)   test 137.2400 ( +7.2721)   own gap -0.7001
      2d c=3072  train 137.9374 ( +7.6352)   test 137.2374 ( +7.2694)   own gap -0.7001
      1d c=32    train 139.3222 ( +9.0200)   test 138.6409 ( +8.6729)   own gap -0.6813
      1d c=96    train 139.2909 ( +8.9887)   test 138.6100 ( +8.6420)   own gap -0.6809
      1d c=256   train 139.2818 ( +8.9796)   test 138.6011 ( +8.6331)   own gap -0.6807
      1d c=512   train 139.2787 ( +8.9765)   test 138.5980 ( +8.6300)   own gap -0.6807
      1d c=1536  train 139.2770 ( +8.9748)   test 138.5963 ( +8.6283)   own gap -0.6807
```

**The win is larger held out than in-sample, both arms, every `c`, at low σ.** The RF *does*
overfit — its own gap grows monotonically with `c`, and the 2-D arm's is ~1.6× the 1-D arm's
at equal `c`, so the extra freedom does buy some memorisation — but Wiener's gap is 6–20×
bigger.

⚠ The σ ≥ 1.61 "own gaps" are negative and nearly `c`-independent. Those are **not**
generalisation gaps; see §7.

### 4.2 `N`-sweep

```
       model              N=10k      20k      40k     move(4xN)
       s=0.127  2D c=512     7.2271   7.2592   7.2668   +0.0397
       s=0.127  2D c=1536    6.8592   6.8961   6.9066   +0.0473
       s=0.127  1D c=1536    8.0768   8.1041   8.1115   +0.0346
       s=0.127  WIENER       7.8997   8.0241   8.0795   +0.1798
       s=0.452  2D c=512    29.5482  29.6027  29.6135   +0.0653
       s=0.452  2D c=1536   28.8223  28.8824  28.8964   +0.0741
       s=0.452  1D c=1536   31.5202  31.5684  31.5705   +0.0503
       s=0.452  WIENER      28.5522  28.7319  28.8059   +0.2537
```

Both arms pass. In *excess* terms the margin over the same-`N` Wiener grows with `N` — the
opposite of the dense memorisation signature — but the mechanism is visible only in the raw
losses: the RFs move 0.03–0.07 over 4× `N` while the in-sample Wiener moves 0.18–0.25.

An in-sample loss rises with `N` by exactly that model's optimism, so **this is an independent
measurement of the quantity §4.1 measures directly**, and the two agree in scale (Wiener
+0.18 vs gap +0.459; 2-D `c=1536` +0.047 vs +0.073; 1-D +0.035 vs +0.041).

⇒ The honest headline is *not* that the RF generalises better as `N` grows. It is that **at
`N = 10⁴` the linear baseline was the most overfit object in the comparison**, and every
"excess over Wiener" in this project's older tables is correspondingly ~0.4 too flattering to
linear at σ = 0.127 (~0.5 at σ = 0.452).

### 4.3 Aside: where the population linear floor is

⚠ **This subsection is the only place in the document where anything is fitted on more than
10,000 images, and it is deliberately not the headline.** The primary comparison (§5) trains
every model — RF and Wiener alike — on the same 10,000 images. This aside exists only to
answer the separate question *"how much of the remaining linear headroom is left on the
table by stopping at `N = 10⁴`?"*, which needs larger `N` by construction.

```
    --- sigma=0.127 ---
      N        in-sample  held-out   midpoint |  normalised by Tr(Sigma), x1e3
      10000      7.8997    8.3590    8.1293 |    41.247    44.042    42.645
      20000      8.0241    8.2423    8.1332 |    41.962    43.428    42.695
      40000      8.0795    8.1854    8.1324 |    42.380    43.125    42.752
      50000      8.0857    8.1739    8.1298 |    42.467    43.065    42.766
    --- sigma=0.452 ---
      N        in-sample  held-out   midpoint |  normalised by Tr(Sigma), x1e3
      10000     28.5522   29.1278   28.8400 |   149.081   153.471   151.276
      20000     28.7319   28.9811   28.8565 |   150.255   152.699   151.477
      40000     28.8059   28.9097   28.8578 |   151.097   152.312   151.705
      50000     28.7933   28.8946   28.8439 |   151.225   152.235   151.730
```

The two columns converge from **opposite** sides (optimism halves as `N` doubles = `d/N`), and
the raw midpoint is stable to 0.004 over a 5× range in `N`:

> **best achievable linear denoiser ≈ 8.130 at σ = 0.127, ≈ 28.844 at σ = 0.452.**

`7.900` is optimistic (in-sample at `N = 10⁴`); `8.359` is pessimistic (a 10⁴-fitted model's
test risk). Trace-normalising (§7) shifts the floor to ≈8.19 / ≈29.06 — a 0.06 difference that
changes none of the conclusions below.

---

## 5. What the results support, scoped

**Everything in this section is matched: every model — both RF arms and the Wiener baseline —
is fitted on the same 10,000 CIFAR training images and scored on the same 10,000 CIFAR test
images.** No floor estimate, no `1/N` extrapolation, no trace normalisation, nothing trained
on 50k. The numbers are read straight off §4.1.

```
  sigma = 0.127     held-out loss on the same 10,000 test images
    Wiener  (10k)       8.3590
    1-D RF  (10k)       8.1174   c = 1536    -0.2416
    2-D RF  (10k)       6.7469   c = 3072    -1.6121

  sigma = 0.452
    Wiener  (10k)      29.1278
    1-D RF  (10k)      31.5223   c = 1536    +2.3945
    2-D RF  (10k)      28.3947   c = 3072    -0.7331
```

* **2-D `Z_32 × Z_32` beats the linear denoiser outright at low σ**, by 1.61 at σ = 0.127 and
  0.73 at σ = 0.452. It crosses at `c ≈ 96` (σ = 0.127) and `c ≈ 1536` (σ = 0.452). This is
  the project's first genuine win over linear on raw pixels.
* **1-D `Z_3072` also beats it at σ = 0.127**, by 0.24 at `c = 1536`, and crosses at `c ≈ 96`.
  It does **not** at σ = 0.452 (+2.39, and its curve is flat in `c`).
* ⚠ **At σ ≥ 1.61 nothing crosses, and the old claim survives.** Holding out moves the RFs
  only ~0.4 closer to linear and they remain 5.6–8.6 *above* it, with both curves floored in
  `c` (2-D moves 0.01 from `c = 512` to `1536` at σ = 5). So the baseline correction changes a
  verdict **only at σ ≤ 0.452**.
* The crossing in σ therefore lies between 0.452 and 1.61 and has not been measured.
  Job 47923736 (`scripts/run_rf_heldout_sig2.sh`) adds σ = 0.621, 0.853, 1.172, 2.212 into the
  same table to locate it.

### The handicapped form

Separately — and this is a *strengthening*, not the primary claim — the 2-D win survives
giving the baseline five times the data. Against the ≈8.130 population floor of §4.3, or
equivalently against a Wiener fitted on all 50,000 training images and scored on the same test
split (`8.1739`), the 2-D RF trained on 10k still wins by ~1.4 at σ = 0.127. The 1-D arm does
**not** survive that handicap: `8.1174` sits inside the `[8.086, 8.174]` bracket, so against a
50k-trained baseline it is a tie rather than a win. Both readings are in §4.1/§4.3; quote the
matched-10k one unless the handicap is the point being made.

### What this retracts

> *"Circulant never crosses linear at any σ"* — recorded repeatedly in this project's tables
> and in `docs/rf_circulant_relaxations.md`.

It was an **in-sample** statement against a Wiener baseline that is itself overfit by +0.46 at
`N = 10⁴`. At matched training data it is now false at σ ≤ 0.452 — the 2-D arm crosses at both
σ, the 1-D arm crosses at σ = 0.127 — and it remains true at σ ≥ 1.61. Any deficit quoted
anywhere against `linear = 7.900 / 28.552` is train-vs-train and needs the same correction.

**Lesson (fifth of this family).** Before calling a loss difference a class property, hold out
*both* sides. An in-sample baseline with 4.7M free parameters is not a baseline.

---

## 6. Do the two routes agree?

The population `L*` must be **bracketed**: in-sample climbs toward it from below, held-out
falls toward it from above. Richardson-extrapolating the in-sample series in `1/N`:

```
      model              RAW extrap  heldout   bracket |  NORMALISED (x1e3) bracket
      s=0.127  2D c=512      7.2744    7.2765  +0.0021 |    38.272    38.339   +0.067
      s=0.127  2D c=1536     6.9170    6.9302  +0.0131 |    36.391    36.514   +0.123
      s=0.127  1D c=1536     8.1189    8.1174  -0.0015 |    42.715    42.770   +0.055
      s=0.127  WIENER        8.1349    8.3590  +0.2241 |    42.797    44.042   +1.245
      s=0.452  2D c=512     29.6244   29.6002  -0.0242 |   155.859   155.960   +0.101
      s=0.452  2D c=1536    28.9104   28.8831  -0.0272 |   152.102   152.182   +0.079
      s=0.452  1D c=1536    31.5726   31.5223  -0.0503 |   166.109   166.088   -0.022
      s=0.452  WIENER       28.8799   29.1278  +0.2479 |   151.940   153.471   +1.531
```

* At σ = 0.127 the RF cells agree to **0.002–0.013** from two independent routes, and both
  routes rank the models identically (Wiener ≫ 2-D > 1-D in optimism). This is the validation
  that the held-out estimator measures what it claims.
* Wiener's bracket stays 0.22–0.25 wide. That is not a disagreement: `d/N = 0.31`, its
  optimism is still large at `N = 40k` and the `1/N` extrapolation has not converged — the
  same reason the floor needed the `N = 50k` point.
* In **raw** units the σ = 0.452 brackets go negative (worst `−0.050`, ~30× that cell's
  0.0016 seed sd) — apparently impossible, since it would say the in-sample loss at `N = 40k`
  exceeds the `N = 10k` model's test loss. **Normalised, every violation disappears.** It was
  scale mixing, not a defect in either measurement. See §7.

---

## 7. Two scale traps, and the rule that avoids both

### 7.1 The two CIFAR splits are not exchangeable

`Tr(Σ_test) = 189.794` vs `Tr(Σ_train) = 191.522` — the test images carry **1.728 less energy
(0.9%)**. Found because the σ = 1.61 "own gaps" came out negative, which no honest
generalisation gap can be.

Diagnostic: run the split **both ways**; the antisymmetric part is the trace offset, the mean
is the real optimism. For `linear_split`:

```
   sigma   fit-tr->test   fit-te->train    mean gap    asymmetric?
   0.127     +0.4593         +0.4441        +0.4517    no (3%)
   0.452     +0.5756         +0.5855        +0.5805    no (2%)
   1.610     +0.2654         +0.4399        +0.3526    YES
   5.000     -0.3342         +0.6409        +0.1533    YES, sign flips
```

* At low σ the headline is clean — symmetric to 2–3%, so the +0.459 optimism, the ≈8.13 floor
  and the 2-D win all stand exactly as recorded.
* At σ ≥ 1.61 a one-direction held-out number is contaminated: `L ≈ Tr(Σ) − (small)`, so the
  0.9% deficit passes almost 1:1 into the test loss and can outweigh the optimism entirely
  (σ = 5: apparent gap −0.334, true +0.153). **Use the mean of both directions above σ ≈ 1.**
* The same test on the RF (1-D, `c = 256`, σ = 0.452) gives `+0.0003` one way and `+0.0220`
  the other ⇒ **symmetrised optimism +0.0112**. This *softens* an earlier claim that the 1-D
  circulant has literally zero generalisation gap at σ = 0.452: it is small — still ~50× below
  Wiener's +0.58 — but not zero.
* Test-vs-test **excess** comparisons are unaffected: both models are scored on the same
  split, so the offset cancels in the difference. Only per-model absolute gaps need
  symmetrising.

### 7.2 The training subsets are not on a common scale either

```
    train[:10000 ]   191.5220   vs train[:10000]: +0.0000
    train[:20000 ]   191.2217   vs train[:10000]: -0.3003
    train[:40000 ]   190.6448   vs train[:10000]: -0.8773
    train[:50000 ]   190.4011   vs train[:10000]: -1.1209
    TEST (10000)    189.7936   vs train[:10000]: -1.7284
```

Bigger CIFAR subsets are *less* energetic. So an in-sample loss at `N = 40k` is depressed on
that account alone, and a raw `N`-move **understates** the optimism. This is what produced the
impossible brackets in §6.

> **Rule.** For any comparison across different image sets — different `N`, or train vs test —
> either normalise by `Tr(Σ)` or keep the evaluation set fixed.

This is the third instance of this family (the 2025 σ = 5 "floor violation" was the first:
at large σ a trace difference between two sample sets propagates ~1:1 into the loss).

---

## 8. Reproduction

| what | how |
|---|---|
| all tables in this doc | `python scripts/rf_heldout_report.py` (~60 s, CPU; sections A–G) |
| the RF direction test of §7.1 | `SWAP=1 SECTIONS=H python scripts/rf_heldout_report.py` (GPU, ~2 min) |
| the held-out run, σ = 0.127/0.452/1.61/5.0 | `sbatch scripts/run_rf_heldout.sh` → `tables/rf_pixel_heldout.npz` (job 47897466, 3 h 29 m) |
| the held-out run, σ = 0.621/0.853/1.172/2.212 | `sbatch scripts/run_rf_heldout_sig2.sh` → same table (job 47923736, in flight) |
| the `N`-sweep | `sbatch scripts/run_rf_circ2d_nsweep.sh` → `tables/rf_circ{2d,1d}_nsweep_*.npz` (job 47894575, 1 h 36 m) |
| the `c`-sweep | `sbatch scripts/run_rf_circ2d.sh` → `tables/rf_pixel_circ2d.npz` (job 47890358) |
| estimator self-tests | `python -m core.rf_circulant2d` and `python -m core.rf_circulant_struct` |

Held-out costs ~2× in-sample (both passes run on both splits, and `B_train`/`B_test` are live
simultaneously), which is why `sizing2d`/`sizing1d` in the driver halve the frequency budget
relative to the in-sample drivers.

---

## 9. Open

* Dense at `k/d ≤ 8` (`tables/rf_pixel_dense_sweep.npz`) is still **in-sample only**. Those
  numbers need the same held-out correction before they can be compared against anything here.
* The band-modulated nonlinear RF differential at `c = 256` and `512` — still the decisive test
  of whether the nonlinear gain transfers to a larger linear class (see
  `docs/rf_band_relaxation.md` §5 and `docs/rf_blockdiag_derivation_and_toll.md`). Note the
  `c`-sweep's evidence on that question is now *positive*: the nonlinear gain **grew** under
  the `Z_32 × Z_32` class rather than shrinking.
* The 2-D low-σ curve is still falling at `c = 3072`; none of the σ ≤ 0.452 numbers are a
  floor in `c`.
