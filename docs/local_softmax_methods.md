# Local softmax denoisers (Kamb & Ganguli, Lukoianov et al.) on the held-out footing

**Code.** `core/local_softmax.py` (estimators), `scripts/local_softmax_heldout.py` (driver +
`VALIDATE=1` asserts), `scripts/local_softmax_heldout_plot.py` (figure
`figures/local_softmax_vs_sigma.{png,pdf,csv}`).
**Tables.** `tables/local_softmax_heldout.npz` (wiener / LS / Lukoianov / EDM, first 1000 test
images, σ = 0.005 … 40), `tables/bayes_oracle_heldout_wide.npz` (global oracle at the added σ),
`tables/local_softmax_heldout_els.npz` (partial ELS, see §4).

## 1. What these are, relative to what the project already had

`scripts/bayes_oracle_heldout.py` is the empirical-Bayes posterior mean with the pool
`train[:10000]` as atoms:

```
x̂(y) = Σ_n softmax_n(−‖y − x_n‖² / 2σ²) x_n
```

Held out it scores about 100 per image for every σ ≤ 1.6, which is 12× the Wiener at σ = 0.127:
a test image is never one of the atoms. With the pool containing the evaluated image
(train = eval, the in-sample oracle) it goes to 0, because it recalls that image.

The three local estimators below compute the same softmax but restrict the distance to a
window `M_m` around each output coordinate `m`:

```
x̂(y)[m] = Σ_cand softmax(−‖M_m (y − cand)‖² / 2σ²) · cand[m]
```

| arm | paper | candidates for output pixel u | window `M_m` | boundary |
|---|---|---|---|---|
| `ls|σ|k` | Kamb & Ganguli, LS | train image n at the same u (N per pixel) | k×k square, all channels | zero pad (window truncated) |
| `els|σ|k` | Kamb & Ganguli, ELS | every k×k patch of every train image at every location (N·H·W per pixel), value = patch centre | k×k square, all channels | circular |
| `luk|σ|τ|global` | Lukoianov et al. 2025 | train image n at the same coordinate | row m of the Wiener filter `W = Σ(Σ+σ²I)⁻¹`, divided by `W_mm`, binarised at `τ · max` (spans channels) | n/a |

Two limits hold by construction and are asserted in `_validate`:
- LS with k ≥ 2H−1 is the global oracle. So `ls|σ|63` *is* the global held-out curve on the
  same images and noise, and it matches the full-test table to within MC error.
- ELS with k = H on the torus is the global oracle over all circular shifts of the pool.

## 2. Port from analytic-diffusion-studio

Source: `src/local_diffusion/models/local_score_machines.py` and `models/pca_locality.py`,
CC BY-NC 4.0.

**VP to EDM.** The reference uses VP logits `−‖x_t − √ᾱ x_n‖² / 2(1−ᾱ)`. With
`y = x_t/√ᾱ` and `σ² = (1−ᾱ)/ᾱ` these become `−‖y − x_n‖² / 2σ²`. Everything here runs in
EDM units (`x_edm = 2x − 1`, `σ_edm = 2σ`), which leave both the softmax and the Wiener
filter invariant. Errors are divided by 4 to report [0,1]-pixel units.

**Deliberate departures from the reference:**
1. **Whole-pool softmax.** Every softmax runs over the whole pool with an online
   log-sum-exp. The reference `pca_locality` normalises within each 512-image dataloader batch
   and averages the batches (`WeightedStreamingSoftmax.add`). That is not the paper's Eq. 8,
   and it blurs at low σ.
2. **Continuous σ.** σ is continuous rather than snapped to one of 1000 DDPM timesteps.
3. **Covariance from the pool.** The Lukoianov covariance comes from the 10k pool, not from the
   50k PCA the reference downloads.
4. **Hyper-parameter selection.** k and τ are picked per σ on validation
   `train[10000:10000+NVAL]`, never on test. Kamb & Ganguli instead calibrate k greedily
   against a target network; the stored k-sweeps allow that later without re-running.

**ELS cost.** For query pixel u and candidate (n, s),
`d²(u; n, s) = box_k[ ‖y(w) − x_n(w+s)‖² ](u)`. So one shift-gather of the pool and one
circular integral image serve every k at once. The layout is (u, candidates), so the per-u
softmax reduces over the contiguous dimension and the numerator is a batched matmul.

Cost is about 6 s per image for 9 k values on an H100 (float32, which reproduces float64 to
3 decimals). That is why ELS runs on 200 test images rather than 1000.

## 3. Conditional mode (not yet run)

`COND=1` restricts every softmax pool to the train images of the query's own class, which is
Kamb & Ganguli's `label=`. It adds the per-class Wiener `W_C` and uses the class-conditional
EDM (`cond-vp`). The Lukoianov mask stays unconditional. This mode is implemented but has not
been run yet.

## 4. Results (held out, unconditional, σ = 0.0002 … 40)

`figures/local_softmax_vs_sigma.png` (table twin `.csv`), `python scripts/local_softmax_heldout_plot.py`.
Runs: `logs/local_softmax_heldout.log`, `logs/local_softmax_wide{,2,3}.log`,
`scripts/run_local_softmax_wide.sh`. First 1000 test images, paired noise. The global
train=eval curve comes from `tables/bayes_oracle_heldout{,_wide}.npz` (10k train images,
in-sample).

The table gives MSE per image in [0,1] pixel units, with the ratio to Wiener on the same
images in brackets.

| σ | Wiener | Lukoianov (τ) | LS (k) | EDM | global held out | global train=eval |
|---|---|---|---|---|---|---|
| 0.0002 | 1.24e-4 (= dσ² 1.23e-4) | 0 (8-bit snap) | 0.33 (1) | 1.25e-4 1.01× † | 99.8 | 0 |
| 0.001 | 3.23e-3 | 1.8e-3 0.55× (8-bit snap) | 0.33 (1) | 2.59e-3 0.80× | 99.8 | 0 |
| 0.002 | 0.0127 | 0.0123 0.96× | 0.33 (1) | 0.0085 0.67× | 99.8 | 0 |
| 0.005 | 0.069 | 0.08 (0.3) 1.11× | 0.35 (1) 5.1× | 0.040 0.58× | 99.8 | 0 |
| 0.05 | 2.61 | 3.42 (0.2) 1.31× | 5.39 (1) 2.07× | 1.39 0.53× | 99.8 | 0 |
| 0.127 | 8.41 | 9.33 (0.1) 1.11× | 9.82 (3) 1.17× | 4.63 0.55× | 99.9 | 0 |
| 0.853 | 47.50 | 48.49 (0.1) 1.02× | 51.05 (9) 1.07× | 37.72 0.79× | 100.0 | 0.01 |
| 2.212 | 86.64 | 87.84 (0.1) 1.01× | 91.99 (19) 1.06× | 81.01 0.94× | 105.3 | 22.7 |
| 5 | 130.18 | 129.95 (0.05) 1.00× | 135.07 (31) 1.04× | 128.98 0.99× | 131.2 | 124.5 |
| 10 | 162.19 | 161.86 1.00× | 166.90 (31) 1.03× | 162.00 1.00× | 161.9 | 164.3 |
| 40 | 185.79 | 187.17 1.01× | 187.32 (15) 1.01× | 186.61 1.00× | 185.8 | 189.5 |

- **Neither local softmax beats the held-out Wiener at any σ.** Lukoianov is the closer of
  the two, within 2% for 0.45 ≤ σ ≤ 40. Both are worst at low σ, where the posterior
  collapses onto one training patch (N_eff ≈ 1) and the error floor becomes the
  nearest-neighbour distance rather than dσ². The best settings there are the smallest
  windows (LS k = 1, Lukoianov τ ≥ 0.3, i.e. about the diagonal).
- **EDM rejoins linear at the LOW end too.** The EDM/Wiener ratio goes 0.53× (σ = 0.05) →
  0.67× (0.002) → 0.80× (0.001) → 0.93× (0.0005) → 1.01× (0.0002). Wiener itself → dσ²
  (identity) there. † σ < 0.001 is below EDM's σ_min (σ_edm = 0.002), so it is extrapolation.
- **8-bit artefact.** Lukoianov → 0 for σ ≤ 0.001: its mask shrinks to the diagonal as σ → 0,
  i.e. a per-coordinate nearest value among 10k training pixels, which snaps to the exact
  1/255 level once σ ≪ 1/255. LS k = 1 matches the 3-channel colour jointly instead, so it
  floors at the mean nearest-colour distance (0.33).
- **EDM ≈ 0.55× Wiener from σ = 0.005 to 0.127 and converges to linear from σ ≈ 5 up.**
  All held-out curves meet at σ ≥ 10, where the optimal denoiser is the mean plus a small
  linear correction. At σ = 40, EDM (0.822 ± 0.166) and Lukoianov sit slightly above
  Wiener.
- **Global train=eval is exactly 0 from σ = 0.0002 up to σ ≈ 0.6** (it never approaches linear
  from below, in ratio), because it recalls the evaluated image. It is
  still 0.26× Wiener at σ = 2.2 and joins the others by σ ≈ 5–10. Held out, the same
  estimator is ~100 flat up to σ ≈ 1 (up to 1457× Wiener at σ = 0.005): the test image is
  never an atom.
- **Selection noise at σ = 40.** Lukoianov's τ picked there (0.5) is essentially noise:
  every τ is within ~1 of the others.

**ELS was dropped (2026-09-28).** It is implemented and validated in `core/local_softmax.py`.
Two σ values were finished on 200 test images (in `tables/local_softmax_heldout_els.npz`,
not plotted); the run was stopped for cost, about 24 min per σ (10¹⁰ candidate patches per
image, ~50× Lukoianov per image per setting).

## 5. FFHQ 32×32 and AFHQ 32×32 (2026-09-29)

**How to reproduce:**
- Runs: `scripts/run_local_softmax_faces.sh {ffhq32,afhq32}`, logged to
  `logs/local_softmax_{ffhq32,afhq32}.log`.
- Tables: `tables/local_softmax_heldout_{ffhq32,afhq32}.npz`.
- Figures: `figures/local_softmax_vs_sigma_{ffhq32,afhq32}[_ylin].{png,pdf,csv}` via
  `python scripts/local_softmax_heldout_plot.py --dataset ffhq32 [--ylinear]`.

**Setup.** The data are the 32×32 tensors of `~/Github/DiffusionLearningCurve`.
- **Pool** = images [0:10000], shared by Wiener, LS, Lukoianov and both global softmax
  curves.
- **Test** = 1000 images: FFHQ [60000:61000], AFHQ [10000:11000].
- **Validation** = the last 200 images.
- **Global train = eval** is evaluated on pool images [0:1000]; the pool contains each one.

**U-nets.** All are the same SongUNet (1 block/level, 128 channels, DSM, batch 256), from
`DL_Projects/DiffusionSpectralLearningCurve`:

| run | trained on | steps | test images |
|---|---|---|---|
| `FFHQ32_10000_*_split1` / `AFHQ32_10000_*_split1` | [0:10000] = the pool | 50k | unseen |
| `FFHQ32_30000_*_split1` | [0:30000] | 50k | unseen |
| `FFHQ32_*_saveckpt_fewsample_longtrain` | all 70k | 250k | SEEN (FFHQ has no held-out images) |
| `AFHQ32_*_saveckpt_fewsample` | all 15.8k | 50k | SEEN |

**The 10k U-nets memorise.** On FFHQ at σ = 0.127 each split scores 1.30 on its own
training images vs 21.9 on the other split's (Wiener 10.2). Held out they are worse than
linear for σ ≲ 1.

The AFHQ full-data U-net's 0.25× at σ = 0.127 is on images it trained on. At 15.8k images
that is likely partly recall, so it is not a generalisation number.

**Ratio to Wiener on the same images** (the `.csv` files have everything):

| σ | FFHQ Lukoianov | FFHQ LS | FFHQ U-net 30k (held out) | FFHQ U-net long (seen) | AFHQ Lukoianov | AFHQ LS | AFHQ U-net full (seen) |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.15 | 1.95 | 0.61 | 0.57 | 1.15 | 1.70 | 0.61 |
| 0.127 | 1.04 | 1.12 | 0.79 | 0.42 | 1.05 | 1.13 | 0.25 |
| 0.452 | 0.989 ± 0.003 | 1.05 | 0.86 | 0.56 | 1.01 | 1.02 | 0.33 |
| 1.61 | 0.997 ± 0.002 | 1.03 | 0.92 | 0.89 | 1.01 | 1.05 | 0.86 |
| 5 | 0.994 ± 0.002 | 1.02 | 1.00 | 0.99 | 1.00 | 1.02 | 1.00 |

**Findings:**
- **Global softmax.** It reproduces the CIFAR pattern: held out it is flat at about 115
  (FFHQ) and 124 (AFHQ) per image, and train = eval is 0 up to σ ≈ 0.6.
- **8-bit artefacts reproduce too.** LS levels off at 0.27–0.33 at low σ, and Lukoianov
  drops to about 0 at σ ≤ 0.001.
- **Lukoianov on FFHQ beats the held-out Wiener, narrowly but consistently,** at every σ
  from 0.45 to 10: 0.6–1.1 % lower, paired SE ≈ 0.2–0.3 %. This is the only dataset where
  either local softmax does. On AFHQ it is 1.00–1.01× there, and on CIFAR 1.01–1.02×.
- **The long-trained U-net converges to Wiener at both ends,** as EDM does on CIFAR:
  1.00× at σ = 0.0002 and 0.99–1.01× for σ ≥ 5.

## 6. 64 × 64: FFHQ64 and AFHQ64 (2026-10-01)

**How to reproduce:**
- Data: `scripts/prep_64px.py` caches EDM's `ffhq-64x64` / `afhqv2-64x64` zips as uint8 in
  `STORE_DIR/CondDiffNonlinearTheory/data/`.
- Held out: `scripts/run_local_softmax_64.sh {ffhq64,afhq64}` →
  `tables/local_softmax_heldout_{ffhq64,afhq64}.npz`, plotted with
  `local_softmax_heldout_plot.py --dataset ffhq64 [--ylinear] [--nets a,b]`.
- Sampling: `scripts/run_local_softmax_sampling_64.sh` → `tables/local_softmax_samples_*64.npz`.

**Setup.** Same contract as 32 px:
- Pool [0:10000]; 500 test and 100 validation images; 19 σ.
- Test images: FFHQ64 [60000:60500], AFHQ64 [10000:10500].
- Lukoianov's τ grid is the full 9 values for σ ≤ 0.01, and {0.02, 0.05, 0.1, 0.2, 0.3}
  above (see the run script).

**The covariance problem and the B variant.**
- At d = 12288 the 10k-pool covariance has rank 9999. The pool Wiener therefore zeroes about
  2300 directions and floors at 0.38 per image at σ = 0.0002, where the identity map scores
  5·10⁻⁴.
- `LUKB_COV` adds variant B, a full-rank covariance from more images, used for both:
  - the `wienerB` arm (a Wiener filter), which is the ratio reference at 64 px;
  - the `lukB` arm (Lukoianov masks).
- B's images: FFHQ [0:60000]; AFHQ [0:10000] + [10500:15703].
- Lukoianov A and B agree within about 0.01 (FFHQ64) and 0.01–0.12 (AFHQ64, largest at
  σ = 0.01) at every σ. The receptive-field estimate is not what limits the method.

**U-nets** (DiffusionSpectralLearningCurve SongUNets plus the official EDM 64-px pickles):

| U-net | trained on | test images |
|---|---|---|
| FFHQ64 10k | [0:10k] = the pool, 50k or 250k steps | unseen |
| FFHQ64 30k | [0:30k], 50k or 250k steps | unseen |
| FFHQ64 full long | all 70k, last ckpt ~221k steps | SEEN |
| EDM ffhq-64 | all 70k | SEEN |
| AFHQ64 full long | all images, ckpt ~221k steps | SEEN |
| EDM afhqv2-64 | all images | SEEN |

**Held-out results** (ratio to Wiener B):

| σ | FFHQ Luk A | FFHQ LS | FFHQ U-net 30k | FFHQ U-net 10k | FFHQ EDM | AFHQ Luk A | AFHQ EDM |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.61 | 2.14 | 0.53 | 0.59 | 0.49 | 1.20 | 0.44 |
| 0.127 | 1.07 | 1.13 | 0.52 | 1.42 | 0.44 | 1.04 | 0.47 |
| 0.452 | 1.00 | 1.06 | 0.67 | 1.19 | 0.56 | 1.00 | 0.52 |
| 1.61 | 1.01 | 1.06 | 0.84 | 0.93 | 0.78 | 1.02 | 0.72 |
| 10 | 1.00 | 1.02 | 1.01 | 1.01 | 0.99 | 1.00 | 1.00 |

**Held-out findings:**
- **Training longer worsens held-out error.** At σ = 0.127 the FFHQ64 U-nets score:

  | U-net | ratio |
  |---|---|
  | 10k, 50k steps | 1.42 |
  | 10k, 250k steps | 2.40 |
  | 30k, 250k steps | 0.81 |
  | 30k, 50k steps | 0.52 |

- **AFHQ64's U-nets saw the test images.** Their 0.27–0.47 is on seen images. They are also
  0.41–0.86× Wiener at σ ≤ 0.002, where FFHQ64's U-nets are about 1.0×, i.e. they partly
  recall those images.

**Heun-30 sampling, 256 shared seeds.** Final-sample R² against each U-net's sample from the
same seed:

| method | FFHQ64 vs EDM | FFHQ64 vs U-net 30k | FFHQ64 vs U-net 10k | AFHQ64 vs EDM | AFHQ64 vs full |
|---|---|---|---|---|---|
| Lukoianov | 0.755 | 0.606 | 0.754 | 0.643 | 0.490 |
| Wiener B | 0.740 | 0.586 | 0.768 | 0.651 | 0.492 |
| LS | 0.628 | 0.484 | 0.673 | 0.525 | 0.349 |
| global softmax | −0.04 | −0.13 | −0.10 | −0.31 | −0.50 |

**Sampling findings:**
- **None of the analytic samplers produce realistic images.** Wiener, Lukoianov and LS give
  face- or animal-shaped textures whose layout follows the U-net from the same seed.
- **Lukoianov adds little beyond linear here.** Its agreement with the U-nets is close to
  Wiener B's, sometimes slightly above and sometimes below.
- **The global softmax copies 100 % of the time.**
- **U-nets trained on the pool, or on all images, drift toward their training set.** Their
  median squared distance to the nearest pool image is lower than for held-out test images:

  | U-net | median d₁² | test images |
  |---|---|---|
  | FFHQ64 10k U-net | 399 | 506 |
  | AFHQ64 full U-net | 369 | 509 |

  None of them pass the copy test (d₁/d₂ < 1/3).
