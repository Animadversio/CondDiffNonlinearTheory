# Conditional random-feature denoiser on G = Z_H x Z_W

This is the math behind the `lab=` / `gam=` arguments added to
`core.rf_circulant2d.circulant2d_rf_mmse`.  It extends the label conditioning that
already existed in the GMM-era 1-D path (`core/rf_circulant.py`,
`core/rf_gmm_estimators.py::stein_covariances`) to the 2-D equivariant estimator.
Nothing about the unconditional estimator changes; with `lab=None` (the default) every
number it returns is bit-identical to before.

## 1. The model

Unconditional (what the file computed before):

    phi_a[u] = relu( sum_{ch, v} h_a[ch, v] y[ch, u + v] ),    y = x0 + sigma Z

Conditional, with a one-hot (or otherwise fixed) label vector U in R^{n_cls} per image:

    phi_a[u] = relu( sum_{ch, v} h_a[ch, v] y[ch, u + v]  +  <gamma_a, U> )

so the label enters as a **pre-activation bias**, exactly as in
`stein_covariances`, where the single conditional line is

    M = x0 @ Theta.T + (U @ Gamma.T if conditional else 0.0)

### 1.1 gamma is drawn once per feature PLANE, shared across all shifts

`core/rf_circulant.py::build_circulant_gamma` draws one `gamma_a ~ N(0, I/n_cls)` per
block and replicates it across the d rows of that block.  The 2-D analogue is one
`gamma_a` per feature plane `a`, shared across all |G| = H*W spatial shifts `u`.

**This sharing is load-bearing, not a convenience.**  `<gamma_a, U>` is constant over the
spatial index `u`, so the bias commutes with the Z_H x Z_W action just as the convolution
does.  The design stays G-equivariant, the DFT still block-diagonalises the normal
equations, and the estimator remains one c x c solve per frequency.  A bias that varied
with `u` would destroy the fast path entirely (the Gram would be K x K, K = c|G|).

### 1.2 The readout stays UNCONDITIONAL

In `stein_covariances` the data mean `mu = x0.mean(0)`, the feature mean `G.mean(0)` and
hence the readout `W` and bias `beta = mu - W gmean` are all computed **without** the
label.  This file follows that convention exactly: conditioning lives entirely in the
feature map `phi`, never in the readout.

That is a modelling choice worth stating out loud, because a class-conditional readout
bias would be a strictly LARGER model class and would score better.  The quantity
computed here is "how much does telling the random features the class buy?", not "how
much does telling the whole denoiser the class buy?".

## 2. Why no new math is needed

Write the pre-activation of feature (a, u) on a sample with label U:

    A_{a,u} = (h_a * x0)[u] + <gamma_a, U>  +  sigma (h_a * Z)[u]
            = M_{a,u} + sigma (h_a * Z)[u],      M_{a,u} := (h_a * x0)[u] + <gamma_a, U>

The label contributes a **deterministic** shift, so:

* **The noise scale is unchanged.**  `sigma (h_a * Z)[u]` has sd `s_a = sigma ||h_a||`
  whatever the bias is; the code's `s = sigma * nrm` needs no modification.
* **The noise correlation is unchanged.**  `rho_{(a,u),(b,u')} = psi_ab(u - u')` with
  `psi_ab(m) = sum_{ch,v} h_a[ch,v] h_b[ch,v+m] / (||h_a|| ||h_b||)` is built from `h`
  alone.  The lag index set, the lag symmetry `S_ab(-m) = S_ba(m)` and the size of the
  lag tensor are therefore all identical.
* **Every Stein quantity is a pointwise function of (M, s).**  G = M Phi(M/s) + s phi(M/s),
  the three Mehler coefficient fields `C_n`, the exact conditional variance
  `(M^2 + s^2) Phi + M s phi - G^2` and its truncated counterpart are all evaluated
  elementwise at `M`.  Shifting `M` is all that conditioning does to them.
* **The Hermite/Mehler expansion is the same function of (M, s, rho).**  The expansion is
  a statement about a pair of jointly Gaussian variables with means `M_i, M_j`, sds
  `s_i, s_j` and correlation `rho_ij`; it never assumed the means came from a convolution.
* **The lag Grams average over samples, each carrying its own label.**  `S^n_ab(m)` is a
  sample average of `C_n[a, u+m] C_n[b, u]`, and each sample's `C_n` already includes that
  sample's bias, so `E_{x0,U}[Cov(phi | x0, U)]` is estimated correctly with no change to
  the assembly.

Consequently the Hermitian fold, the frequency decoupling and the per-frequency c x c
solve all survive verbatim, and the implementation change is a single `M = M + bias`.

## 3. Where it enters the code

Two injection points, one per code path:

* `core/rf_circulant2d.py::circulant2d_rf_mmse` -> `feats()`, immediately after the
  `irfft2` that produces `M` of shape `(c, nb, D)` and before `z = M / sa`.  The bias
  `(gam @ lab.T)` has shape `(c, ns)` and broadcasts over the spatial axis `D`.
* `core/rf_circulant2d.py::_kk_moments`, after `M = xf @ Th.T`.  Here the rows of the
  explicit design are ordered `(a, u1, u2)` with `a` **outermost** (see
  `_explicit_theta`), so the explicit (K, n_cls) label matrix is
  `gam.repeat_interleave(H*W, dim=0)` -- the same replication, written out.

The brute-force reference therefore exercises the conditioning independently: it
materialises the K x K covariance of the conditioned features and solves the constrained
least squares over an explicit real BCCB parameterization.

### 3.1 Held-out evaluation

`lab_test=` accompanies `x0_test=`.  The labels of the test split are the test split's own,
but `mu`, `gmean` and `w_f` still come from train -- centring is part of the model (module
docstring, "CENTRING IS PART OF THE MODEL").  `gamma` is part of the random DESIGN, like
`h`, so it is shared across the splits and never redrawn.

## 4. Validation

`selftest2d` in the same file:

* the five pre-existing unconditional cases must return **bit-identical** numbers
  (defaults `None` => the new code is not reached);
* `gam = 0` with a nontrivial `lab` must reproduce the unconditional number at exactly
  `0.0` relative error -- this proves the plumbing perturbs nothing;
* new conditional cases (n_cls = 2, 3, 5) with a **label-dependent data shift**, so that
  `Cov(x0, phi)` genuinely depends on the label and a bias that was silently dropped would
  show up, checked against the brute force on `train`, `train_resid` and `test`;
* a conditional `lam = 0`, `x0_test = x0`, `lab_test = lab` identity check, which pins the
  sign and normalisation of the held-out assembly without the brute force.

Tolerance 1e-9 relative throughout, matching the unconditional suite.

## 5. What this does NOT give you

* **There is no conditional linear baseline on pixels.**  `scripts/rf_pixel_heldout.py`'s
  `linear_split` is an unconditional Wiener filter, and every pixel table in this project
  (RF, dense, band, EDM) is unconditional.  A conditional RF number has nothing to be
  scored against until a class-conditional linear denoiser exists -- probably the larger
  half of the work.
* **The pixel drivers currently discard labels.**  `scripts/rf_pixel_heldout.py::load`
  does `for xb, _ in dl`, so any conditional pixel experiment needs loader plumbing as
  well as the estimator.

Neither is touched here; this change is the estimator only.
