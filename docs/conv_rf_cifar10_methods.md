# CIFAR-10 Convolutional RF MMSE Experiment

This experiment tests whether fixed random convolutional ReLU features improve
denoising MMSE in the noise range where pretrained EDM denoisers beat the linear
Wiener baseline.

## Model

For CIFAR images `x0 in [0,1]^(3x32x32)` and noisy observations

```text
y = x0 + sigma z,   z ~ N(0, I),
```

the random feature map is

```text
phi(y) = ReLU(circular_conv2d(y, H_rf) + b_rf),
```

where `H_rf` has `F` random filters of size `3x3`, `5x5`, or `7x7`.
The readout is the best linear circular convolution from features to pixels.
No SGD is needed: with circular boundary conditions, the optimal readout
decouples into one small `F x F` ridge solve per 2D Fourier frequency.

The script also evaluates `[y; phi(y)]`, which asks whether nonlinear random
conv features improve over the convolutional linear denoiser.

It also evaluates a fixed full-Wiener residual hybrid:

```text
xhat(y) = full_dense_Wiener(y) + conv_readout(phi(y)).
```

This keeps the dense Wiener branch fixed, streams the residual target
`x0 - full_dense_Wiener(y)`, and solves the best convolutional RF readout for
that residual. This directly measures whether conv RF contains nonlinear
information beyond the full dense linear denoiser.

## Bias Modes

The default reported curve uses `bias_mode="channel"`:

- only the DC frequency is centered;
- this corresponds to a true convolutional denoiser with one free bias per
  output channel.

The script also caches `bias_mode="free_spatial"`:

- every frequency is centered;
- this allows an arbitrary spatial bias and is useful when comparing against
  dense affine covariance estimators.

## Noise Grid

The pilot grid focuses on the EDM-interesting region:

```text
sigma = [0.3, 0.45, 0.65, 0.85, 1.2, 1.7, 2.2]
```

Existing CIFAR tables show EDM beats the full linear Wiener denoiser most
clearly around `sigma ~= 0.3-1.7`, so this is where conv RF has room to show a
meaningful nonlinear gain.

## Outputs

Summary arrays are cached in:

```text
tables/conv_rf_cifar10/conv_rf_cifar10_<tag>.npz
```

Cluster runs should set `ARTIFACT_DIR` to global lab storage, e.g.

```text
/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/CondDiffNonlinearTheory/conv_rf_cifar10
```

The `.npz` contains plot-ready arrays:

```text
conv_linear_channel
conv_linear_free_spatial
conv_rf_channel
conv_rf_free_spatial
linear_plus_conv_rf_channel
linear_plus_conv_rf_free_spatial
dense_wiener
dense_wiener_plus_conv_rf_channel
dense_wiener_residual_trace_channel
elapsed_combo_seconds
elapsed_dense_residual_seconds
```

The plotting cache is intentionally enough to restyle figures without repeating
feature extraction or covariance accumulation.

## Commands

Local synthetic estimator smoke test:

```bash
python scripts/conv_rf_cifar10.py --smoke --device cpu
```

Tiny CIFAR CPU path test:

```bash
python scripts/conv_rf_cifar10.py \
  --n_samples 64 --n_noise 1 --batch_size 32 --num_workers 0 \
  --sigmas 0.85 --num_filters 2 4 --kernel_sizes 3 --seeds 0 \
  --artifact_dir /tmp/conv_rf_test --figure_dir /tmp/conv_rf_fig \
  --tag smoke_local --device cpu
```

Slurm smoke:

```bash
sbatch --export=ALL,MODE=smoke scripts/run_conv_rf_cifar10.sh
```

Slurm pilot:

```bash
sbatch --export=ALL,MODE=pilot scripts/run_conv_rf_cifar10.sh
```

Slurm full sweep:

```bash
sbatch --export=ALL,MODE=full scripts/run_conv_rf_cifar10.sh
```

Monitor progress:

```bash
tail -f logs/conv_rf_cifar10_<jobid>.out
watch -n 30 'ls -lh /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/CondDiffNonlinearTheory/conv_rf_cifar10'
```

Each long setting prints elapsed time and an ETA. If the pilot ETA is
unreasonable, profile whether time is dominated by convolution/FFT, batched
linear solves, or data loading before launching the full sweep.
