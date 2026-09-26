# Compact Hermite noise storage

The conditional and unconditional band RF loss paths now sum the three Hermite-order
contributions before storing each Delta-resolved noise tensor.  Both use
`core/rf_circulant2d_band.py::circulant2d_band_rf_mmse`; no run flag is needed.

For a lag `m` and frequency difference `Delta`, the old representation stored

```
Bc[a,b,n*nL+m] = coef[n] * psi[a,b,m]**(n+1) * T[n,a,b,m,Delta]
```

and repeated the Fourier phase three times.  Since the phase depends on `m` but not `n`,
the new representation sums over `n` once and stores `(c,c,nL)` per Delta.  The negative-lag
transpose/phase relation and the separate exact-diagonal correction are unchanged.  The
same third-order approximation is evaluated with a different floating-point summation
order; bitwise-identical results are not expected.

## Memory and runtime

With complex128 storage, `nL=(2*t-1)^2` and `nD=((4*B+1)^2+1)//2`, the two stored noise
tensors require `2*nD*c*c*nL*16` bytes.  For `c=128`, `t=7`:

| B | Before, GiB | After, GiB | Train compact tensor + test lag accumulators, GiB |
|---|---:|---:|---:|
| 4 | 35.90 | 11.97 | 14.98 |
| 5 | 54.71 | 18.24 | 22.85 |
| 6 | 77.49 | 25.83 | 32.37 |

These are component sizes, **not total peak memory**.  Pass 1 still accumulates all three
Hermite orders in `Tre/Tim`, costing `(2*nD-1)*3*((nL+1)//2)*c*c*8` bytes per split.  During
the test pass the compact train tensor is also resident.  Rolled feature buffers, the
Delta-slice being assembled, input transforms, frequency covariance matrices, identity
matrix and solver workspaces add further memory.  The old measured peak multiplier of
2.12 times one uncompressed noise tensor does not apply to the compact representation.
Historical sizing tables in the run-script comments describe that older representation.

The phase contraction uses one third as many entries.  Feature generation, lag accumulation
and solve dimensions are unchanged.  Both train and test noise tensors remain resident,
preserving the existing single sweep of output frequencies.  The shared driver's `sizing()`
uses the smaller stored tensors when choosing frequency chunks, so it can also reduce
repeated feature generation.  It remains a heuristic, not a guarantee that a cell fits a
particular card; frequency chunks cannot fix a pass-1 OOM.  Whole-run CUDA speed and peak
memory need a production-scale measurement.

## Validation

Run the bounded independent reference checks without CIFAR data:

```bash
python scripts/selftest_band_compact.py --device cpu
```

They compare all three loss outputs with the explicit real readout at `B=1,2`, exercising
unconditional RF and conditional `vu`, `feat`, and `vu0`, shifted test distributions,
different class proportions, training-only calls, and different chunk boundaries.

To compare against a saved pre-compaction version of `core/rf_circulant2d_band.py`:

```bash
python scripts/selftest_band_compact.py --device cuda --reference /path/to/old_band.py
```

This additionally compares unconditional and conditional `vu` at `B=0`, plus `B=4,5,6`
with `t=7`, odd/even spatial dimensions and low/middle/high noise.  The wide-band cases
use small channel counts to keep validation practical; they check equivalence to the old
estimator rather than an independent brute-force solution at those sizes.  The existing
`selftest_band_cond()` also checks the plain-model reduction and zero-ridge identities.

Local validation passed on CPU with PyTorch 2.14.0: the independent reference checks,
training-only/chunk checks, all before/after cases above, and the existing
`selftest_band_cond()` suite.  The largest before/after absolute loss difference was
`1.262e-11` (scaled difference `6.015e-14`).  CUDA and production-scale peak memory/runtime
have not been benchmarked for this change.
