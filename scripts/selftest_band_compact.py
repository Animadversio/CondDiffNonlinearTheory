"""Bounded checks for the compact Hermite noise tensors; no CIFAR data required.

By default, compare unconditional and conditional held-out losses with the independent
explicit-readout references at small sizes.  An optional pre-compaction module also checks
the full estimator at t=7, B=4/5/6 without the enormous explicit readout basis:

    python scripts/selftest_band_compact.py --device cpu
    python scripts/selftest_band_compact.py --device cuda --reference /path/to/old_band.py

The reference must export circulant2d_band_rf_mmse.  Both implementations receive identical
images, filters, labels, ridge and chunk sizes.  Wide-band checks validate equivalence to
the old estimator, not an independent brute-force solution at those sizes.  These small-c
cases test numerical correctness; their timing/peak memory is not a production benchmark.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from core.rf_circulant2d_band import (
    circulant2d_band_rf_mmse,
    circulant2d_band_rf_mmse_bruteforce,
    circulant2d_band_rf_mmse_bruteforce_cond,
)


def draw(H, W, Cin, c, t, device, seed=42):
    """Nonstationary images and different train/test class proportions and means."""
    rng = np.random.default_rng(seed)
    labels = np.arange(30) % 3
    test_labels = np.array([0] * 9 + [1] * 5 + [2] * 4)
    offsets = rng.normal(size=(3, Cin, H, W)) * 0.3
    position = np.linspace(-0.5, 0.5, H * W).reshape(1, 1, H, W)

    def images(lab, shift, scale):
        z = rng.normal(size=(len(lab), Cin, H, W))
        z += 0.3 * rng.normal(size=(len(lab), 1, 1, 1))
        return torch.tensor(scale * z ** 3 / 3 + offsets[lab] + position + shift,
                            dtype=torch.float64, device=device)

    x = images(labels, 0.0, 1.0)
    xt = images(test_labels, 0.25, 1.15)
    h = torch.zeros(c, Cin, H, W, dtype=torch.float64, device=device)
    h[:, :, :t, :t] = torch.tensor(rng.normal(size=(c, Cin, t, t)) / np.sqrt(Cin*t*t),
                                   dtype=torch.float64, device=device)
    gamma = torch.tensor(rng.normal(size=(c, 3)) / np.sqrt(3),
                         dtype=torch.float64, device=device)
    return (x, xt, h, torch.tensor(labels, device=device),
            torch.tensor(test_labels, device=device), gamma)


def compare(tag, actual, expected):
    keys = ('train', 'train_resid', 'test') if isinstance(actual, dict) else (None,)
    max_abs = max_rel = 0.0
    for key in keys:
        a = actual[key] if key is not None else actual
        b = expected[key] if key is not None else expected
        if not np.isfinite(a) or not np.isfinite(b):
            raise AssertionError(f'{tag}: nonfinite {key}: {a}, {b}')
        max_abs = max(max_abs, abs(a - b))
        max_rel = max(max_rel, abs(a - b) / max(1.0, abs(b)))
        if not np.isclose(a, b, rtol=1e-9, atol=1e-10):
            raise AssertionError(f'{tag}: {key}: {a} != {b}')
    print(f'{tag}: PASS  max abs={max_abs:.3e}  scaled={max_rel:.3e}', flush=True)


def check_case(shape, B, t, sigma, mode, device, reference=None):
    H, W, Cin, c = shape
    x, xt, h, lab, lab_test, gamma = draw(H, W, Cin, c, t, device)
    cond = {} if mode == 'unconditional' else dict(
        lab=lab, lab_test=lab_test,
        gam=None if mode == 'vu0' else gamma,
        class_centre=mode != 'feat')
    common = dict(lam=1e-6, device=device, x0_test=xt, **cond)
    chunks = dict(sample_chunk=11, pass1_chunk=13, freq_chunk=17, super_chunk=19)
    actual = circulant2d_band_rf_mmse(x, h, sigma, t, B, **common, **chunks)
    if reference is not None:
        expected = reference(x, h, sigma, t, B, **common, **chunks)
    elif mode == 'unconditional':
        expected = circulant2d_band_rf_mmse_bruteforce(x, h, sigma, B, **common)
    else:
        expected = circulant2d_band_rf_mmse_bruteforce_cond(x, h, sigma, B, **common)
    tag = f'{H}x{W} Cin={Cin} c={c} t={t} B={B} sigma={sigma} {mode}'
    compare(tag, actual, expected)
    return x, h, common, chunks, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print('Independent explicit-readout checks', flush=True)
    for shape, B, t, sigma in [((4, 6, 2, 2), 1, 2, 1.61),
                                ((5, 5, 1, 1), 2, 3, 0.621)]:
        for mode in ('unconditional', 'vu', 'feat', 'vu0'):
            x, h, common, chunks, heldout = check_case(
                shape, B, t, sigma, mode, args.device)
            # Training-only output and alternate chunk boundaries exercise both contracts.
            common.pop('x0_test')
            common.pop('lab_test', None)
            chunks.update(sample_chunk=7, pass1_chunk=9, freq_chunk=3, super_chunk=13)
            train = circulant2d_band_rf_mmse(x, h, sigma, t, B, **common, **chunks)
            compare(f'  {mode} train-only / different chunks', train, heldout['train'])
    if args.reference:
        spec = importlib.util.spec_from_file_location('band_before_compaction', args.reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.circulant2d_band_rf_mmse
        print('Before/after checks at larger bands and 7x7 taps', flush=True)
        cases = [((8, 8, 3, 3), 0, 3, 0.127),
                 ((13, 14, 2, 2), 4, 7, 0.127),
                 ((13, 13, 3, 2), 5, 7, 0.853),
                 ((13, 14, 3, 2), 6, 7, 5.0)]
        for shape, B, t, sigma in cases:
            for mode in ('unconditional', 'vu'):
                check_case(shape, B, t, sigma, mode, args.device, reference)
    print('All compact-noise checks passed.', flush=True)


if __name__ == '__main__':
    main()
