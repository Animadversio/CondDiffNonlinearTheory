"""
rf_rho_star_measured.py — replace the unquantified (KMC) constants c_0, A by DIRECT
MEASUREMENT of the quantity they exist to bound.

The problem
-----------
K_tensor = c_0 d^{n_0} (log d)^{-A} cannot be evaluated numerically. c_0 and A come from
the kernel-matrix-concentration statement (KMC) that Lemma (tensor Gram conditioning)
quotes from [MMM21, MZ22]; those are asymptotic results whose absolute constants are not
tracked, and the writeup itself flags this as "the only unverified external input to this
document besides Mehler's identity". No library exposes them, because nobody computes them.

Two constant-free substitutes
-----------------------------
(1) EXACT ALGEBRAIC CEILING, stated in the note itself: R^{o n} is the Gram matrix of the k
    tensors hat_theta_j^{otimes n} in Sym^n(R^d), so
        rank R^{o n} <= binom(d+n-1, n)   =>   lmin(R^{o n}) = 0 for k above it.
    This is exact and constant-free. It UPPER bounds any admissible K_tensor.
(2) DIRECT MEASUREMENT. (KMC) is invoked only to certify rho_* = min_n lmin(R^{o n}) is
    ~1 (the assembly needs rho_* bounded below; the proof uses rho_* >= 1/2). rho_* is a
    computable function of a random draw, so measuring it at the k of interest replaces the
    constants entirely -- with the caveat that this certifies the realized draw rather than
    proving a high-probability statement.

    python scripts/rf_rho_star_measured.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from math import comb

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
DIMS = [int(x) for x in os.environ.get('DIMS', '784,3072').split(',')]
KS = [int(x) for x in os.environ.get('KS', '1024,2048,4096,8192').split(',')]
BAND = [int(x) for x in os.environ.get('BAND', '2,3').split(',')]


def main():
    print("(1) EXACT algebraic ceiling  rank R^on <= binom(d+n-1,n):")
    for d in DIMS:
        print("    d=%-6d " % d + "  ".join(f"n={n}: {comb(d+n-1,n):,}" for n in BAND))
    print("\n(2) MEASURED rho_* = min_n lmin(R^on)   [(KMC) asserts this is ~1]")
    print(f"{'d':>6} {'k':>7} {'k/d':>6} " + " ".join(f"{'lmin(R^o'+str(n)+')':>13}" for n in BAND)
          + f" {'rho_*':>8} {'>=1/2?':>7}")
    for d in DIMS:
        for k in KS:
            g = torch.Generator(device=DEV); g.manual_seed(0)
            T_ = torch.randn(k, d, device=DEV, dtype=DT, generator=g)
            T_ = T_ / torch.linalg.norm(T_, dim=1, keepdim=True)
            R = T_ @ T_.T; R.fill_diagonal_(1.0)
            ls = []
            for n in BAND:
                M = R.clone()
                for _ in range(n - 1):
                    M = M * R
                ls.append(float(torch.linalg.eigvalsh(M).min()))
            rho = min(ls)
            print(f"{d:>6} {k:>7} {k/d:>6.1f} " + " ".join(f"{v:>13.5f}" for v in ls)
                  + f" {rho:>8.5f} {'YES' if rho >= 0.5 else 'no':>7}")
            del R, T_; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
