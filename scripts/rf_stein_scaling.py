"""
rf_stein_scaling.py — test the WRITEUP'S OWN threshold table (Sec. "threshold table",
rows (i) and (ii)) on product measures, where Lemma 9 actually applies.

What the note predicts
----------------------
Lemma 9 gives, for a product measure satisfying Assumption 1,
    eps_d^2  <=  C (1 + ||kappa_3||_F^2) / d^2          (skewed coordinates)
    eps_d^2  <=  C (1 + ||kappa_4||_F^2) / d^3          (symmetric coordinates, kappa_3 = 0)
so with i.i.d.-type coordinates (||kappa_m||_F^2 ~ d):
    (i)  bounded skewness   -> eps_d^2 ~ 1/d    -> k_* = d/eps^2 ~ d^2
    (ii) symmetric          -> eps_d^2 ~ 1/d^2  -> k_* ~ d^3
These are one-sided (upper bounds on the defect), so the measured exponent should be
AT LEAST as steep as -1 resp. -2.

Design
------
Coordinates i.i.d. from a 2-component Gaussian mixture, standardized to mean 0 / unit
variance (so Sigma = I: uniformly PD and bounded, trivially).
    symmetric : 1/2 N(+m, v) + 1/2 N(-m, v)          -> kappa_3 = 0, kappa_4 != 0
    skewed    : p N(m1, v) + (1-p) N(m2, v), p != 1/2 -> kappa_3 != 0
Both are genuine product measures, so every hypothesis of Lemma 9 holds.

Estimator (this matters)
------------------------
Delta = Cov(x0, psi_s(theta^T x0)) - alphabar * theta is a SMALL DIFFERENCE of two O(1)
quantities, so the naive plug-in ||Delta_hat||^2 is biased upward by sum_a Var(Delta_hat_a),
which at large d swamps the signal (the bias scales like d/N while ||Delta||^2 ~ d^-1..d^-2).
We therefore use the SPLIT-HALF estimator: draw two independent sample halves, form
Delta_hat_1 and Delta_hat_2, and use
        <Delta_hat_1, Delta_hat_2>     (unbiased for ||Delta||^2, noise cross-term averages to 0).
The naive value is printed alongside so the size of the bias is visible.

    python scripts/rf_stein_scaling.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
SIGMA = float(os.environ.get('SIGMA', '1.0'))
DIMS = [int(x) for x in os.environ.get('DIMS', '8,16,32,64,128').split(',')]
N_THETA = int(os.environ.get('N_THETA', '256'))
N_X = int(os.environ.get('N_X', '4000000'))          # per half
CHUNK = int(os.environ.get('CHUNK', '200000'))


def coord_params(kind):
    """2-component mixture per coordinate, standardized to mean 0, variance 1."""
    if kind == 'symmetric':
        p, m1, m2, v = 0.5, 1.0, -1.0, 0.25
    else:                                  # skewed
        p, m1, m2, v = 0.25, 1.5, -0.5, 0.25
    mean = p * m1 + (1 - p) * m2
    m1c, m2c = m1 - mean, m2 - mean
    var = p * (m1c ** 2 + v) + (1 - p) * (m2c ** 2 + v)
    sc = 1.0 / np.sqrt(var)
    return p, m1c * sc, m2c * sc, v * sc ** 2


def cumulants(kind):
    p, m1, m2, v = coord_params(kind)
    mom = lambda r: p * _gauss_mom(m1, v, r) + (1 - p) * _gauss_mom(m2, v, r)
    m3, m4 = mom(3), mom(4)
    return m3, m4 - 3.0                      # kappa_3, kappa_4 (unit variance)


def _gauss_mom(mu, var, r):
    s = np.sqrt(var)
    if r == 3: return mu ** 3 + 3 * mu * s ** 2
    if r == 4: return mu ** 4 + 6 * mu ** 2 * s ** 2 + 3 * s ** 4
    raise ValueError


def sample_coords(kind, n, d, gen):
    p, m1, m2, v = coord_params(kind)
    b = (torch.rand(n, d, device=DEV, dtype=DT, generator=gen) < p)
    mu = torch.where(b, torch.tensor(m1, device=DEV, dtype=DT),
                     torch.tensor(m2, device=DEV, dtype=DT))
    return mu + np.sqrt(v) * torch.randn(n, d, device=DEV, dtype=DT, generator=gen)


def defect_half(kind, Theta, s, d, n_total, gen):
    """Accumulate Cov(x0, psi_s(u)) and alphabar over n_total fresh samples -> Delta (d,M)."""
    M = Theta.shape[0]
    acc_cov = torch.zeros(d, M, device=DEV, dtype=DT)
    acc_a = torch.zeros(M, device=DEV, dtype=DT)
    done = 0
    while done < n_total:
        nb = min(CHUNK, n_total - done)
        X = sample_coords(kind, nb, d, gen)                 # (nb,d), mean 0 var 1
        U = X @ Theta.T                                     # (nb,M)
        z = U / s[None, :]
        Phi = torch.special.ndtr(z)
        phi = torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        c0 = U * Phi + s[None, :] * phi                     # psi_s(u)
        acc_cov += X.T @ c0
        acc_a += Phi.sum(0)
        done += nb
    Cov = acc_cov / n_total                                 # Sigma = I
    abar = acc_a / n_total
    return Cov - abar[None, :] * Theta.T                    # (d,M)


def main():
    print(f"device={DEV}  sigma={SIGMA}  N_theta={N_THETA}  N_x={N_X:,} per half")
    print("Predicted by Lemma 9 / the threshold table:  skewed eps^2 ~ d^-1,"
          "  symmetric eps^2 ~ d^-2\n")
    out = {}
    for kind in os.environ.get('KINDS','skewed,symmetric').split(','):
        k3, k4 = cumulants(kind)
        print(f"=== {kind}:  per-coordinate kappa_3 = {k3:+.4f}, kappa_4 = {k4:+.4f} ===")
        print(f"{'d':>5} {'E||D||^2 (split-half)':>22} {'naive (biased)':>16} {'k_*=d/eps^2':>13}")
        es = []
        for d in DIMS:
            gen = torch.Generator(device=DEV); gen.manual_seed(1234 + d)
            Theta = torch.randn(N_THETA, d, device=DEV, dtype=DT, generator=gen) / np.sqrt(d)
            s = SIGMA * torch.linalg.norm(Theta, dim=1)
            D1 = defect_half(kind, Theta, s, d, N_X, gen)
            D2 = defect_half(kind, Theta, s, d, N_X, gen)
            unb = (D1 * D2).sum(0)                          # (M,) unbiased ||Delta||^2
            naive = 0.5 * ((D1 ** 2).sum(0) + (D2 ** 2).sum(0))
            e2 = float(unb.mean())
            se = float(unb.std()/np.sqrt(len(unb)))
            if e2 <= 3*se:
                print(f'{d:>5}   below MC resolution: {e2:.2e} +- {se:.2e}'); es.append(np.nan); continue
            es.append(e2)
            print(f"{d:>5} {e2:>22.3e} {float(naive.mean()):>16.3e} {d/e2:>13.3e}")
        lg = np.polyfit(np.log(DIMS), np.log(es), 1)
        out[kind] = (es, lg[0])
        print(f"  fitted exponent  d(log eps^2)/d(log d) = {lg[0]:+.3f}"
              f"   (predicted {'-1' if kind=='skewed' else '-2'})\n")
    print("Summary")
    for kind in out:
        pred = -1.0 if kind == 'skewed' else -2.0
        got = out[kind][1]
        ok = 'consistent (bound is one-sided: measured may be steeper)' if got <= pred + 0.25 \
             else 'SHALLOWER than predicted'
        print(f"  {kind:>10}: measured {got:+.3f} vs predicted {pred:+.1f}  -> {ok}")
    np.savez('tables/rf_stein_scaling.npz', dims=np.array(DIMS),
             **{f'eps2_{k}': np.array(v[0]) for k, v in out.items()},
             **{f'slope_{k}': v[1] for k, v in out.items()})


if __name__ == '__main__':
    main()
