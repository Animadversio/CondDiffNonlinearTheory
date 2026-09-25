"""Why the band: at B=0 only the DIAGONAL of F Sigma F^* reaches the loss.

Produces the table in docs/band_readout.tex, "Motivation: only the diagonal reaches
the loss".  Three ensembles with the SAME marginal smoothness but different
non-stationarity:

  stationary      exact Z_d symmetrisation -> Chat is diagonal -> nothing to buy
  smooth          centred envelope (CIFAR-like) -> mass on the NEAR off-diagonals
  comb            period-4 texture             -> mass at Delta = d/4 only

and both relaxations priced on each.  The point is the converse pair: the band reads
the near off-diagonals and the comb reads Delta in (d/p)Z, so each is blind to the
other's ensemble.  Neither buys anything on a stationary source, which is the Lemma.

Run:  python scripts/band_offdiag_motivation.py
"""
import numpy as np

D, SIG, N = 64, 0.45, 60000
RNG = np.random.default_rng(0)


def ensemble(n, kind):
    ker = np.exp(-np.minimum(np.arange(D), D - np.arange(D)) / 4.0)
    x = np.real(np.fft.ifft(np.fft.fft(RNG.standard_normal((n, D)), axis=1)
                            * np.fft.fft(ker), axis=1))
    if kind == 'smooth':
        x *= (1.0 + 1.6 * np.exp(-((np.arange(D) - D / 2) / (D / 6.0)) ** 2))[None, :]
    elif kind == 'comb':
        x *= (1.0 + 1.2 * (np.arange(D) % 4 == 0))[None, :]
    elif kind == 'stationary':
        x = np.stack([np.roll(v, s) for v, s in zip(x, RNG.integers(0, D, n))])
    return x - x.mean(0)


def chat(x):
    Xh = np.fft.fft(x, axis=1, norm='ortho')
    return (Xh.conj().T @ Xh).conj() / x.shape[0]


def _solve(C, wins, sig):
    tr, red = float(np.trace(C).real), 0.0
    for k in range(D):
        w = wins(k)
        P = C[np.ix_(w, w)] + sig ** 2 * np.eye(len(w))
        q = C[w, k]
        red += float((q.conj() @ np.linalg.solve(P, q)).real)
    return tr - red


def band(C, B, sig):
    return _solve(C, lambda k: (k + np.arange(-B, B + 1)) % D, sig)


def comb(C, p, sig):
    return _solve(C, lambda k: (k + np.arange(p) * (D // p)) % D, sig)


def main():
    print(f"d={D}  sigma={SIG}  N={N}\n")
    print(f"{'quantity':>28}" + "".join(f"{k:>18}" for k in ('stationary', 'smooth', 'comb')))
    print("-" * 82)
    res = {}
    for kind in ('stationary', 'smooth', 'comb'):
        C = chat(ensemble(N, kind))
        dg = float((np.abs(np.diag(C)) ** 2).sum())
        res[kind] = {
            'm1': float((np.abs(np.diag(C, 1)) ** 2).sum()) / dg,
            'm2': float((np.abs(np.diag(C, 2)) ** 2).sum()) / dg,
            'B0': band(C, 0, SIG), 'B1': band(C, 1, SIG),
            'B2': band(C, 2, SIG), 'p4': comb(C, 4, SIG)}
    for lab, f in (('off-diag mass, Delta=1', lambda r: f"{r['m1']:.1e}"),
                   ('off-diag mass, Delta=2', lambda r: f"{r['m2']:.1e}"),
                   ('toll removed, band B=1', lambda r: f"{r['B0']-r['B1']:+.4f}"),
                   ('toll removed, band B=2', lambda r: f"{r['B0']-r['B2']:+.4f}"),
                   ('toll removed, comb p=4', lambda r: f"{r['B0']-r['p4']:+.4f}")):
        print(f"{lab:>28}" + "".join(f"{f(res[k]):>18}" for k in ('stationary', 'smooth', 'comb')))
    print("-" * 82)
    s = res['stationary']
    assert abs(s['B0'] - s['B2']) < 1e-3, "LEMMA VIOLATED: band buys on a stationary source"
    assert abs(s['B0'] - s['p4']) < 1e-3, "LEMMA VIOLATED: comb buys on a stationary source"
    assert res['smooth']['B0'] - res['smooth']['B1'] > 0.3, "band should buy on smooth"
    assert res['comb']['B0'] - res['comb']['B2'] < 1e-2, "band should be blind to a comb texture"
    print("Lemma holds: on a shift-stationary source Chat is diagonal and neither "
          "relaxation buys anything.")


if __name__ == '__main__':
    main()
