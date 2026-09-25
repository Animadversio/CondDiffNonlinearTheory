"""Machine-precision selftest for docs/band_readout.tex.

Checks, at small (d, c, B):

  (0) the boxed closed form
          L = Tr(Sigma_p0) - sum_k q_k^H P_k^{-1} q_k,
          (P_k)_{(a,s),(b,s')} = P^{(a,b)}_{k-s,k-s'},  (q_k)_{(a,s)} = Q^{(a)}_{k-s,k}
      against a BRUTE-FORCE real least squares over the explicit parameters
      {what^{a,s}_k} of W_a = sum_{|s|<=B} C_{a,s} E_s.  This is the load-bearing one:
      it validates the index formula, the block assembly and the Wirtinger solve at once.

  (1) Lemma 2 (modulation): the enlarged moments equal the B=0 moments of the
      modulated features psi_{a,s} = e_s (*) phi_a.

  (2) Proposition 1 (reality pairing): P_{d-k} = J conj(P_k) J with J:(a,s)->(a,-s),
      which is what keeps the d per-frequency solves independent.

  (3) B=0 reproduces the Section 4 formula.

Run:  python scripts/selftest_band_readout.py
"""
import numpy as np

TOL = 1e-8


def dft(d):
    j = np.arange(d)
    return np.exp(-2j * np.pi * np.outer(j, j) / d) / np.sqrt(d)


def circ(first):
    """Circulant matrix whose FIRST ROW is `first` (the convention of Section 1)."""
    return np.stack([np.roll(first, k) for k in range(len(first))])


def moments(phi, x0, F):
    """P^{(a,b)} = F Sigma_phi^{(a,b)} F^*,  Q^{(a)} = F Sigma_{phi_a,x0} F^*."""
    N, Fs = phi.shape[0], F.conj().T
    Pab = np.einsum('nad,nbe->abde', phi, phi) / N
    Qa = np.einsum('nad,ne->ade', phi, x0) / N
    return (np.einsum('jd,abde,ke->abjk', F, Pab, Fs),
            np.einsum('jd,ade,ke->ajk', F, Qa, Fs))


def blocks(Pf, Qf, k, c, offs, d):
    idx = [(a, s) for a in range(c) for s in offs]
    P = np.array([[Pf[a, b, (k - s) % d, (k - t) % d] for (b, t) in idx] for (a, s) in idx])
    q = np.array([Qf[a, (k - s) % d, k] for (a, s) in idx])
    return P, q, idx


def sample(d, c, sigma, N, seed):
    r = np.random.default_rng(seed)
    ker = np.exp(-np.minimum(np.arange(d), d - np.arange(d)) / 2.5)
    x0 = np.real(np.fft.ifft(np.fft.fft(r.standard_normal((N, d)), axis=1) * np.fft.fft(ker), axis=1))
    x0 -= x0.mean(0)
    y = x0 + sigma * r.standard_normal((N, d))
    h = r.standard_normal((c, d)) / np.sqrt(d)
    eps = r.standard_normal(c) * 0.4
    phi = np.stack([np.maximum(y @ circ(h[a]).T + eps[a], 0.0) for a in range(c)], 1)
    return x0, phi - phi.mean(0, keepdims=True)


def closed_form(d, c, B, x0, phi):
    F, offs = dft(d), list(range(-B, B + 1))
    Pf, Qf = moments(phi, x0, F)
    tot = 0.0
    for k in range(d):
        P, q, _ = blocks(Pf, Qf, k, c, offs, d)
        assert np.abs(P - P.conj().T).max() < 1e-10, f"P_{k} not Hermitian"
        tot += float((q.conj() @ np.linalg.solve(P, q)).real)
    return float((x0 ** 2).sum(1).mean()) - tot


def brute_force(d, c, B, x0, phi):
    """Least squares over the explicit class, with no Fourier argument used anywhere."""
    offs = list(range(-B, B + 1))
    E = {s: np.exp(2j * np.pi * s * np.arange(d) / d) for s in offs}
    cols = []
    for a in range(c):
        for s in offs:
            for m in range(d):
                e = np.zeros(d); e[m] = 1.0
                cols.append(circ(e) @ (E[s] * phi[:, a, :]).T)      # (d, N)
    A = np.stack(cols).reshape(len(cols), -1).T
    M = np.concatenate([A.real, -A.imag], 1)
    t = x0.T.reshape(-1)
    sol, *_ = np.linalg.lstsq(M, t, rcond=None)
    return float(((M @ sol - t) ** 2).sum() / x0.shape[0])


def check_lemma2_and_prop1(d, c, B, x0, phi):
    F, offs = dft(d), list(range(-B, B + 1))
    Pf, Qf = moments(phi, x0, F)
    N = phi.shape[0]
    e = np.exp(2j * np.pi * np.outer(offs, np.arange(d)) / d)
    psi = (phi[:, :, None, :] * e[None, None, :, :]).reshape(N, c * len(offs), d)
    psih = np.einsum('jd,nmd->nmj', F, psi)
    x0h = np.einsum('jd,nd->nj', F, x0)
    Ppsi = np.einsum('nmj,nlj->jml', psih, psih.conj()) / N
    qpsi = np.einsum('nmj,nj->jm', psih, x0h.conj()) / N
    dP = dq = dJ = 0.0
    for k in range(d):
        P, q, idx = blocks(Pf, Qf, k, c, offs, d)
        dP = max(dP, np.abs(P - Ppsi[k]).max()); dq = max(dq, np.abs(q - qpsi[k]).max())
        J = [idx.index((a, -s)) for (a, s) in idx]
        Pm, _, _ = blocks(Pf, Qf, (d - k) % d, c, offs, d)
        dJ = max(dJ, np.abs(Pm - P.conj()[np.ix_(J, J)]).max())
    return dP, dq, dJ


def main():
    print("selftest: band-limited readout of order B   (docs/band_readout.tex)")
    print("-" * 78)
    ok = True
    for (d, c, B) in [(12, 3, 0), (12, 3, 1), (12, 3, 2), (16, 2, 1), (9, 4, 1)]:
        x0, phi = sample(d, c, 0.7, 6000, seed=0)
        lc, lb = closed_form(d, c, B, x0, phi), brute_force(d, c, B, x0, phi)
        dP, dq, dJ = check_lemma2_and_prop1(d, c, B, x0, phi)
        bad = max(abs(lc - lb) / max(1.0, abs(lc)), dP, dq, dJ)
        ok &= bad < TOL
        print(f"d={d:3d} c={c} B={B} R={2*B+1:2d} | closed={lc:12.9f} brute={lb:12.9f} "
              f"|d|={abs(lc-lb):.1e} | lem2 P {dP:.1e} q {dq:.1e} | prop1 {dJ:.1e} "
              f"| {'OK' if bad < TOL else 'FAIL'}")
    # B=0 must equal the Section 4 result computed the Section 4 way (diagonals only)
    d, c = 12, 3
    x0, phi = sample(d, c, 0.7, 6000, seed=0)
    F = dft(d); Pf, Qf = moments(phi, x0, F)
    tot = sum(float((Qf[:, k, k].conj()
                     @ np.linalg.solve(Pf[:, :, k, k], Qf[:, k, k])).real) for k in range(d))
    s4 = float((x0 ** 2).sum(1).mean()) - tot
    b0 = closed_form(d, c, 0, x0, phi)
    ok &= abs(s4 - b0) < TOL
    print(f"{'':>4}B=0 vs Section 4 closed form: |diff| = {abs(s4-b0):.3e} "
          f"{'OK' if abs(s4-b0) < TOL else 'FAIL'}")
    print("-" * 78)
    print("ALL CHECKS PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
