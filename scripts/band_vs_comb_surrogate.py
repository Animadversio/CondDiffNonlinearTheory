"""BAND vs COMB: is `period-P` the wrong relaxation axis for the equivariant readout?

The repo prices the readout toll by relaxing Z_d -> Z_{d/P} (period-P sharing), i.e. the
filter is allowed to depend on position mod P.  In the Fourier domain that couples output
frequency f to inputs f + j*(d/P) -- a COMB, i.e. the filter's spatial variation is
HIGH-frequency (period P pixels).

The non-stationarity of images is the opposite: smooth, whole-image scale (centre vs
border, sky above).  The matching relaxation is a BAND: output f reads inputs f+Delta,
|Delta| <= B, which is exactly "the filter varies smoothly over the image, band-limited to
B spatial modes".  A band is NOT a subgroup of Z_d, which is why a search over subgroups
misses it.

Both relaxations decouple the readout least squares into d independent small solves, so
both are equally compatible with the fast path.

This script measures both on an ensemble with realistic natural-image statistics.
"""
import numpy as np, itertools, sys

RNG = np.random.default_rng(0)
H = W = 32
D = 3 * H * W
TR_TARGET = 191.52          # CIFAR-10 raw pixels, N=10^4, [0,1] units (repo value)
SIGS = [0.127, 0.452, 1.610, 5.0]


# ---------------------------------------------------------------- data ------
def photos():
    import skimage.data as SD
    names = ['astronaut', 'coffee', 'chelsea', 'cat', 'rocket', 'hubble_deep_field',
             'immunohistochemistry', 'retina']
    out = []
    for n in names:
        a = getattr(SD, n)().astype(np.float64) / 255.0
        if a.ndim == 3 and a.shape[2] == 3:
            out.append(a)
    return out


def crops(imgs, n):
    """n random 32x32x3 crops from the photo set."""
    out = np.empty((n, H, W, 3))
    for i in range(n):
        a = imgs[RNG.integers(len(imgs))]
        r = RNG.integers(a.shape[0] - H); c = RNG.integers(a.shape[1] - W)
        out[i] = a[r:r + H, c:c + W]
    return out


def ensemble(n, kind):
    """kind='stationary' : plain crops, then a random cyclic shift -> exactly shift-stationary.
       kind='framed'     : centred-object composite -> CIFAR-like smooth non-stationarity."""
    if kind == 'stationary':
        X = crops(photos(), n)
        sh = RNG.integers(0, H, size=(n, 2))
        for i in range(n):                                   # exact Z_32 x Z_32 symmetrisation
            X[i] = np.roll(X[i], (sh[i, 0], sh[i, 1]), axis=(0, 1))
    else:
        P = photos()
        bg, fg = crops(P, n), crops(P, n)
        y, x = np.mgrid[0:H, 0:W]
        rr = np.sqrt(((y - 15.5) / 15.5) ** 2 + ((x - 15.5) / 15.5) ** 2)
        M = (1.0 / (1.0 + np.exp((rr - 0.62) / 0.16)))[..., None]     # soft centred mask
        tilt = (1.0 + 0.45 * ((y - 15.5) / 15.5))[..., None]          # sky-above asymmetry
        g = RNG.lognormal(0.0, 0.45, size=(n, 1, 1, 1))               # per-image object gain
        X = (1 - M) * bg * tilt + M * fg * g
    X = X.transpose(0, 3, 1, 2).reshape(n, D)                # channel-major, as in the repo
    X -= X.mean(0)
    X *= np.sqrt(TR_TARGET / (X ** 2).sum(1).mean())         # match Tr(Sigma) to CIFAR's
    return X


# ------------------------------------------------------- linear baselines ---
def wiener(Sig, sig):
    ev = np.linalg.eigvalsh(Sig)
    return float((sig ** 2 * ev / (ev + sig ** 2)).sum())


def period2d(X, TR, p, sig):
    """Repo's relaxation: A equivariant to spatial shifts by p, free 3-channel mixing."""
    n, m, B = X.shape[0], H // p, 3 * p * p
    V = X.reshape(n, 3, m, p, m, p).transpose(0, 2, 4, 1, 3, 5).reshape(n, m, m, B)
    F = (np.fft.fft2(V, axes=(1, 2)) / m).reshape(n, m * m, B).transpose(1, 0, 2)
    P = np.einsum('fna,fnb->fab', F, F.conj()) / n
    I = np.eye(B)[None] * (sig ** 2)
    return TR - float(np.einsum('fab,fba->', P, np.linalg.solve(P + I, P)).real)


# --------------------------------------------------- the proposed relaxation -
def fourier_cov(X):
    """Chat = E[ xhat xhat^H ] over the 2-D DFT, index (channel, fr, fc)."""
    n = X.shape[0]
    Xh = np.fft.fft2(X.reshape(n, 3, H, W), axes=(2, 3), norm='ortho').reshape(n, D)
    return (Xh.conj().T @ Xh).conj() / n          # Chat[a,b] = E[xh_a conj(xh_b)]


def band_index(B):
    """(win, out, nb): win is (1024, 3nb) input coords, out is (1024, 3) output coords."""
    offs = np.array([(a, b) for a in range(-B, B + 1) for b in range(-B, B + 1)])
    nb = len(offs)
    fr, fc = np.divmod(np.arange(H * W), W)                            # (1024,)
    gr = (fr[:, None] + offs[None, :, 0]) % H                          # (1024, nb)
    gc = (fc[:, None] + offs[None, :, 1]) % W
    base = gr * W + gc                                                 # (1024, nb)
    ch = np.arange(3) * (H * W)
    win = (base[:, :, None] + ch[None, None, :]).reshape(H * W, 3 * nb)
    out = (fr * W + fc)[:, None] + ch[None, :]
    return win, out, nb


def band2d(Chat, TR, B, sigs, chunk=256):
    """A = sum_s diag(e_s) * BCCB(a_s), s in a 2-D band |s_r|,|s_c| <= B.
    Output frequency f reads inputs f+Delta for Delta in the band; each parameter belongs
    to exactly one f, so the solve is 1024 independent blocks of size 3(2B+1)^2.
    B=0 is exactly the strictly-equivariant (period2d p=1) model."""
    win, out, nb = band_index(B)
    m = 3 * nb
    red = {s: 0.0 for s in sigs}
    I = np.eye(m)[None]
    for lo in range(0, H * W, chunk):
        w, o = win[lo:lo + chunk], out[lo:lo + chunk]
        P0 = Chat[w[:, :, None], w[:, None, :]]                        # (nf, m, m)
        q = Chat[w[:, :, None], o[:, None, :]]                         # (nf, m, 3)
        for s in sigs:
            r = np.linalg.solve(P0 + (s * s) * I, q)
            red[s] += float(np.einsum('fam,fam->', q.conj(), r).real)
    return [TR - red[s] for s in sigs], 3 * H * W * m, nb


# ------------------------------------------------------------------ main ----
def run(kind, n=4000):
    X = ensemble(n, kind)
    Sig = (X.T @ X) / n
    TR = float(np.trace(Sig))
    Chat = fourier_cov(X)
    assert abs(float(np.trace(Chat).real) - TR) < 1e-6 * TR, "Parseval check failed"

    lin = {s: wiener(Sig, s) for s in SIGS}
    print(f"\n{'='*104}\nENSEMBLE '{kind}'   N={n}  d={D}  Tr(Sigma)={TR:.3f}\n{'='*104}")
    print(f"{'model':>26} {'params':>11} |" + "".join(f"{f'sg={s}':>18}" for s in SIGS))
    print(f"{'free Wiener':>26} {D*D:>11,} |"
          + "".join(f"{lin[s]:>10.3f}{0.0:>+8.3f}" for s in SIGS))

    print(f"{'-'*104}\n  COMB  (period-p, the repo's dial)")
    for p in (1, 2, 4, 8, 16):
        L = [period2d(X, TR, p, s) for s in SIGS]
        print(f"{f'period2d p={p}':>26} {3*D*p*p:>11,} |"
              + "".join(f"{a:>10.3f}{a-lin[s]:>+8.3f}" for s, a in zip(SIGS, L)))
        if p == 1:
            ref = L

    print(f"{'-'*104}\n  BAND  (smoothly space-varying filter)")
    for B in (0, 1, 2, 3):
        L, np_, nb = band2d(Chat, TR, B, SIGS)
        tag = 'CHECK vs p=1' if B == 0 else ''
        print(f"{f'band2d B={B} ({nb} taps)':>26} {np_:>11,} |"
              + "".join(f"{a:>10.3f}{a-lin[s]:>+8.3f}" for s, a in zip(SIGS, L)) + f"  {tag}")
        if B == 0:
            err = max(abs(a - b) for a, b in zip(L, ref))
            print(f"{'':>26} {'':>11}  |  max |band B=0 - period p=1| = {err:.2e}"
                  f"   {'OK' if err < 1e-8 else 'MISMATCH'}")


if __name__ == '__main__':
    for kind in ('stationary', 'framed'):
        run(kind)
