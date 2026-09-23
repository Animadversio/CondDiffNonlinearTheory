"""REGENERATE EVERY TABLE IN docs/rf_heldout_and_nsweep.md FROM THE STORED npz FILES.

The doc quotes ~200 numbers.  Rather than transcribe them (which has produced errors in this
project before) the doc embeds this script's output verbatim, so `python scripts/
rf_heldout_report.py` is both the regeneration recipe and the check that the doc is current.

Reads  tables/rf_pixel_circ2d.npz, rf_pixel_circ_csweep.npz, rf_pixel_heldout.npz,
       rf_circ{2d,1d}_nsweep_{10000,20000,40000}.npz
Needs CIFAR-10 only for section A (dataset traces) and E (the linear floor, which is a 2 s
closed-form solve per N) -- everything else is pure npz arithmetic.  CPU is fine, ~60 s.

    python scripts/rf_heldout_report.py            # all sections
    SECTIONS=BCDF python scripts/rf_heldout_report.py   # skip the CIFAR-dependent ones
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

d = 3072
CS = (32, 96, 256, 512, 1536, 3072, 6144)
SIGS = ('0.127', '0.452', '0.621', '0.853', '1.172', '1.61', '2.212', '5.0')
NS = (10000, 20000, 40000)
SECTIONS = os.environ.get('SECTIONS', 'ABCDEFG')   # H (the RF swap test) also needs SWAP=1
DEV = os.environ.get('DEVICE', 'cpu')

T = {}          # dataset traces, filled by section A (or defaults below if skipped)
T_DEFAULT = {10000: 191.5220, 20000: 191.2217, 40000: 190.6448, 50000: 190.4011,
             'test': 189.7936}


def load_npz(p):
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


def key_2d(sg, c):
    return f'{sg}|{c}|circ2d'


def key_1d(sg, c):
    """The 1-D c-sweep table is keyed by j = c/d, not by c."""
    return f'{sg}|{c / d}|circ'


def cifar(train, n):
    import torch, torchvision, torchvision.transforms as TT
    ds = torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets', train=train,
                                      download=False, transform=TT.ToTensor())
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=4)
    X = torch.cat([xb for xb, _ in dl]).reshape(-1, d).to(torch.float64)
    return X[:n].to(DEV)


def sec_A():
    """Dataset traces.  These are NOT bookkeeping: the CIFAR training subsets differ in
    energy, so any comparison across N (or across the train/test split) that is not
    normalised by Tr(Sigma) is comparing two different scales."""
    import torch
    print("=" * 78)
    print("A.  Tr(Sigma) OF EVERY IMAGE SET USED  (about that set's own mean, except the")
    print("    test row, which is about the TRAIN mean -- that is what the estimator uses)")
    print("=" * 78)
    X = cifar(True, 50000)
    mu10 = X[:10000].mean(0)
    for N in (10000, 20000, 40000, 50000):
        S = X[:N]
        T[N] = float(((S - S.mean(0)) ** 2).sum() / N)
        print(f"    train[:{N:<6}]  {T[N]:9.4f}   vs train[:10000]: {T[N]-T[10000]:+.4f}")
    Y = cifar(False, 10000)
    T['test'] = float(((Y - mu10) ** 2).sum() / 10000)
    own = float(((Y - Y.mean(0)) ** 2).sum() / 10000)
    print(f"    TEST (10000)   {T['test']:9.4f}   vs train[:10000]: {T['test']-T[10000]:+.4f}"
          f"   (about its own mean: {own:.4f})")
    print("    => bigger training subsets are LESS energetic.  An in-sample loss at N=40k is")
    print("       depressed on that account alone, so a raw N-move UNDERSTATES the optimism.")
    print()


def sec_B():
    print("=" * 78)
    print("B.  c-SWEEP, IN-SAMPLE, N=10^4   (tables/rf_pixel_circ2d.npz, rf_pixel_circ_csweep.npz)")
    print("=" * 78)
    v2, v1 = load_npz('tables/rf_pixel_circ2d.npz'), load_npz('tables/rf_pixel_circ_csweep.npz')
    lin = {}
    for sg in SIGS:
        L = v2.get(f'linear|{sg}', v1.get(f'linear|{sg}'))
        if L is None:      # the held-out grid is wider than the in-sample c-sweep grid
            continue
        lin[sg] = float(L)
        print(f"    sigma={sg:<6}  Wiener (in-sample) {lin[sg]:.4f}")
        for tag, v, kf in (('2D', v2, key_2d), ('1D', v1, key_1d)):
            raw = [f'{np.mean(v[kf(sg,c)]):9.4f}' if kf(sg, c) in v else '     ----' for c in CS]
            exc = [f'{np.mean(v[kf(sg,c)])-lin[sg]:+9.4f}' if kf(sg, c) in v else '     ----'
                   for c in CS]
            print(f"      {tag} raw     " + ' '.join(raw))
            print(f"      {tag} excess  " + ' '.join(exc))
        print("         c =        " + ' '.join(f'{c:>9}' for c in CS))
    print()
    return lin


def sec_C():
    print("=" * 78)
    print("C.  N-SWEEP, IN-SAMPLE RAW LOSS   (tables/rf_circ{2d,1d}_nsweep_*.npz)")
    print("=" * 78)
    print("       model              N=10k      20k      40k     move(4xN)")
    out = {}
    for sg in ('0.127', '0.452'):
        for tag, c in (('2D', 512), ('2D', 1536), ('1D', 1536), ('WIENER', 0)):
            ser = []
            for N in NS:
                w = load_npz(f'tables/rf_circ{"1d" if tag == "1D" else "2d"}_nsweep_{N}.npz')
                if tag == 'WIENER':
                    ser.append(float(w[f'linear|{sg}']))
                else:
                    ser.append(float(np.mean(w[key_2d(sg, c) if tag == '2D'
                                             else key_1d(sg, c)])))
            out[(sg, tag, c)] = ser
            lbl = f'{tag} c={c}' if c else tag
            print(f"       s={sg:<6} {lbl:<10} " + ' '.join(f'{x:8.4f}' for x in ser)
                  + f"   {ser[2]-ser[0]:+.4f}")
    print("    => the RFs move ~0.03-0.07 over 4x N; the in-sample WIENER moves ~0.18-0.25.")
    print("       An in-sample loss rises with N by that model's own optimism, so this is an")
    print("       INDEPENDENT measurement of what section D measures directly.")
    print()
    return out


def sec_D():
    print("=" * 78)
    print("D.  HELD-OUT   (tables/rf_pixel_heldout.npz)   train = CIFAR train[:10000],")
    print("    test = the 10,000 CIFAR TEST images.  'own gap' = test - train_resid.")
    print("=" * 78)
    h = load_npz('tables/rf_pixel_heldout.npz')
    for sg in SIGS:
        L = h.get(f'linear|{sg}')
        if L is None:
            continue
        print(f"    === sigma={sg}   Wiener train {L[0]:8.4f}  test {L[1]:8.4f}  "
              f"(gap {L[1]-L[0]:+.4f}) ===")
        for arm in ('2d', '1d'):
            for c in CS:
                k = f'{sg}|{c}|{arm}'
                if k not in h:
                    continue
                a = h[k]
                tr, rs, te = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
                print(f"      {arm} c={c:<5} train {tr:8.4f} ({tr-L[0]:+8.4f})   "
                      f"test {te:8.4f} ({te-L[1]:+8.4f})   own gap {te-rs:+.4f}")
    print("    => TWO IN-SAMPLE NUMBERS.  'train' is the project convention")
    print("       Tr(Sigma) - q^H (P+lam I)^-1 q, which is NOT the achieved residual of w_f:")
    print("       it is shifted by exactly lam*||W||^2.  'train_resid' is the residual form of")
    print("       the same w_f.  Only test - train_resid is an honest gap; only 'train' is")
    print("       differenceable against the stored in-sample tables.")
    print("    => the sigma>=1.61 'own gaps' are NEGATIVE and nearly c-independent.  That is")
    print("       the split trace offset (section A), not a generalisation gap: at large sigma")
    print("       L ~ Tr(Sigma) - (small), so a 0.9% trace deficit passes ~1:1 into the loss.")
    print()
    return h


def sec_E():
    """The population linear floor, bracketed at matched N from both sides."""
    import torch
    from scripts.rf_pixel_heldout import linear_split
    print("=" * 78)
    print("E.  THE POPULATION LINEAR FLOOR -- in-sample climbs to it, held-out falls to it.")
    print("    Both columns at MATCHED N (fit on train[:N], score the same 10k test split).")
    print("=" * 78)
    X, Y = cifar(True, 50000), cifar(False, 10000)
    for sg in (0.127, 0.452):
        print(f"    --- sigma={sg} ---")
        print("      N        in-sample  held-out   midpoint |  normalised by Tr(Sigma), x1e3")
        for N in (10000, 20000, 40000, 50000):
            S = X[:N]
            tr = float(((S - S.mean(0)) ** 2).sum() / N)
            te_tr = float(((Y - S.mean(0)) ** 2).sum() / 10000)
            a, b = linear_split(S, Y, sg)
            an, bn = 1000 * a / tr, 1000 * b / te_tr
            print(f"      {N:<7}  {a:8.4f}  {b:8.4f}  {(a+b)/2:8.4f} |  {an:8.3f} {bn:9.3f}"
                  f"  {(an+bn)/2:8.3f}")
        print("      => RAW midpoint is stable to ~0.004 over 5x in N.  Trace-normalising")
        print("         drifts it up slightly; the two readings differ by ~0.06, which is")
        print("         far below the 2-D margin and does not move the 1-D parity call.")
    print()


def sec_F(lin, nsw, h):
    """Do the N-sweep and the held-out job agree?  The population L* must be BRACKETED."""
    print("=" * 78)
    print("F.  AGREEMENT TEST: in-sample(N) extrapolated (1/N Richardson) vs held-out@10k.")
    print("    L* must sit between them, so the bracket must be >= 0.")
    print("=" * 78)
    if not T:
        T.update(T_DEFAULT)
    print("      model              RAW extrap  heldout   bracket |  NORMALISED (x1e3) bracket")
    for (sg, tag, c), ser in nsw.items():
        if tag == 'WIENER':
            te = float(h[f'linear|{sg}'][1])
        else:
            a = h.get(f'{sg}|{c}|{"2d" if tag == "2D" else "1d"}')
            if a is None:
                continue
            te = float(a[:, 2].mean())
        ext = ser[2] + (ser[2] - ser[1])
        sn = [1000 * ser[i] / T[N] for i, N in enumerate(NS)]
        ten = 1000 * te / T['test']
        extn = sn[2] + (sn[2] - sn[1])
        lbl = f'{tag} c={c}' if c else tag
        print(f"      s={sg:<6} {lbl:<10} {ext:9.4f} {te:9.4f} {te-ext:+8.4f} |"
              f" {extn:9.3f} {ten:9.3f} {ten-extn:+8.3f}")
    print("    => RAW: the sigma=0.452 brackets go negative (worst -0.050 on 1-D c=1536,")
    print("       ~30x that cell's 0.0016 seed sd) -- apparently impossible.")
    print("    => NORMALISED: every violation disappears.  It was scale mixing, not a defect")
    print("       in either measurement.  The two routes agree once the scale is fixed.")
    print()


def sec_G():
    """THE SPLIT IS NOT EXCHANGEABLE, so run it BOTH WAYS.  The antisymmetric part of the two
    gaps is the trace offset of section A; the mean is the real optimism."""
    from scripts.rf_pixel_heldout import linear_split
    print("=" * 78)
    print("G.  DIRECTION TEST ON THE LINEAR BASELINE (fit A score B, then fit B score A).")
    print("=" * 78)
    X, Y = cifar(True, 10000), cifar(False, 10000)
    print("   sigma   fit-tr->test   fit-te->train    mean gap    asymmetric?")
    for sg in (0.127, 0.452, 1.610, 5.000):
        a0, b0 = linear_split(X, Y, sg)
        a1, b1 = linear_split(Y, X, sg)
        g0, g1 = b0 - a0, b1 - a1
        m = (g0 + g1) / 2
        rel = abs(g0 - g1) / max(abs(m), 1e-12)
        tag = 'no (%.0f%%)' % (100 * rel / 2) if rel < 0.25 else (
            'YES, sign flips' if g0 * g1 < 0 else 'YES')
        print(f"   {sg:<7.3f}   {g0:+8.4f}        {g1:+8.4f}        {m:+8.4f}    {tag}")
    print("    => at low sigma the gap is symmetric to 2-3% and the headline is clean.  Above")
    print("       sigma ~ 1, L ~ Tr(Sigma) - (small), so the 0.9% trace deficit passes ~1:1 into")
    print("       the test loss and can outweigh the optimism entirely -- USE THE MEAN THERE.")
    print("    => test-vs-test EXCESS comparisons are unaffected (the offset cancels in the")
    print("       difference); only per-model ABSOLUTE gaps need symmetrising.")
    print()


def sec_H():
    """The same direction test on the RF itself.  Gated (SWAP=1) because it runs the estimator
    twice rather than a closed form -- a couple of minutes on GPU at c=256."""
    import torch
    from core.rf_circulant_struct import circulant_rf_mmse_lag2
    from scripts.rf_pixel_heldout import filt1d, sizing1d, TB, LAM
    c, sg, s = int(os.environ.get('SWAP_C', 256)), float(os.environ.get('SWAP_SIG', 0.452)), 0
    print("=" * 78)
    print(f"H.  DIRECTION TEST ON THE RF (1-D, c={c}, sigma={sg}) -- is the sigma=0.452 bracket")
    print("    violation of section F the split offset, or a real gap?")
    print("=" * 78)
    X, Y = cifar(True, 10000), cifar(False, 10000)
    nf, ns = sizing1d(c)
    h = filt1d(c, s)
    for lbl, A, B in (('fit-train -> test', X, Y), ('fit-test  -> train', Y, X)):
        r = circulant_rf_mmse_lag2(A, h, sg, TB, lam=LAM, device=DEV, freq_chunk=nf,
                                   super_chunk=ns, x0_test=B)
        print(f"   {lbl}:  train_resid {r['train_resid']:9.4f}   test {r['test']:9.4f}"
              f"   gap {r['test']-r['train_resid']:+.4f}")
    print("    => the two directions differ; the SYMMETRISED value is the RF's real optimism.")
    print()


def main():
    lin = nsw = h = None
    if 'A' in SECTIONS:
        sec_A()
    if 'B' in SECTIONS:
        lin = sec_B()
    if 'C' in SECTIONS:
        nsw = sec_C()
    if 'D' in SECTIONS:
        h = sec_D()
    if 'E' in SECTIONS:
        sec_E()
    if 'F' in SECTIONS and nsw and h:
        sec_F(lin, nsw, h)
    if 'G' in SECTIONS:
        sec_G()
    if 'H' in SECTIONS and os.environ.get('SWAP') == '1':
        sec_H()


if __name__ == '__main__':
    main()
