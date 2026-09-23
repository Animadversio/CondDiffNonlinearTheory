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
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

d = 3072
CS = (32, 96, 256, 512, 1536, 3072, 6144)
SIGS = ('0.127', '0.452', '0.621', '0.853', '1.172', '1.61', '2.212', '5.0')
NS = (10000, 20000, 40000)
SECTIONS = os.environ.get('SECTIONS', 'ABCDEFGIJKL')  # H (the RF swap test) also needs SWAP=1
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


# ---------------------------------------------------------------------------
# the two arms added 2026-09-23 on michimin's request: dense, and the EDM U-Net
# ---------------------------------------------------------------------------
DJS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def sec_I():
    """Dense RF held out (tables/rf_pixel_dense_heldout.npz).

    The dense arm is the one that had a standing reason to fail: the 2026-09-21 N-sweep
    caught its advantage over linear collapsing 0.825 -> 0.336 -> 0.039 as N went
    10k -> 20k -> 40k at k/d = 4.  Its `train` column is asserted by the driver to reproduce
    tables/rf_pixel_dense_sweep.npz, so it is also the in-sample source used in section J.
    """
    print("=" * 78)
    print("I.  DENSE RF, HELD OUT  (tables/rf_pixel_dense_heldout.npz)  same 10k/10k split")
    print("    as section D.  k = j*d trained parameters per output coordinate.")
    print("=" * 78)
    v = load_npz('tables/rf_pixel_dense_heldout.npz')
    if not v:
        print("    (table missing)\n")
        return v
    print("    I.1  EXCESS OVER THE HELD-OUT WIENER (test vs test -- trace offset cancels)")
    print(f"    {'sigma':>7} {'Wiener_te':>10} " + " ".join(f"k/d={j:<6g}" for j in DJS))
    for sg in SIGS:
        L = v.get(f'linear|{sg}')
        if L is None:
            continue
        row = []
        for j in DJS:
            a = v.get(f'{sg}|{j}|dense')
            row.append(f"{a[:, 2].mean()-L[1]:+9.4f}" if a is not None else "    --   ")
        print(f"    {sg:>7} {L[1]:10.4f} " + " ".join(row))
    print()
    print("    I.2  IN-SAMPLE vs HELD-OUT, excess over the MATCHING Wiener column")
    print(f"    {'sigma':>7} {'k/d':>5} {'train':>9} {'(vs W)':>9} {'test':>9} {'(vs W)':>9}"
          f" {'own gap':>9}")
    for sg in SIGS:
        L = v.get(f'linear|{sg}')
        if L is None:
            continue
        for j in DJS:
            a = v.get(f'{sg}|{j}|dense')
            if a is None:
                continue
            tr, rs, te = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
            print(f"    {sg:>7} {j:5g} {tr:9.4f} {tr-L[0]:+9.4f} {te:9.4f} {te-L[1]:+9.4f}"
                  f" {te-rs:+9.4f}")
    print()
    print("    => THE SIGN FLIPS AT LOW SIGMA.  In sample dense beats Wiener from k/d ~ 3 at")
    print("       sigma=0.127 and the margin grows to -1.80 at k/d=8.  Held out it never")
    print("       beats it at ANY sigma or width, and past k/d=4 the held-out curve turns")
    print("       back UPWARD -- the in-sample curve is monotone decreasing, so that")
    print("       turnaround is overfitting and nothing else.")
    print("    => own gap grows monotonically in k/d at every sigma, reaching +4.47 at")
    print("       sigma=0.127, k/d=8.  Wiener's is +0.459, the 2-D circulant's +0.097.")
    print("    => at sigma >= 1.61 the dense arm is nearly honest (own gap +0.08..+0.37) and")
    print("       in-sample and held-out excesses agree to ~0.1.  The retraction is LOW-SIGMA")
    print("       ONLY -- the same shape as the circulant retraction in section D.")
    print("    => the sigma=5.0 own gaps are NEGATIVE for every width, as Wiener's is")
    print("       (-0.334).  That is the section-A trace offset, not a generalisation gap.")
    print()
    return v


def _cross(js, vals, target):
    """First k/d at which the dense curve reaches `target`, log-interpolated in k/d.

    Returns None if it never gets there -- which is the interesting outcome at low sigma,
    where the held-out dense curve is U-shaped and its MINIMUM sits above the circulant.
    """
    for i in range(len(js) - 1):
        a, b = vals[i], vals[i + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            f = (a - target) / (a - b)
            return float(2 ** (np.log2(js[i]) + f * (np.log2(js[i + 1]) - np.log2(js[i]))))
    return None


def sec_J(v=None, h=None):
    """Does dense still overtake the circulant once both sides are held out?

    The published crossing (k/d ~ 1.6-2.7, 2026-09-16/17) is measured entirely in sample, on
    the side of the comparison that the N-sweep later showed was memorising.  Both columns
    below come from the SAME table (rf_pixel_dense_heldout.npz), so in-sample and held-out
    crossings are computed by identical code on identical images and differ only in which
    column is read.  The circulant target is its c=1536 value, the largest c held out for
    both arms; at low sigma the 2-D arm is still falling in c there, so the 2-D crossing is
    if anything understated.
    """
    v = v if v is not None else load_npz('tables/rf_pixel_dense_heldout.npz')
    h = h if h is not None else load_npz('tables/rf_pixel_heldout.npz')
    print("=" * 78)
    print("J.  THE DENSE-vs-CIRCULANT CROSSING, IN SAMPLE AND HELD OUT")
    print("    k/d at which dense first reaches the circulant's c=1536 loss.  'never' means")
    print("    the dense curve does not reach it at any width measured (k/d <= 8).")
    print("=" * 78)
    print(f"    {'sigma':>7} | {'vs 1-D circ':>22} | {'vs 2-D circ':>22}")
    print(f"    {'':>7} | {'in-samp':>10} {'held-out':>11} | {'in-samp':>10} {'held-out':>11}")
    for sg in SIGS:
        if v.get(f'linear|{sg}') is None:
            continue
        js, tr_v, te_v = [], [], []
        for j in DJS:
            a = v.get(f'{sg}|{j}|dense')
            if a is not None:
                js.append(j); tr_v.append(a[:, 0].mean()); te_v.append(a[:, 2].mean())
        out = []
        for arm in ('1d', '2d'):
            c = h.get(f'{sg}|1536|{arm}')
            if c is None:
                out += ['    --   ', '     --    ']
                continue
            for col, series in ((0, tr_v), (2, te_v)):
                x = _cross(js, series, c[:, col].mean())
                out.append(f"{x:10.2f}" if x else "     never")
        print(f"    {sg:>7} | {out[0]:>10} {out[1]:>11} | {out[2]:>10} {out[3]:>11}")
    print()
    print("    => AT HIGH SIGMA THE CROSSING SURVIVES ALMOST UNMOVED: nothing on either side")
    print("       is overfitting there, so holding out changes both columns together.")
    print("    => AT sigma = 0.127 IT CEASES TO EXIST.  Held out, dense's best width (k/d=4,")
    print("       37.7M trained parameters) scores 9.93 while the 2-D circulant at c=3072")
    print("       (9.4M parameters, a QUARTER as many) scores 6.75 and the 1-D arm 8.12.")
    print("       Dense is beaten by both arms and by the Wiener baseline at once.")
    print("    => so 'dense overtakes circulant at matched free parameters' is a HIGH-SIGMA")
    print("       statement.  At low sigma it was an artefact of scoring the arm with 75M")
    print("       free parameters on the 10,000 images it was fitted on.")
    print()


def sec_K():
    """The EDM U-Net on the same test images.

    NOT a matched-10k comparison: EDM was trained on all 50,000 CIFAR train images, so this
    is the analogue of the 50k-Wiener row in section 4.3 of the doc -- same evaluation set,
    five times the training data.  Its train column evaluates the network on 10,000 images
    that ARE in its training set, so train-vs-test measures how much a real diffusion model
    memorises, on the same footing as Wiener's +0.459.
    """
    print("=" * 78)
    print("K.  EDM U-NET, SAME 10,000 CIFAR TEST IMAGES  (tables/edm_pixel_heldout.npz)")
    print("    +- is the MC standard error over full sweeps of all 10,000 images; the noise")
    print("    z is sampled here, not integrated analytically as in every RF arm above.")
    print("    !! EDM SAW ALL 50,000 CIFAR TRAIN IMAGES.  Handicap in ITS favour, 5:1.")
    print("=" * 78)
    e = load_npz('tables/edm_pixel_heldout.npz')
    if not e:
        print("    (table missing)\n")
        return
    tags = sorted({k.split('|')[0] for k in e if not k.startswith('linear')})
    for tag in tags:
        print(f"    === {tag} ===")
        print(f"    {'sigma':>7} {'train(seen)':>12} {'test(unseen)':>13} {'memo gap':>10}"
              f" {'Wiener_te':>10} {'vs Wiener':>10}")
        for sg in SIGS:
            a = e.get(f'{tag}|{sg}')
            L = e.get(f'linear|{sg}')
            if a is None or L is None:
                continue
            print(f"    {sg:>7} {a[0]:12.4f} {a[2]:13.4f} {a[2]-a[0]:+10.4f}"
                  f" {L[1]:10.4f} {a[2]-L[1]:+10.4f}")
        print()
    print("    => EDM BEATS THE HELD-OUT WIENER AT EVERY SIGMA, BY FAR MORE THAN ANY RF ARM.")
    print("       This is the first genuinely nonlinear denoiser in the comparison that is")
    print("       not a nearest-neighbour memoriser (the 'oracle Bayes' curve in")
    print("       figures/dnn_feature_mmse_*.png was retracted on 2026-08-10 for exactly")
    print("       that: posterior N_eff = 1.00 for every sigma <= 1.61 at N = 10^4).")
    print("    => ITS OWN MEMORISATION GAP IS LARGER THAN WIENER'S over much of the grid.")
    print("       A trained diffusion model memorises its training set more than a 4.7M-")
    print("       parameter Gaussian fit does -- measured, not assumed.")
    print("    => the gap is a LOWER bound on EDM's memorisation: the train column is 10,000")
    print("       of the 50,000 images it saw, so each was seen with 1/5 the weight that a")
    print("       10k-trained model would have given it.")
    print()


def sec_L():
    """Every class at its best width, against the 50k linear denoiser (michimin, 06:11).

    This is the table behind figures/rf_heldout_vs_sigma.png.  The baseline here is linear
    fitted on ALL 50,000 train images -- deliberately handicapped against every RF arm, which
    saw 10,000 -- because that is the comparison that needs no floor, no extrapolation and no
    trace normalisation: same evaluation set, and the RF is the one giving away data.

    The envelope over widths is a min taken ON THE TEST SET, so it carries a mild selection
    bias.  It matters only near the dense minimum at low sigma, where the curve is flat; the
    circulant arms are still monotone in c over the widths measured.
    """
    print("=" * 78)
    print("L.  EVERY CLASS AT ITS BEST MEASURED WIDTH, vs LINEAR FITTED ON ALL 50,000")
    print("    (the table behind figures/rf_heldout_vs_sigma.png)")
    print("    All losses on the SAME 10,000 CIFAR test images.  RF arms saw 10,000 train")
    print("    images; linear(50k) and EDM saw all 50,000 -- a 5:1 handicap AGAINST the RFs.")
    print("=" * 78)
    h = load_npz('tables/rf_pixel_heldout.npz')
    dn = load_npz('tables/rf_pixel_dense_heldout.npz')
    ed = load_npz('tables/edm_pixel_heldout.npz')
    l5 = load_npz('tables/rf_linear50k_heldout.npz')
    if not l5:
        print("    (tables/rf_linear50k_heldout.npz missing -- run scripts/rf_heldout_plot.py"
              " with REBUILD_LIN50=1)\n")
        return

    def env(tab, sg, suf, ws):
        best = None
        for w in ws:
            a = tab.get(f'{sg}|{w}|{suf}')
            if a is not None:
                v = float(a[:, 2].mean())
                best = v if best is None or v < best else best
        return best

    rows = []
    print(f"    {'sigma':>7} {'lin50':>9} | {'dense':>8} {'1-D':>8} {'2-D':>8} {'lin10':>8}"
          f" {'EDM':>8}   (excess over lin50)")
    for sg in SIGS:
        if f'lin50|{sg}' not in l5:
            continue
        L = float(l5[f'lin50|{sg}'][1])
        dv = env(dn, sg, 'dense', DJS)
        a1, a2 = env(h, sg, '1d', CS), env(h, sg, '2d', CS)
        l1 = h.get(f'linear|{sg}')
        l1 = float(l1[1]) if l1 is not None else None
        e = ed.get(f'uncond-ve|{sg}')
        e = float(e[2]) if e is not None else None
        f = lambda v: f'{v - L:+8.4f}' if v is not None else '     ---'
        print(f"    {sg:>7} {L:9.4f} | {f(dv)} {f(a1)} {f(a2)} {f(l1)} {f(e)}")
        rows.append((float(sg), L, dv, a1, a2, e))
    print()

    print("    fraction of the linear(50k) -> EDM gap that each class closes:")
    print(f"    {'sigma':>7} {'gap':>8} | {'dense':>8} {'1-D':>8} {'2-D':>8}")
    for sg, L, dv, a1, a2, e in rows:
        if e is None:
            continue
        g = L - e
        p = lambda v: f'{100 * (L - v) / g:7.1f}%' if v is not None else '     ---'
        print(f"    {sg:>7} {g:8.3f} | {p(dv)} {p(a1)} {p(a2)}")
    print()

    xs = [(s, (a2 - L)) for s, L, _, _, a2, _ in rows if a2 is not None]
    for i in range(len(xs) - 1):
        (s0, e0), (s1, e1) = xs[i], xs[i + 1]
        if e0 * e1 <= 0 and e0 != e1:
            fr = e0 / (e0 - e1)
            sc = math.exp(math.log(s0) + fr * (math.log(s1) - math.log(s0)))
            print(f"    => THE 2-D ARM CROSSES linear(50k) AT sigma ~ {sc:.3f}.")
    print("       That is inside the octave between sigma=0.452 and 0.621 that nothing had")
    print("       been measured in, and it is robust to the c=3072 cells still missing there:")
    print("       at sigma=0.452 the c=1536->3072 step is worth -0.49, and applying that whole")
    print("       step to the 0.621 point still only moves the crossing to ~0.50.")
    print("    => THE DENSE CURVE IS FLAT AND NEVER CROSSES: +1.76 at sigma=0.127, a shallow")
    print("       minimum of +0.98 near sigma=0.85, +1.34 at sigma=5.0.  Over a 40x range in")
    print("       sigma, held-out dense is a near-constant ~1 above the linear denoiser.")
    print("    => !! WE DO BEST WHERE THE PRIZE IS SMALLEST.  The linear->EDM gap peaks at")
    print("       sigma ~ 0.62-0.85 (9.73) and is only 3.58 at sigma=0.127.  The 2-D arm")
    print("       captures 40% of the gap at sigma=0.127, 5.5% at 0.452, and is net-negative")
    print("       from 0.621 on -- so the one genuine win is in the regime the gap curve says")
    print("       is least interesting, and no class touches the peak.")
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
    dv = sec_I() if 'I' in SECTIONS else None
    if 'J' in SECTIONS:
        sec_J(dv, h)
    if 'K' in SECTIONS:
        sec_K()
    if 'L' in SECTIONS:
        sec_L()


if __name__ == '__main__':
    main()
