"""
mi_from_immse.py — estimate I(X;U) on CIFAR-10 / MNIST from denoising losses ALONE,
via the I-MMSE identity, using the oracle Bayes curves already computed by
scripts/dnn_feature_mmse.py (reads tables/, recomputes nothing).

Identity
--------
For Y_t = X + sqrt(t) Z (t = sigma^2 = noise variance), I-MMSE (Guo-Shamai-Verdu) plus the
chain rule with the Markov chain U -> X -> Y gives

    I(X;U) = (1/2) INT_0^inf [ mmse(X|Y_t) - mmse(X|Y_t,U) ] dt / t^2          (nats)

(the U-conditioned and unconditioned I-MMSE identities differ by I(U;Y_t), which tends to
I(U;X) as t -> 0).  In sigma:  dt = 2 sigma d(sigma), so equivalently

    I(X;U) = INT_0^inf Delta(sigma) sigma^-3 d(sigma),   Delta = mmse_uncond - mmse_cond.

So the shaded area between the unconditional and class-conditional ORACLE BAYES denoising
curves, weighted by sigma^-3, IS the mutual information — no density estimation, no
variational bound, no critic network. This is the point: MI between a 3072-dim image and
its label is not directly estimable, but the denoising losses are.

What we integrate
-----------------
  bayes_uncond / bayes_cond   : oracle (Nadaraya-Watson) Bayes denoiser of x0, pool = all
                                samples vs pool = same-class samples. These ARE E[x0|y] and
                                E[x0|y,U] for the empirical distribution.
  linear_uncond / wiener_class_cond : the same gap for the best LINEAR denoiser.

Result (see the printout): the Bayes integral lands just under H(U) = log 10, as it must,
while the LINEAR integral overshoots it by orders of magnitude. That is not a bug — see the
discussion in the docstring of report() below.

    python scripts/mi_from_immse.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG2 = np.log(2.0)

DATASETS = [
    ('CIFAR-10', 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz', 10),
    ('MNIST',    'tables/dnn_feature_mmse_mnist_N10000_noise5_sigma30.npz',   10),
]


def integrate_immse(sigma, gap):
    """I = (1/2) INT gap(t) dt/t^2, t=sigma^2  ==  INT gap(sigma) sigma^-3 d(sigma).
    Trapezoid on the log-sigma grid (dsigma = sigma dln sigma => integrand gap*sigma^-2),
    plus the analytic tail beyond sigma_max assuming gap saturates (it does):
        INT_smax^inf g ds/s^3 = g / (2 smax^2).
    Returns (total, grid_part, tail, integrand)."""
    o = np.argsort(sigma)
    s, g = np.asarray(sigma)[o], np.asarray(gap)[o]
    integrand = g * s ** -2
    grid = float(np.trapezoid(integrand, np.log(s)))
    tail = float(g[-1] / (2.0 * s[-1] ** 2))
    return grid + tail, grid, tail, integrand


def report():
    """
    Interpreting the two integrals.

    ORACLE BAYES  -> a genuine estimate of I(X;U). It must satisfy I <= H(U) = log 10, and
    it does (CIFAR-10: 2.266 nats = 3.270 bits vs the 3.322-bit ceiling, i.e. 98.4% of it).
    That the estimate lands just below a bound it was never told about is a real check on
    the whole pipeline. It is close to the ceiling because the oracle uses a self-inclusive
    sample pool, so at small sigma it identifies the exact image and hence its label —
    the empirical distribution really does carry ~all of H(U).

    LINEAR -> NOT an MI estimate. It overshoots H(U) by ~370x on CIFAR-10, which is
    impossible for a mutual information. Mechanism: at small sigma the Bayes denoiser
    recovers x0 essentially perfectly whether or not it is told the class, so the Bayes gap
    Delta -> 0 and contributes nothing. The LINEAR denoiser cannot recover x0 at ANY sigma,
    so its gap stays strictly positive as sigma -> 0 — and the sigma^-3 weight there is
    enormous, so the integral blows up.

    Hence the natural-sounding claim "feeding the denoiser the conditioning variable can
    only help by at most the mutual information" is TRUE ONLY for the Bayes-optimal
    denoiser, where it holds with equality and that equality IS the I-MMSE identity. For a
    restricted denoiser class it is FALSE: conditioning buys two separate things,
      (i) the genuine information I(X;U), and
      (ii) a reduction of the class's own APPROXIMATION error, because the conditional
           problem is easier (per-class data is more nearly unimodal/Gaussian, so a linear
           filter fits it far better than it fits the full mixture).
    Only (i) is information. (ii) is model mismatch, and it is unbounded by H(U).
    """
    rows = []
    for name, path, n_cls in DATASETS:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found"); continue
        d = np.load(path, allow_pickle=True)
        sig = d['sigma']
        H_U = np.log(n_cls)
        gb = d['bayes_uncond'] - d['bayes_cond']
        gl = d['linear_uncond'] - d['wiener_class_cond']
        Ib, gridb, tailb, intb = integrate_immse(sig, gb)
        Il, gridl, taill, intl = integrate_immse(sig, gl)
        rows.append(dict(name=name, sigma=sig, gb=gb, gl=gl, Ib=Ib, Il=Il, H_U=H_U,
                         intb=intb, intl=intl, d=d))
        print(f"\n===== {name} =====  H(U) = log {n_cls} = {H_U:.4f} nats = {H_U/LOG2:.4f} bits")
        print(f"  I(X;U) via I-MMSE + ORACLE BAYES = {Ib:.4f} nats = {Ib/LOG2:.4f} bits"
              f"   ({100*Ib/H_U:.1f}% of H(U))   <= H(U)? {'YES' if Ib <= H_U else 'NO'}")
        print(f"      (grid {gridb:.4f} + analytic tail {tailb:.2e})")
        print(f"  same integral with LINEAR denoisers = {Il:.4f} nats = {Il/LOG2:.4f} bits"
              f"   <= H(U)? {'YES' if Il <= H_U else 'NO — exceeds by %.0fx' % (Il/H_U)}")
        print(f"  ratio linear / bayes = {Il/Ib:.1f}x")
    return rows


def main():
    rows = report()
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(7.0 * n, 9.0))
    axes = np.asarray(axes).reshape(2, n)
    fig.suptitle('Mutual information from denoising alone — I-MMSE identity\n'
                 r'$I(X;U)=\frac{1}{2}\int_0^\infty[\mathrm{mmse}(X|Y_t)-\mathrm{mmse}(X|Y_t,U)]\,dt/t^2$,'
                 '   $t=\\sigma^2$   (area under the red curve = MI)', fontsize=12)
    for j, r in enumerate(rows):
        s, d = r['sigma'], r['d']
        ax = axes[0, j]
        ax.loglog(s, d['bayes_uncond'], color='crimson', lw=2, label='oracle Bayes, uncond.')
        ax.loglog(s, d['bayes_cond'], color='crimson', lw=2, ls='--', label='oracle Bayes, cond. on U')
        ax.loglog(s, d['linear_uncond'], color='darkorange', lw=1.6, label='linear, uncond.')
        ax.loglog(s, d['wiener_class_cond'], color='darkorange', lw=1.6, ls='--', label='linear, cond. on U')
        ax.fill_between(s, d['bayes_cond'], d['bayes_uncond'], color='crimson', alpha=.15)
        ax.set_xlabel('$\\sigma$'); ax.set_ylabel('MSE'); ax.grid(True, alpha=.3)
        ax.set_title(f"{r['name']}: denoising losses"); ax.legend(fontsize=8)

        ax2 = axes[1, j]
        ax2.semilogx(s, r['intb'], color='crimson', lw=2.2, marker='o', ms=3,
                     label='oracle Bayes  → $I(X;U)$')
        ax2.semilogx(s, r['intl'], color='darkorange', lw=1.8, ls='--', marker='s', ms=3,
                     label='linear (NOT an MI)')
        ax2.set_yscale('symlog', linthresh=1e-2)
        ax2.axhline(0, color='k', lw=.6)
        ax2.set_xlabel('$\\sigma$')
        ax2.set_ylabel(r'integrand  $\Delta(\sigma)\,\sigma^{-3}$  (per $d\ln\sigma$)')
        ax2.set_title(f"{r['name']}: I-MMSE integrand"); ax2.grid(True, alpha=.3)
        txt = (f"I  Bayes  = {r['Ib']:.3f} nats = {r['Ib']/LOG2:.3f} bits\n"
               f"I  linear = {r['Il']:.1f} nats  ({r['Il']/r['H_U']:.0f}x H(U)!)\n"
               f"H(U)      = {r['H_U']:.3f} nats = {r['H_U']/LOG2:.3f} bits")
        ax2.text(.03, .97, txt, transform=ax2.transAxes, va='top', fontsize=8.5,
                 family='monospace', bbox=dict(fc='white', alpha=.85, ec='gray'))
        ax2.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/mi_from_immse.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")
    np.savez('tables/mi_from_immse.npz',
             **{f"{r['name']}_{k}": r[k] for r in rows for k in ('sigma', 'gb', 'gl', 'Ib', 'Il', 'H_U')})


if __name__ == '__main__':
    main()
