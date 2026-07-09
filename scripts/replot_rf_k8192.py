"""
Replot k=8192 random-feature results with clearly distinct line styles.
Loads saved .npz and writes a new figure.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('tables/rf_theory_vs_empirical_cifar10_k8192.npz')
sg = data['sigma']
K = 8192; d = 3072
me = 4   # markevery

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    f'Random-Feature Denoiser — CIFAR-10, k={K} (> d={d}), N=10 000',
    fontsize=13
)

# Panel 1: Unconditional
ax = axes[0]

ax.plot(sg, data['linear_wiener'],
        color='k',   lw=2.5, ls='-',
        label='Linear Wiener (analytic)')

ax.plot(sg, data['rf_uncond_empirical_train'],
        color='C0',  lw=2,   ls='-',  marker='o', ms=5, markevery=me,
        label='RF uncond - train x0 + fresh Z  [finite-dataset std]')

ax.plot(sg, data['rf_uncond_theory'],
        color='C3',  lw=2,   ls='-',  marker='^', ms=5, markevery=me,
        label='RF uncond - theory (Hermite n<=2)')

ax.plot(sg, data['rf_uncond_empirical_cf'],
        color='C0',  lw=1.5, ls='--', marker='s', ms=4, markevery=me,
        label='RF uncond - empirical CF (in-sample, optimistic)')

ax.plot(sg, data['rf_uncond_empirical_dir'],
        color='C9',  lw=1.2, ls=':',  marker='D', ms=4, markevery=me,
        label='RF uncond - test images + fresh Z')

ax.set_xscale('log')
ax.set_xlabel('sigma (noise level)', fontsize=11)
ax.set_ylabel('MMSE loss', fontsize=11)
ax.set_title(f'Unconditional  L_sigma   (k={K}, d={d})', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

# Panel 2: Conditional
ax = axes[1]

ax.plot(sg, data['linear_wiener_cond'],
        color='k',   lw=2.5, ls='-',
        label='Cond. Linear Wiener (class eigenvalues)')

ax.plot(sg, data['rf_cond_empirical_train'],
        color='C2',  lw=2,   ls='-',  marker='o', ms=5, markevery=me,
        label='RF cond - train x0 + fresh Z  [finite-dataset std]')

ax.plot(sg, data['rf_cond_theory'],
        color='C1',  lw=2,   ls='-',  marker='^', ms=5, markevery=me,
        label='RF cond - theory (Hermite n<=2)')

ax.plot(sg, data['rf_cond_empirical_cf'],
        color='C2',  lw=1.5, ls='--', marker='s', ms=4, markevery=me,
        label='RF cond - empirical CF (in-sample, optimistic)')

ax.plot(sg, data['rf_cond_empirical_dir'],
        color='C8',  lw=1.2, ls=':',  marker='D', ms=4, markevery=me,
        label='RF cond - test images + fresh Z')

ax.set_xscale('log')
ax.set_xlabel('sigma (noise level)', fontsize=11)
ax.set_ylabel('MMSE loss', fontsize=11)
ax.set_title(f'Conditional  L_{{sigma,U}}   (k={K}, d={d})', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'figures/rf_theory_vs_empirical_cifar10_k8192_v2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
