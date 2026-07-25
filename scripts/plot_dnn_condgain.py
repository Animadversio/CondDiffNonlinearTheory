"""Redraw the CIFAR-10 MMSE 3-panel figure with a customized RIGHT panel:
per-method conditioning gain = (uncond MMSE) − (cond MMSE), positive = conditioning helps.
Methods: Linear Wiener, Oracle Bayes (self-inclusive pool), EDM UNet.
Reads the precomputed table; does not recompute anything."""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ  = 'tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz'
FIG  = 'figures/dnn_feature_mmse_cifar10_condgain.png'
d = np.load(NPZ, allow_pickle=True)
sigma = d['sigma']
def g(k): return np.array(d[k])

lin_u, lin_c = g('linear_uncond'), g('linear_cond')
bay_u, bay_c = g('bayes_uncond'),  g('bayes_cond')
edm_u, edm_c = g('edm_uncond'),    g('edm_cond')

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('CIFAR-10 MMSE Denoiser Loss  (N=10000, σ in pixel [0,1] units)', fontsize=12)

# --- Panel 1: MMSE loss vs sigma (same as base figure) ---
ax = axes[0]
ax.plot(sigma, bay_c,               'g-',   lw=2.5, label='Oracle Bayes cond (LB, same-class pool)')
ax.plot(sigma, bay_u,               'b-',   lw=2.5, label='Oracle Bayes uncond (LB, all-class pool)')
ax.plot(sigma, g('wiener_class_cond'),'g--',lw=2,   label='Class-cond Wiener (analytic)')
ax.plot(sigma, lin_c,               'k--',  lw=2,   label='Linear+U Wiener (analytic)')
ax.plot(sigma, lin_u,               'k-',   lw=2,   label='Linear Wiener uncond (analytic)')
ax.plot(sigma, edm_u,               'r-o',  lw=2, ms=4, label='EDM UNet uncond (VE)')
ax.plot(sigma, edm_c,               'r-s',  lw=2, ms=4, label='EDM UNet cond (VE)')
ax.plot(sigma, g('resnet_uncond'),  'C0-o', lw=2, ms=4, label='ResNet18 uncond')
ax.set_xscale('log'); ax.set_xlabel('sigma (pixel units)'); ax.set_ylabel('MSE loss')
ax.set_title('MMSE Denoiser Loss vs sigma')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# --- Panel 2: DNN gain over linear (same as base figure) ---
ax = axes[1]
ax.plot(sigma, lin_u - edm_u, 'r-o', lw=2, ms=4, label='EDM uncond gain over linear Wiener')
ax.plot(sigma, lin_c - edm_c, 'r-s', lw=2, ms=4, label='EDM cond gain over Wiener+U')
if 'linear_plus_dnn' in d:
    ax.plot(sigma, lin_u - g('linear_plus_dnn'), 'm-^', lw=2, ms=4,
            label='[y;ResNet] combined gain over Wiener')
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xscale('log'); ax.set_xlabel('sigma'); ax.set_ylabel('L_linear − L_DNN')
ax.set_title('DNN gain over linear (positive = DNN better)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# --- Panel 3 (CUSTOM): per-method conditioning gain = uncond − cond ---
ax = axes[2]
ax.plot(sigma, lin_u - lin_c, 'k-s', lw=2, ms=4, label='Linear Wiener:  uncond − cond')
ax.plot(sigma, bay_u - bay_c, 'b-o', lw=2.5, ms=4, label='Oracle Bayes:  uncond − cond')
ax.plot(sigma, edm_u - edm_c, 'r-^', lw=2, ms=5, label='EDM UNet:  uncond − cond')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xscale('log'); ax.set_xlabel('sigma (pixel units)')
ax.set_ylabel('conditioning gain:  L_uncond − L_cond')
ax.set_title('Conditioning gain (uncond − cond MMSE; positive = conditioning helps)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches='tight')
print(f"Saved {FIG}")
print(f"conditioning-gain high-σ:  Linear={lin_u[-1]-lin_c[-1]:.2f}  "
      f"Bayes={bay_u[-1]-bay_c[-1]:.2f}  EDM={edm_u[-1]-edm_c[-1]:.2f}")
print(f"Bayes peak gain = {(bay_u-bay_c).max():.2f} at sigma={float(sigma[(bay_u-bay_c).argmax()]):.2f}")
