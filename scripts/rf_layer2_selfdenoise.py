"""Denoise the layer2 conv map itself: circulant vs dense RF vs linear.

x0 = ResNet18 layer2 map of a CLEAN CIFAR image, channel-averaged to one 28x28 grid
     (d = 784), standardized so Tr(Sigma)/d = 1.
y  = x0 + sigma z.

Because target and input now live on the SAME index set (Z_784), a block-circulant W is
well defined and michimin's per-frequency closed form applies exactly -- unlike the
pixel-target version, where W: R^k -> R^3072 had no cyclic group to be equivariant under.

Curves: L^dense (free W), L^circ full width, L^circ banded t=8, and the linear (Wiener)
denoiser sum_i sigma^2 lam_i / (lam_i + sigma^2).

NOTE: the channel-averaged map is a poor basis for reconstructing PIXELS (an out-of-sample
ridge readout to pixels leaves residual 219 against Tr(Sigma_pix)=192, i.e. worse than the
mean). That is irrelevant here -- the map is the target, not an intermediate -- but it is
why this experiment cannot be compared to the EDM pixel curve.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scripts.rf_circulant_dnn_layers import extract, standardize, losses_at, DEV, DT

C_LIST=[int(x) for x in os.environ.get('C_LIST','1,2,4,8').split(',')]
SIGS=[float(x) for x in os.environ.get('SIGS','0.25,0.5,1.0,2.0').split(',')]
NSEED=int(os.environ.get('NSEED','4')); TB=8

X=standardize(extract('CIFAR-10',n_img=int(os.environ.get('NIMG','10000')))['layer2'].to(DEV,DT))
N,d=X.shape
Xc=X-X.mean(0); ev=torch.linalg.eigvalsh((Xc.T@Xc)/N)
print(f"layer2 channel-mean map: N={N}, d={d} (28x28), Tr(Sigma)/d={float(ev.sum())/d:.4f}\n")
res={}
for sg in SIGS:
    lin=float((sg**2*ev/(ev+sg**2)).sum())
    res[(sg,'lin')]=lin
    print(f"=== sigma={sg}  linear(Wiener) L={lin:.3f} ===")
    for c in C_LIST:
        A=np.array([losses_at(X,sg,c,TB,seed=137*s) for s in range(NSEED)])
        m,sd=A.mean(0),A.std(0,ddof=1)
        for i,q in enumerate(('dense','circ','band')):
            res[(sg,c,q)]=(m[i],sd[i])
        print(f"  k/d={c:>2} k={c*d:>6} dense={m[0]:7.3f}+-{sd[0]:.3f} "
              f"circ={m[1]:7.3f}+-{sd[1]:.3f} band={m[2]:7.3f}+-{sd[2]:.3f} "
              f"| band-dense={m[2]-m[0]:+7.3f} | vs linear {m[2]-lin:+7.3f}",flush=True)
np.savez('tables/rf_layer2_selfdenoise.npz',
         **{f"{k[0]}|{k[1]}|{k[2] if len(k)>2 else ''}":np.array(v) for k,v in res.items()})

fig,axes=plt.subplots(1,len(SIGS),figsize=(5.0*len(SIGS),4.4))
for i,sg in enumerate(SIGS):
    ax=axes[i]
    for q,col,mk,nm in (('dense','#1f4e9c','o',r'\rm dense'),('circ','#7b3fa0','s',r'\rm circ\ (full)'),
                        ('band','#c0392b','^',rf'\rm circ\ (t={TB})')):
        mu=[res[(sg,c,q)][0] for c in C_LIST]; sd=[res[(sg,c,q)][1] for c in C_LIST]
        ax.errorbar(C_LIST,mu,yerr=sd,color=col,marker=mk,lw=2.2,ms=5,capsize=3,
                    label=rf'$\mathcal{{L}}^{{{nm}}}$')
    ax.axhline(res[(sg,'lin')],color='darkorange',ls='--',lw=2.2,label='linear (Wiener)')
    ax.set_xscale('log',base=2); ax.set_xticks(C_LIST); ax.set_xticklabels([str(c) for c in C_LIST])
    ax.set_title(rf'$\sigma$={sg}'); ax.set_xlabel('$k/d$'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel(r'$\mathcal{L}_\sigma$'); ax.legend(fontsize=9)
fig.suptitle('Denoising the ResNet18 layer2 conv map itself ($d=784$, 28$\\times$28, genuine 2-D locality)\n'
             'target and input share $Z_{784}$, so a circulant $W$ is well defined and the closed form applies exactly',
             fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig('figures/rf_layer2_selfdenoise.png',dpi=150,bbox_inches='tight')
print("Saved figures/rf_layer2_selfdenoise.png")
