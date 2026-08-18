"""Plot: circulant with m x the ROWS of dense but 512/m x FEWER free parameters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rB=np.load('tables/rf_circulant_rowmult_A_t8.npz'); rF=np.load('tables/rf_circulant_rowmult_A.npz'); r=rB; z=np.load('tables/rf_circulant_avgpool_scaleup.npz')
zs={round(float(s),3):s for s in z['sigmas']}
JS=[1,2,4,8,16]; MS=[2,4,8]; d=512
SIG=[s for s in [0.010,0.036,0.127,0.452,2.212,7.880]]
plt.rcParams.update({'font.size':11,'axes.labelsize':12.5,'axes.titlesize':12})
fig,axes=plt.subplots(2,3,figsize=(16.5,8.8))
for i,s in enumerate(SIG):
    ax=axes[i//3][i%3]; zk=zs[round(s,3)]; cs=list(z[f'{zk}|c']); de=z[f'{zk}|dense']
    js=[j for j in JS if j in cs]
    ax.plot(js,[de[cs.index(j)] for j in js],color='#1f4e9c',marker='o',lw=2.4,ms=6,
            label=r'$\mathcal{L}^{\rm dense}$  ($k{=}jd$ rows, $2jd^2$ params)')
    for m,col,mk in zip(MS,('#e07b39','#c0392b','#7b3fa0'),('^','s','D')):
        xs=[j for j in js if f'{j}|{m}|{s}' in rB.files]
        ys=[float(rB[f'{j}|{m}|{s}'][0]) for j in xs]
        es=[float(rB[f'{j}|{m}|{s}'][1]) for j in xs]
        ax.errorbar(xs,ys,yerr=es,color=col,marker=mk,lw=2.4,ms=6,capsize=3,
                    label=rf'BANDED $t=8$, $m={m}$ ($c={m}j$, {m}$\times$rows, '
                          rf'{d//m}$\times$fewer params)')
        xf=[j for j in js if f'{j}|{m}|{s}' in rF.files]
        yf=[float(rF[f'{j}|{m}|{s}'][0]) for j in xf]
        ax.plot(xf,yf,color=col,lw=1.2,ls=':',alpha=.75,
                label=(rf'full width, $m={m}$ (reference)' if m==MS[0] else None))
    ax.set_xscale('log',base=2); ax.set_xticks(js); ax.set_xticklabels([str(j) for j in js])
    ax.set_title(rf'$\sigma$={s:.3f}'); ax.grid(True,alpha=.3); ax.set_xlabel('$k/d$')
    if i%3==0: ax.set_ylabel(r'denoiser loss $\mathcal{L}_\sigma$')
    if i==0: ax.legend(fontsize=7.4)
fig.suptitle('CIFAR-10, ResNet18 avgpool $\\phi$ ($d=512$): circulant with MORE ROWS but FEWER '
             'FREE PARAMETERS than dense\n'
             r'at $k/d=j$ the circulant uses $c=mj$ blocks $\Rightarrow$ $m\times$ the rows and '
             r'exactly $d/m=512/m\times$ fewer parameters, at every $j$.  '
             'Error bars = spread over 2 $\\Theta$ draws.',fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig('figures/rf_circulant_rowmult_banded.png',dpi=150,bbox_inches='tight')
print("Saved figures/rf_circulant_rowmult_banded.png")
