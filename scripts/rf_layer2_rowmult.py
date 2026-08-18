"""layer2 self-denoising: circulant with m x the rows of dense (c = m*j blocks), banded t=8.

x0 = ResNet18 layer2 map, channel-mean 28x28 (d=784), standardized; y = x0 + sigma z.
At dense width k = j*d the circulant uses c = m*j blocks, so m x the rows and exactly
d/m = 784/m x fewer free parameters. Circulant W is well defined here (target and input
share Z_784), so the closed form applies and core.rf_circulant_struct is exact.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scripts.rf_circulant_dnn_layers import extract, standardize, DEV, DT, LAM
from core.rf_circulant_struct import circulant_rf_mmse_struct
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t

JS=[int(x) for x in os.environ.get('JS','1,2,4,8').split(',')]
MS=[int(x) for x in os.environ.get('MS','2,4,8').split(',')]
SIGS=[float(x) for x in os.environ.get('SIGS','0.25,0.5,1.0,2.0').split(',')]
NSEED=int(os.environ.get('NSEED','3')); TB=8

X=standardize(extract('CIFAR-10',n_img=int(os.environ.get('NIMG','10000')))['layer2'].to(DEV,DT))
N,d=X.shape; Xc=X-X.mean(0); ev=torch.linalg.eigvalsh((Xc.T@Xc)/N)
U=torch.zeros(N,1,dtype=DT,device=DEV)
print(f"layer2 map d={d} (28x28), N={N}\n")
res={}
for sg in SIGS:
    res[(sg,'lin')]=float((sg**2*ev/(ev+sg**2)).sum())
    print(f"=== sigma={sg}  linear={res[(sg,'lin')]:.3f} ===")
    for j in JS:
        k=j*d; G=torch.zeros(k,1,dtype=DT,device=DEV)
        ld=np.mean([stein_finiteN_mmse_t(X,U,np.random.default_rng(9+s).standard_normal((k,d))/np.sqrt(d),
                    G,sg,LAM,conditional=False,device=DEV,dtype=DT) for s in range(2)])
        res[(sg,j,0)]=(ld,0.0); row=f"  k/d={j:>2} dense={ld:8.3f}"
        del G; torch.cuda.empty_cache()
        for m in MS:
            c=m*j; vals=[]
            for s in range(NSEED):
                g=torch.Generator(device=DEV); g.manual_seed(900+13*s+c)
                h=torch.zeros(c,d,device=DEV,dtype=DT)
                h[:,:TB]=torch.randn(c,TB,generator=g,device=DEV,dtype=DT)/np.sqrt(TB)
                vals.append(circulant_rf_mmse_struct(X,h,sg,lam=LAM,device=DEV,
                                                     block_chunk=min(c,32),host_budget_gb=40))
                del h; torch.cuda.empty_cache()
            v=np.array(vals); res[(sg,j,m)]=(v.mean(),v.std(ddof=1))
            row+=f" | m={m}(c={c}) {v.mean():8.3f}+-{v.std(ddof=1):5.2f} ({v.mean()-ld:+7.2f})"
        print(row,flush=True)
        np.savez('tables/rf_layer2_rowmult.npz',
                 **{f"{a}|{b}|{c_}":np.array(v) for (a,b,*r) in res for c_,v in [(r[0] if r else '',res[(a,b,*r)])]})
fig,axes=plt.subplots(1,len(SIGS),figsize=(5.0*len(SIGS),4.5))
for i,sg in enumerate(SIGS):
    ax=axes[i]
    ax.plot(JS,[res[(sg,j,0)][0] for j in JS],color='#1f4e9c',marker='o',lw=2.4,ms=6,
            label=r'$\mathcal{L}^{\rm dense}$')
    for m,col,mk in zip(MS,('#e07b39','#c0392b','#7b3fa0'),('^','s','D')):
        mu=[res[(sg,j,m)][0] for j in JS]; sd=[res[(sg,j,m)][1] for j in JS]
        ax.errorbar(JS,mu,yerr=sd,color=col,marker=mk,lw=2.2,ms=5,capsize=3,
                    label=rf'circ $t$=8, $m$={m} ({m}$\times$rows, {d//m}$\times$fewer params)')
    ax.axhline(res[(sg,'lin')],color='darkorange',ls='--',lw=2.2,label='linear (Wiener)')
    ax.set_xscale('log',base=2); ax.set_xticks(JS); ax.set_xticklabels([str(j) for j in JS])
    ax.set_title(rf'$\sigma$={sg}'); ax.set_xlabel('$k/d$'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel(r'$\mathcal{L}_\sigma$'); ax.legend(fontsize=8)
fig.suptitle('layer2 conv map ($d=784$, 28$\\times$28): banded circulant with $m\\times$ the rows of dense '
             'and $784/m\\times$ fewer free parameters\nerror bars = spread over 3 $\\Theta$ draws',fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig('figures/rf_layer2_rowmult.png',dpi=150,bbox_inches='tight')
print("Saved figures/rf_layer2_rowmult.png")
