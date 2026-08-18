"""RAW PIXELS: x0 = CIFAR image in [0,1] (d=3072), y = x0 + sigma z.
Banded circulant (c = m*j blocks, m x the rows of dense, 3072/m x fewer params) vs dense RF,
against the linear denoiser and the EDM UNet. No conv map, no representation bottleneck --
so sigma, the loss scale and the EDM/linear curves are all directly comparable.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torchvision, torchvision.transforms as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from core.rf_circulant_struct import circulant_rf_mmse_struct
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t
DEV='cuda'; DT=torch.float64; d=3072
JS=[int(x) for x in os.environ.get('JS','1,2,4').split(',')]
MS=[int(x) for x in os.environ.get('MS','2,4,8').split(',')]
SIGS=[float(x) for x in os.environ.get('SIGS','0.127,0.452,0.853,1.610,3.039,5.736').split(',')]
NSEED=int(os.environ.get('NSEED','3')); TB=8; LAM=1e-6
ds=torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets',train=True,download=False,transform=T.ToTensor())
dl=torch.utils.data.DataLoader(ds,batch_size=512,num_workers=4); im=[]
for xb,_ in dl:
    im.append(xb.reshape(xb.shape[0],-1))
    if sum(o.shape[0] for o in im)>=10000: break
X=torch.cat(im)[:10000].to(DEV,DT); N=X.shape[0]
Xc=X-X.mean(0); ev=torch.linalg.eigvalsh((Xc.T@Xc)/N)
tab=np.load('tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz',allow_pickle=True)
U=torch.zeros(N,1,dtype=DT,device=DEV)
print(f"RAW PIXELS d={d}, N={N}, Tr(Sigma)={float(ev.sum()):.2f}\n")
res={}
for sg in SIGS:
    lin=float((sg**2*ev/(ev+sg**2)).sum()); edm=float(np.interp(sg,tab['sigma'],tab['edm_uncond']))
    res[(sg,'lin')]=lin; res[(sg,'edm')]=edm
    print(f"=== sigma={sg}  linear={lin:.3f}  EDM={edm:.3f} ===")
    for j in JS:
        k=j*d; G=torch.zeros(k,1,dtype=DT,device=DEV)
        ld=float(np.mean([stein_finiteN_mmse_t(X,U,np.random.default_rng(3+s).standard_normal((k,d))/np.sqrt(d),
                  G,sg,LAM,conditional=False,device=DEV,dtype=DT) for s in range(2)]))
        res[(sg,j,0)]=(ld,0.0); row=f"  k/d={j} dense k={k:>6} L={ld:8.3f}"
        del G; torch.cuda.empty_cache()
        for m in MS:
            c=m*j; vals=[]
            for s_ in range(NSEED):
                g=torch.Generator(device=DEV); g.manual_seed(400+17*s_+c)
                h=torch.zeros(c,d,device=DEV,dtype=DT)
                h[:,:TB]=torch.randn(c,TB,generator=g,device=DEV,dtype=DT)/np.sqrt(TB)
                vals.append(circulant_rf_mmse_struct(X,h,sg,lam=LAM,device=DEV,
                            block_chunk=int(os.environ.get("BC","4")),host_budget_gb=40))
                del h; torch.cuda.empty_cache()
            v=np.array(vals); res[(sg,j,m)]=(v.mean(),v.std(ddof=1))
            row+=f" | m={m}(c={c},k={c*d}) {v.mean():8.3f}+-{v.std(ddof=1):5.2f}"
        print(row,flush=True)
        np.savez('tables/rf_pixel_rowmult.npz',**{f"{a}|{b}|{r[0] if r else ''}":np.array(v) for (a,b,*r),v in res.items()})
fig,axes=plt.subplots(2,3,figsize=(16.5,8.8))
for i,sg in enumerate(SIGS):
    ax=axes[i//3][i%3]
    ax.plot(JS,[res[(sg,j,0)][0] for j in JS],color='#1f4e9c',marker='o',lw=2.4,ms=6,label=r'$\mathcal{L}^{\rm dense}$')
    for m,col,mk in zip(MS,('#e07b39','#c0392b','#7b3fa0'),('^','s','D')):
        mu=[res[(sg,j,m)][0] for j in JS]; sd=[res[(sg,j,m)][1] for j in JS]
        ax.errorbar(JS,mu,yerr=sd,color=col,marker=mk,lw=2.2,ms=5,capsize=3,
                    label=rf'circ $t$=8, $m$={m} ({m}$\times$rows, {d//m}$\times$fewer params)')
    ax.axhline(res[(sg,'lin')],color='darkorange',ls='--',lw=2.4,label='linear (Wiener)')
    ax.axhline(res[(sg,'edm')],color='seagreen',ls='-.',lw=2.4,label='EDM UNet')
    ax.set_xscale('log',base=2); ax.set_xticks(JS); ax.set_xticklabels([str(j) for j in JS])
    ax.set_title(rf'$\sigma$={sg}'); ax.set_xlabel('$k/d$'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel(r'$\mathcal{L}_\sigma$'); ax.legend(fontsize=8)
fig.suptitle('RAW CIFAR-10 PIXELS ($d=3072$): banded circulant RF vs dense RF vs linear vs EDM UNet\n'
             'no conv map, no bottleneck -- all four are denoising the same $x_0$ at the same $\\sigma$',fontsize=13)
fig.tight_layout(rect=[0,0,1,0.90]); fig.savefig('figures/rf_pixel_rowmult.png',dpi=150,bbox_inches='tight')
print("Saved figures/rf_pixel_rowmult.png")
