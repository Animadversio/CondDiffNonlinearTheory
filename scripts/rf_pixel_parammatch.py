"""Matched FREE PARAMETERS: circulant (c blocks) vs dense (k = c rows), raw CIFAR pixels.

Theta has k*d free entries dense and c*d block-circulant, so equal parameter budget means
k = c. The circulant then has c*d ROWS against dense's c -- a factor d = 3072 more features
for the same number of free parameters. Banded t=8. Linear and EDM shown as references.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torchvision, torchvision.transforms as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from core.rf_circulant_struct import circulant_rf_mmse_struct
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t
DEV='cuda'; DT=torch.float64; d=3072; LAM=1e-6; TB=8
CS=[int(x) for x in os.environ.get('CS','2,4,8,16,32,64,128,256').split(',')]
SIGS=[float(x) for x in os.environ.get('SIGS','0.127,0.452,1.610,5.736').split(',')]
NSEED=int(os.environ.get('NSEED','3'))
ds=torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets',train=True,download=False,transform=T.ToTensor())
dl=torch.utils.data.DataLoader(ds,batch_size=512,num_workers=4); im=[]
for xb,_ in dl:
    im.append(xb.reshape(xb.shape[0],-1))
    if sum(o.shape[0] for o in im)>=10000: break
X=torch.cat(im)[:10000].to(DEV,DT); N=X.shape[0]
Xc=X-X.mean(0); ev=torch.linalg.eigvalsh((Xc.T@Xc)/N)
tab=np.load('tables/dnn_feature_mmse_cifar10_N10000_noise5_sigma30.npz',allow_pickle=True)
U=torch.zeros(N,1,dtype=DT,device=DEV)
print(f"RAW PIXELS d={d} N={N} Tr(Sigma)={float(ev.sum()):.2f}\n")
res={}
for sg in SIGS:
    lin=float((sg**2*ev/(ev+sg**2)).sum()); edm=float(np.interp(sg,tab['sigma'],tab['edm_uncond']))
    res[(sg,'lin')]=lin; res[(sg,'edm')]=edm
    print(f"=== sigma={sg}  linear={lin:.3f}  EDM={edm:.3f} ===")
    for c in CS:
        G=torch.zeros(c,1,dtype=DT,device=DEV)
        ld=float(np.mean([stein_finiteN_mmse_t(X,U,np.random.default_rng(5+s).standard_normal((c,d))/np.sqrt(d),
                 G,sg,LAM,conditional=False,device=DEV,dtype=DT) for s in range(2)]))
        del G; torch.cuda.empty_cache()
        vals=[]
        for s_ in range(NSEED):
            g=torch.Generator(device=DEV); g.manual_seed(800+19*s_+c)
            h=torch.zeros(c,d,device=DEV,dtype=DT)
            h[:,:TB]=torch.randn(c,TB,generator=g,device=DEV,dtype=DT)/np.sqrt(TB)
            vals.append(circulant_rf_mmse_struct(X,h,sg,lam=LAM,device=DEV,sample_chunk=64))
            del h; torch.cuda.empty_cache()
        v=np.array(vals); res[(sg,c)]=(ld,v.mean(),v.std(ddof=1))
        print(f"  params={c*d:>8}  dense(k={c:>4} rows)={ld:9.3f} | circ(c={c:>4}, {c*d:>7} rows)="
              f"{v.mean():9.3f}+-{v.std(ddof=1):5.3f}  ({v.mean()-ld:+8.3f})",flush=True)
np.savez('tables/rf_pixel_parammatch.npz',**{f"{a}|{b}":np.array(v) for (a,b),v in res.items()})
fig,axes=plt.subplots(1,len(SIGS),figsize=(5.2*len(SIGS),4.6))
for i,sg in enumerate(SIGS):
    ax=axes[i]
    ax.plot(CS,[res[(sg,c)][0] for c in CS],color='#1f4e9c',marker='o',lw=2.4,ms=6,label=r'dense ($k=c$ rows)')
    mu=[res[(sg,c)][1] for c in CS]; sd=[res[(sg,c)][2] for c in CS]
    ax.errorbar(CS,mu,yerr=sd,color='#c0392b',marker='s',lw=2.4,ms=6,capsize=3,
                label=r'circulant $t$=8 ($c\,d$ rows)')
    ax.axhline(res[(sg,'lin')],color='darkorange',ls='--',lw=2.2,label='linear (Wiener)')
    ax.axhline(res[(sg,'edm')],color='seagreen',ls='-.',lw=2.2,label='EDM UNet')
    ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_xticks(CS); ax.set_xticklabels([str(c) for c in CS])
    ax.set_title(rf'$\sigma$={sg}'); ax.set_xlabel('free parameters / $d$   (= $c$)'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel(r'$\mathcal{L}_\sigma$'); ax.legend(fontsize=8.5)
fig.suptitle('RAW CIFAR PIXELS, MATCHED FREE PARAMETERS: circulant gets $d=3072\\times$ more rows '
             'for the same $c\\,d$ parameters\n(dense uses $k=c$ rows; circulant uses $c$ blocks $=c\\,d$ rows)',fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig('figures/rf_pixel_parammatch.png',dpi=150,bbox_inches='tight')
print("Saved figures/rf_pixel_parammatch.png")
