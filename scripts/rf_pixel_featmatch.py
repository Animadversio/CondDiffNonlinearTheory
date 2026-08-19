"""FEATURE-MATCH on raw CIFAR pixels: circulant with c = j*d blocks vs dense with k = j*d
rows, i.e. equal free-parameter count. Uses the lag-space noise term so k/d=1,2 fit."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torchvision, torchvision.transforms as T
from core.rf_circulant_struct import circulant_rf_mmse_lag
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t
DEV='cuda'; DT=torch.float64; d=3072; LAM=1e-6; T_BAND=8
JS=[int(x) for x in os.environ.get('JS','1,2').split(',')]
SIGS=[float(x) for x in os.environ.get('SIGS','0.127,0.452,1.610').split(',')]
NIMG=int(os.environ.get('NIMG','10000')); NSEED=int(os.environ.get('NSEED','2'))
ds=torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets',train=True,download=False,transform=T.ToTensor())
dl=torch.utils.data.DataLoader(ds,batch_size=512,num_workers=4); im=[]
for xb,_ in dl:
    im.append(xb.reshape(xb.shape[0],-1))
    if sum(o.shape[0] for o in im)>=NIMG: break
X=torch.cat(im)[:NIMG].to(DEV,DT); N=X.shape[0]
Xc=X-X.mean(0); ev=torch.linalg.eigvalsh((Xc.T@Xc)/N)
U=torch.zeros(N,1,dtype=DT,device=DEV)
print(f"FEATURE-MATCH raw pixels d={d} N={N} Tr(Sigma)={float(ev.sum()):.2f} t={T_BAND}\n")
for sg in SIGS:
    print(f"=== sigma={sg}  linear={float((sg**2*ev/(ev+sg**2)).sum()):.3f} ===")
    for j in JS:
        k=j*d; c=j*d
        G=torch.zeros(k,1,dtype=DT,device=DEV)
        ld=float(np.mean([stein_finiteN_mmse_t(X,U,np.random.default_rng(3+s).standard_normal((k,d))/np.sqrt(d),
                 G,sg,LAM,conditional=False,device=DEV,dtype=DT) for s in range(2)]))
        del G; torch.cuda.empty_cache()
        vals=[]
        for s_ in range(NSEED):
            t0=time.time(); g=torch.Generator(device=DEV); g.manual_seed(800+11*s_+c)
            h=torch.zeros(c,d,device=DEV,dtype=DT)
            h[:,:T_BAND]=torch.randn(c,T_BAND,generator=g,device=DEV,dtype=DT)/np.sqrt(T_BAND)
            vals.append(circulant_rf_mmse_lag(X,h,sg,T_BAND,lam=LAM,device=DEV,
                                              sample_chunk=16,freq_chunk=32))
            print(f"    [seed {s_}: {time.time()-t0:.0f}s]",flush=True)
            del h; torch.cuda.empty_cache()
        v=np.array(vals)
        print(f"  k/d={j}: dense(k={k}, {k*d:,} params) = {ld:8.3f} | "
              f"circ(c={c}, {c*d:,} rows, {c*d:,} params) = {v.mean():8.3f}+-{v.std(ddof=1) if NSEED>1 else 0:.3f} "
              f"({v.mean()-ld:+.3f})",flush=True)
print("done")
