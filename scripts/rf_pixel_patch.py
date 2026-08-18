"""Does a genuine 2-D patch support beat a contiguous band, at the same tap budget?

michimin's point: on a raster-flattened image the neighbours of pixel i are i+-1 AND i+-W
(W=32), i+-(W+1), i+-(W-1). A contiguous band of t taps is a 1 x t horizontal STRIP that
wraps into the next row -- it never sees a vertical neighbour. The circulant filter support
should be the periodic offset set of a 2-D patch instead.

Supports compared, all as filters h_a on Z_3072 (CIFAR raster ch*1024 + row*32 + col):
  band8   offsets 0..7                                  (8 taps, 1x8 strip)  [what we ran]
  patch3  offsets {dr*32+dc : dr,dc in -1..1}           (9 taps, 3x3 patch)
  patch3c patch3 plus the same 3x3 on both other colour planes (27 taps)
  band27  offsets 0..26                                 (27 taps, tap-matched to patch3c)
Every support uses variance 1/|S| so E||h_a||^2 = 1 and row norms match dense.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torchvision, torchvision.transforms as T
from core.rf_circulant_struct import circulant_rf_mmse_struct
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t
DEV='cuda'; DT=torch.float64; d=3072; W=32; LAM=1e-6
JS=[int(x) for x in os.environ.get('JS','1,2').split(',')]
MS=[int(x) for x in os.environ.get('MS','2,4,8').split(',')]
SIGS=[float(x) for x in os.environ.get('SIGS','0.127,0.452,1.610').split(',')]
NSEED=int(os.environ.get('NSEED','3'))
SUP={'band8':list(range(8)),
     'patch3':[dr*W+dc for dr in(-1,0,1) for dc in(-1,0,1)],
     'patch3c':[p+ch for ch in(-1024,0,1024) for p in [dr*W+dc for dr in(-1,0,1) for dc in(-1,0,1)]],
     'band27':list(range(27))}
ds=torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets',train=True,download=False,transform=T.ToTensor())
dl=torch.utils.data.DataLoader(ds,batch_size=512,num_workers=4); im=[]
for xb,_ in dl:
    im.append(xb.reshape(xb.shape[0],-1))
    if sum(o.shape[0] for o in im)>=10000: break
X=torch.cat(im)[:10000].to(DEV,DT); N=X.shape[0]
Xc=X-X.mean(0); ev=torch.linalg.eigvalsh((Xc.T@Xc)/N)
U=torch.zeros(N,1,dtype=DT,device=DEV)
print(f"RAW PIXELS d={d} N={N} Tr(Sigma)={float(ev.sum()):.2f}")
print("supports: "+", ".join(f"{k}({len(v)} taps)" for k,v in SUP.items())+"\n")
for sg in SIGS:
    print(f"=== sigma={sg}  linear={float((sg**2*ev/(ev+sg**2)).sum()):.3f} ===")
    for j in JS:
        k=j*d; G=torch.zeros(k,1,dtype=DT,device=DEV)
        ld=float(stein_finiteN_mmse_t(X,U,np.random.default_rng(3).standard_normal((k,d))/np.sqrt(d),
                 G,sg,LAM,conditional=False,device=DEV,dtype=DT)); del G; torch.cuda.empty_cache()
        for m in MS:
            c=m*j; row=f"  k/d={j} m={m}(c={c:>3}) dense={ld:8.3f}"
            for nm,off in SUP.items():
                vals=[]
                for s_ in range(NSEED):
                    g=torch.Generator(device=DEV); g.manual_seed(600+17*s_+c)
                    h=torch.zeros(c,d,device=DEV,dtype=DT)
                    idx=torch.tensor([o%d for o in off],device=DEV)
                    h[:,idx]=torch.randn(c,len(off),generator=g,device=DEV,dtype=DT)/np.sqrt(len(off))
                    vals.append(circulant_rf_mmse_struct(X,h,sg,lam=LAM,device=DEV,block_chunk=4,host_budget_gb=40))
                    del h; torch.cuda.empty_cache()
                v=np.array(vals); row+=f" | {nm}={v.mean():8.3f}+-{v.std(ddof=1):5.2f}"
            print(row,flush=True)
print("done")
