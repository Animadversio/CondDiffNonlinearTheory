"""
rf_kstar_image_data.py — Stein defect and k_* for the data denoised in the EDM/DNN-vs-linear
experiment (scripts/dnn_feature_mmse.py), i.e. the IMAGES themselves.

No exponent alpha is fitted: alpha is a property of a d-indexed FAMILY, and CIFAR-10 /
MNIST each sit at one d. We report d/hat_eps_d^2 DIRECTLY at the native d, which is all
k_* = min(d/hat_eps_d^2, K_tensor) needs.

Estimator (VALIDATED against the exact GMM closed form: 8.41/8.48/8.56e-3 vs exact
8.505e-3 at n = 20k/100k/400k). Two mistakes that silently produce NEGATIVE, i.e.
impossible, split-half values, both fixed here:
  * sharing one Sigma_hat across the two halves anti-correlates them -- a fluctuation that
    inflates Sigma_hat depresses the OTHER half's Sigma_hat^-1 Cov. Use a per-half Sigma.
  * using E[x psi] instead of Cov(x, psi): BOTH x and psi must be centered per half.
Shrinkage on Sigma is required (pixel covariances are ill-conditioned) and materially
moves the answer, so a sensitivity sweep is printed rather than a single number.

Caveat: hat_eps_d is basis-dependent (Delta -> A^-T Delta under x -> Ax), because the RF
features theta ~ N(0,I/d) are isotropic in whatever basis you supply. Reported in the pixel
basis with per-pixel standardization.
"""
import sys, os; sys.path.insert(0,"/n/home12/binxuwang/Github/CondDiffNonlinearTheory")
import numpy as np, torch
DEV='cuda'; DT=torch.float64
SIGMA=float(os.environ.get('SIGMA','1.0')); N_THETA=int(os.environ.get('N_THETA','4096'))
NIMG=int(os.environ.get('NIMG','50000'))
CIFAR='/n/home12/binxuwang/.keras/datasets'
MNI='/n/home12/binxuwang/nanoclaw/data/sessions/discord_diffusion-theory-summer-proj/.keras/datasets'
import torchvision, torchvision.transforms as T
def load(name):
    if name=='MNIST': ds=torchvision.datasets.MNIST(MNI,train=True,download=False,transform=T.ToTensor())
    elif name=='CIFAR-10 gray': ds=torchvision.datasets.CIFAR10(CIFAR,train=True,download=False,transform=T.Compose([T.Grayscale(),T.ToTensor()]))
    else: ds=torchvision.datasets.CIFAR10(CIFAR,train=True,download=False,transform=T.ToTensor())
    dl=torch.utils.data.DataLoader(ds,batch_size=4096,num_workers=2); out=[]
    for xb,_ in dl:
        out.append(xb.reshape(xb.shape[0],-1))
        if sum(o.shape[0] for o in out)>=NIMG: break
    X=torch.cat(out)[:NIMG].to(DEV,DT)
    X=X[torch.randperm(X.shape[0],device=DEV,generator=torch.Generator(device=DEV).manual_seed(5))]
    X=X-X.mean(0,keepdim=True); return X/X.std(0,keepdim=True).clamp(min=1e-8)
def hat_eps2(X,sigma,M,shrink,gen):
    """VALIDATED recipe: per-half Sigma, per-half centering of BOTH x and psi."""
    n,d=X.shape
    Th=torch.randn(M,d,device=DEV,dtype=DT,generator=gen)/np.sqrt(d)
    s=sigma*torch.linalg.norm(Th,dim=1); half=n//2; Ds=[]
    for Xh in (X[:half],X[half:2*half]):
        mu=Xh.mean(0); Xc=Xh-mu
        Sg=(Xc.T@Xc)/Xh.shape[0]
        Sg=Sg+shrink*torch.trace(Sg)/d*torch.eye(d,device=DEV,dtype=DT)
        Si=torch.linalg.inv(Sg)
        U=Xh@Th.T; z=U/s[None,:]
        Phi=torch.special.ndtr(z); ph=torch.exp(-0.5*z*z)/np.sqrt(2*np.pi)
        cc=U*Phi+s[None,:]*ph
        Cv=(Xc.T@(cc-cc.mean(0)))/Xh.shape[0]
        Ds.append(Si@Cv - Phi.mean(0)[None,:]*Th.T)
    n2=(Ds[0]*Ds[1]).sum(0)
    return float(n2.mean()), float(np.sqrt(max(float((n2.clamp(min=0)**2).mean()),0.))), float(n2.std()/np.sqrt(len(n2)))
print(f"Stein defect of the data denoised in the EDM/DNN-vs-linear experiment (x0 = images)",flush=True)
print(f"sigma={SIGMA}  n_theta={N_THETA}  n_img={NIMG}   [validated estimator]\n",flush=True)
gen=torch.Generator(device=DEV); gen.manual_seed(0)
print(f"{'dataset':>15} {'d':>6} {'shrink':>7} {'E||D||^2':>11} {'+-se':>9} {'hat_eps^2':>11} {'d/hat_eps^2':>12} {'d^2':>10} {'k_*(n0=2)':>11}",flush=True)
for nm in ['MNIST','CIFAR-10 gray','CIFAR-10 RGB']:
    X=load(nm); d=X.shape[1]
    for sh in [1e-3,1e-2,1e-1]:
        m2,h2,se=hat_eps2(X,SIGMA,N_THETA,sh,gen)
        kd=d/h2 if h2>0 else float('inf'); ks=min(kd,float(d)**2)
        print(f"{nm:>15} {d:>6} {sh:>7.0e} {m2:>11.3e} {se:>9.1e} {h2:>11.3e} {kd:>12.3e} {float(d)**2:>10.2e} {ks:>11.3e}",flush=True)
