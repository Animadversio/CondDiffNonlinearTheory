"""RF denoiser on the layer2 conv map, predicting RAW PIXELS. Circulant-Theta vs dense.

  y = x0 + sigma z   (noise at PIXEL level, as in dnn_feature_mmse.py)
  z_L2 = channel-mean of ResNet18 layer2(y)  in R^784  (28x28 conv map)
  phi  = relu(Theta z_L2),  Theta in R^{k x 784}       dense  or  block-circulant (banded t)
  xhat = W phi + b,         W in R^{3072 x 784...k}    FREE (see below)
  L    = Tr(Sigma_x0) - Cov(x0,phi) Sigma_phi^{-1} Cov(phi,x0)

TWO THINGS THAT DIFFER FROM EVERY EARLIER SCRIPT HERE, both forced by the setup:

1. W CANNOT BE CIRCULANT. It maps R^k -> R^3072 while Theta's blocks live on Z_784, and
   there is no cyclic group shared between 784 input positions and 3072 output pixels. So
   michimin's per-frequency closed form does not apply and W is necessarily free. This is
   not a limitation but the interesting ablation: circulant FEATURE map, unconstrained
   readout -- the configuration that won in the GMM experiments.

2. THE STEIN/MEHLER COVARIANCES CANNOT BE USED. That machinery marginalises Gaussian noise
   applied directly to Theta's input. Here Theta's input is z_L2, a ResNet feature of a
   noisy image -- a nonlinear function of the noise, not a Gaussian perturbation. So
   Sigma_phi and Cov(x0,phi) must be EMPIRICAL, estimated from sampled noise draws. Both
   the dense and circulant curves use the same empirical estimator, so the comparison stays
   matched; but the numbers are not comparable to the Stein-based figures from earlier.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import torchvision.transforms as T

DEV='cuda'; DT=torch.float64
NIMG  = int(os.environ.get('NIMG','10000'))
NDRAW = int(os.environ.get('NDRAW','2'))
JS    = [int(x) for x in os.environ.get('JS','1,2,4').split(',')]
MS    = [int(x) for x in os.environ.get('MS','2,4').split(',')]
TBAND = int(os.environ.get('TBAND','8'))
SIGS  = [float(x) for x in os.environ.get('SIGS','0.452,0.853,1.610,2.212,3.039,4.175,7.880').split(',')]
LAM   = float(os.environ.get('LAM','1e-4'))
DIN   = 784

def build():
    m=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(DEV).eval()
    return m, nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1,m.layer2)

@torch.no_grad()
def feats(stem, X0img, sigma, gen):
    mean=torch.tensor([0.485,0.456,0.406],device=DEV).reshape(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=DEV).reshape(1,3,1,1)
    out=[]
    for i in range(0,X0img.shape[0],128):
        xb=X0img[i:i+128]
        y=(xb+sigma*torch.randn(xb.shape,device=DEV,generator=gen,dtype=xb.dtype)).clamp(0,1)
        z=(F.interpolate(y.float(),size=224,mode='bilinear',align_corners=False)-mean)/std
        out.append(stem(z).mean(1).flatten(1).double())
    return torch.cat(out)

def theta_dense(k,g): return torch.randn(k,DIN,generator=g,device=DEV,dtype=DT)/np.sqrt(DIN)
def theta_circ(c,t,g):
    h=torch.zeros(c,DIN,device=DEV,dtype=DT)
    h[:,:t]=torch.randn(c,t,generator=g,device=DEV,dtype=DT)/np.sqrt(t)
    idx=(torch.arange(DIN,device=DEV).unsqueeze(0)-torch.arange(DIN,device=DEV).unsqueeze(1))%DIN
    return torch.cat([h[a][idx] for a in range(c)])       # (c*DIN, DIN)

def loss(Z,X0,Th):
    P=torch.relu(Z@Th.T); P=P-P.mean(0); Xc=X0-X0.mean(0)
    k=P.shape[1]; S=P.T@P/P.shape[0]+LAM*torch.eye(k,device=DEV,dtype=DT)
    C=Xc.T@P/P.shape[0]
    return float((Xc**2).sum()/Xc.shape[0]-torch.trace(C@torch.linalg.solve(S,C.T)))

def main():
    ds=torchvision.datasets.CIFAR10('/n/home12/binxuwang/.keras/datasets',train=True,
                                    download=False,transform=T.ToTensor())
    dl=torch.utils.data.DataLoader(ds,batch_size=512,num_workers=4); img=[]
    for xb,_ in dl:
        img.append(xb)
        if sum(o.shape[0] for o in img)>=NIMG: break
    X0img=torch.cat(img)[:NIMG].to(DEV,DT)
    X0=X0img.reshape(NIMG,-1)
    _,stem=build()
    print(f"N={NIMG} x {NDRAW} draws, d_in=784 (layer2 28x28), d_out=3072 (pixels)")
    print(f"Tr(Sigma_x0)={float(((X0-X0.mean(0))**2).sum()/NIMG):.2f}\n")
    res={}
    for sg in SIGS:
        g=torch.Generator(device=DEV); g.manual_seed(int(sg*1000))
        Z=torch.cat([feats(stem,X0img,sg,g) for _ in range(NDRAW)])
        Xr=X0.repeat(NDRAW,1)
        print(f"=== sigma={sg} (N_eff samples={Z.shape[0]}) ===")
        for j in JS:
            k=j*DIN
            gd=torch.Generator(device=DEV); gd.manual_seed(11+j)
            t0=time.time(); ld=loss(Z,Xr,theta_dense(k,gd))
            row=f"  k/d={j:>2} dense k={k:>6} L={ld:>8.3f}"
            res[(sg,j,0)]=ld
            for m in MS:
                c=m*j; gc=torch.Generator(device=DEV); gc.manual_seed(77+c)
                lc=loss(Z,Xr,theta_circ(c,TBAND,gc))
                res[(sg,j,m)]=lc
                row+=f" | circ m={m} (c={c},k={c*DIN}) L={lc:>8.3f} ({lc-ld:+.2f})"
            print(row+f"  [{time.time()-t0:.0f}s]",flush=True)
            np.savez('tables/rf_layer2_pixel.npz',**{f'{a}|{b}|{c_}':v for (a,b,c_),v in res.items()})
        del Z,Xr; torch.cuda.empty_cache()
    print("done")

main()
