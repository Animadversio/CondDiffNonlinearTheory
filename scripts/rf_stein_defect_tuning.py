"""
rf_stein_defect_tuning.py — the GMM Stein defect is computable in closed form, so it can
be DIALED. This sweeps a knob, recomputes k_* = d/hat_eps_d^2, and measures where the RF
denoiser actually crosses the linear one, to test whether k_* is predictive.

Finding (d=32, sigma=1, knob = class-mean scale lambda):
  k_* moves 28x (16078 -> 578) and the observed crossover tracks it with
  corr(log k_*, log k_cross) = +0.985 -- so k_* predicts the ORDERING well.
  But k_* is 5-24x too large in absolute terms and its dynamic range is too wide:
  k_cross varies only 6.1x, i.e. empirically k_cross ~ k_*^0.54 ~ sqrt(k_*),
  suggesting the defect enters the true threshold as hat_eps rather than hat_eps^2.

Separately (see the module docstring of rf_linear_in_disguise.py for the defect formula):
  * m_active (how many coordinates carry structure) barely moves the defect at fixed
    power budget: eps_hat^2 = 0.0004..0.0008 for m_active = 2..32.
  * component variance HETEROSCEDASTICITY is the dominant driver: make_gmm has
    Tr(S2)/Tr(S0) = 7.7 and eps_hat^2 = 0.0099, while the homoscedastic
    make_gmm_active has eps_hat^2 = 0.0008 (12.7x smaller).
"""
import sys; sys.path.insert(0,"/n/home12/binxuwang/Github/CondDiffNonlinearTheory")
import numpy as np, torch, importlib.util
from scipy.stats import norm
spec=importlib.util.spec_from_file_location("g","core/gmm.py"); gm=importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
from core.rf_gmm_estimators_torch import mmse_theory_gmm_pop_t
D=32;NC=3;W=[0.5,0.3,0.2];LAM=1e-4
def make_gmm(mean_scale=1.0, seed=42, d=D):
    rng=np.random.default_rng(seed)
    means=np.zeros((NC,d)); means[0,0]=2.0; means[1,0]=-1.0; means[1,1]=1.5
    means[2,0]=-1.0; means[2,1]=-1.0; means[2,2]=1.2
    means*=mean_scale
    s0=np.full(d,0.4); s0[0]=1.2; s1=np.full(d,0.4); s1[0]=0.4; s1[1]=1.0; s1[2]=0.8
    A=rng.standard_normal((d,d))*0.3; S2=A@A.T+0.5*np.eye(d)
    return gm.GaussianMixture(weights=np.array(W),means=means,covs=np.stack([np.diag(s0),np.diag(s1),S2]))
def eps_hat2(gmm,sigma,M=80000,seed=0):
    d=gmm.d; Sinv=np.linalg.inv(gmm.Sigma); rng=np.random.default_rng(seed)
    Th=rng.standard_normal((M,d))/np.sqrt(d); s=sigma*np.linalg.norm(Th,axis=1)
    m=Th@gmm.means.T; v=np.einsum('kd,cde,ke->kc',Th,gmm.covs,Th); S=np.sqrt(v+s[:,None]**2)
    z=m/S; Phi=norm.cdf(z); c0=m*Phi+S*norm.pdf(z)
    dmu=gmm.means-gmm.mu; SigTh=np.einsum('cde,ke->kcd',gmm.covs,Th)
    Cov=np.einsum('c,kc,cd->kd',gmm.weights,c0,dmu)+np.einsum('c,kc,kcd->kd',gmm.weights,Phi,SigTh)
    ab=(gmm.weights[None,:]*Phi).sum(1); Dl=Cov@Sinv.T-ab[:,None]*Th; n2=(Dl**2).sum(1)
    return float(np.sqrt((n2**2).mean()))
KG=[2**i for i in range(3,15)]   # up to 16384
sigma=1.0
print(f"Tuning the Stein defect via the class-mean scale lambda   (d={D}, sigma={sigma})")
print(f"{'lam':>5} {'eps_hat^2':>10} {'k_*=d/e^2':>10} {'k_*/d':>8} {'k_cross':>8} {'k_cross/d':>10} {'k_*/k_cross':>12}")
rows=[]
for lam in [0.25,0.5,1.0,2.0,3.0,4.0,6.0]:
    g=make_gmm(lam); e=eps_hat2(g,sigma); kst=D/e
    lin=g.mmse_uncond_wiener(sigma)
    rp=np.random.default_rng(100)
    rf=[]
    for k in KG:
        Th=rp.standard_normal((k,D))/np.sqrt(D)
        rf.append(mmse_theory_gmm_pop_t(g,Th,np.zeros((k,NC)),sigma,lam=LAM,conditional=False,
                                        device='cuda',dtype=torch.float64))
    rf=np.array(rf); gap=lin-rf; kc=np.nan
    i=int(np.argmax(gap>0))
    if gap[i]>0 and i>0:
        kc=float(np.exp(np.interp(0.0,[gap[i-1],gap[i]],[np.log(KG[i-1]),np.log(KG[i])])))
    rows.append((lam,e,kst,kc))
    print(f"{lam:>5} {e:>10.5f} {kst:>10.0f} {kst/D:>8.1f} {kc:>8.0f} {kc/D:>10.1f} {kst/kc:>12.1f}")
r=np.array([(a,b,c,dd) for a,b,c,dd in rows])
print(f"\n  k_* varies {r[:,2].max()/r[:,2].min():.0f}x  while  k_cross varies {np.nanmax(r[:,3])/np.nanmin(r[:,3]):.1f}x")
cc=np.corrcoef(np.log(r[:,2]),np.log(r[:,3]))[0,1]
print(f"  corr(log k_*, log k_cross) = {cc:+.3f}")
