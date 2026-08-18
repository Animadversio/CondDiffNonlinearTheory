import re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=[]; sg=None; lin={}
for f in ('logs/pixel_patch.log','logs/pixel_patch_hi.log'):
    for l in open(f):
        m=re.match(r'=== sigma=([\d.]+)\s+linear=([\d.]+)',l)
        if m: sg=float(m.group(1)); lin[sg]=float(m.group(2)); continue
        if l.strip().startswith('k/d='):
            j=int(re.search(r'k/d=(\d+)',l).group(1)); mm=int(re.search(r'm=(\d+)',l).group(1))
            de=float(re.search(r'dense=\s*([\d.]+)',l).group(1))
            v={k:(float(a),float(b)) for k,a,b in re.findall(r'(band8|patch3c|patch3|band27)=\s*([\d.]+)\+-\s*([\d.]+)',l)}
            rows.append((sg,j,mm,de,v))
sgs=sorted(lin)
fig,axes=plt.subplots(2,3,figsize=(16.5,8.6))
for i,s in enumerate(sgs):
    ax=axes[i//3][i%3]
    sub=[r for r in rows if r[0]==s and r[2]==8]
    js=[r[1] for r in sub]
    ax.plot(js,[r[3] for r in sub],color='#1f4e9c',marker='o',lw=2.4,ms=6,label='dense')
    for nm,col,mk,lab in (('band8','#c0392b','s','band  $t$=8 (8 taps, 1$\\times$8 strip)'),
                          ('patch3','#2d7d46','^','patch 3$\\times$3 (9 taps, 2-D)'),
                          ('band27','#e07b39','v','band  $t$=27 (27 taps)'),
                          ('patch3c','#7b3fa0','D','patch 3$\\times$3$\\times$3 (27 taps, +colour)')):
        mu=[r[4][nm][0] for r in sub]; sd=[r[4][nm][1] for r in sub]
        ax.errorbar(js,mu,yerr=sd,color=col,marker=mk,lw=2.1,ms=5,capsize=3,label=lab)
    ax.axhline(lin[s],color='darkorange',ls='--',lw=2.2,label='linear (Wiener)')
    ax.set_xticks(js); ax.set_xticklabels([str(x) for x in js])
    ax.set_title(rf'$\sigma$={s}'); ax.set_xlabel('$k/d$'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel(r'$\mathcal{L}_\sigma$'); ax.legend(fontsize=8)
fig.suptitle('Does 2-D patch geometry beat a contiguous band? RAW CIFAR PIXELS, $m$=8, tap-matched supports\n'
             'band8 vs patch3 (8 vs 9 taps) and band27 vs patch3c (27 vs 27) are the controlled contrasts',fontsize=13)
fig.tight_layout(rect=[0,0,1,0.89]); fig.savefig('figures/rf_pixel_patch_geometry.png',dpi=150,bbox_inches='tight')
print("Saved figures/rf_pixel_patch_geometry.png")
print("\npatch3 - band8  (m=8, the concentrated points):")
for s in sgs:
    for r in [x for x in rows if x[0]==s and x[2]==8]:
        d1=r[4]['patch3'][0]-r[4]['band8'][0]; d2=r[4]['patch3c'][0]-r[4]['band27'][0]
        print(f"  sigma={s:<6} k/d={r[1]}  patch3-band8={d1:+7.3f}   patch3c-band27={d2:+7.3f}")
