"""
rf_circulant_dnn_layers.py — L^circ vs L^dense on DNN layer representations.

Setup
-----
x0 := a ResNet18 layer representation of a CIFAR-10 / MNIST image; y = x0 + sigma Z; the
denoiser is D(y) = W omega(Theta y + eps) + b. We compare, at MATCHED k/d and integer k/d:

  L^dense : Theta rows i.i.d. N(0, I/d), free readout W in R^{d x k}
  L^circ  : block-circulant Theta (c = k/d blocks, block a = circ(h_a)), circulant readout,
            evaluated by michimin's closed form
                L^circ = Tr(Sigma_p0) - sum_{f=1}^{d} q_f^H P_f^{-1} q_f
            with (P_f)_{ab} = f_f^H Sigma_{phi_a,phi_b} f_f and (q_f)_a = f_f^H Sigma_{phi_a,x0} f_f.
  L^circ-banded : same, with h_a supported on t contiguous entries, h_a[:t] ~ N(0, I/t)
            ("Banded Circulant Matrix" section). Variance 1/t (not 1/d) keeps E||h_a||^2 = 1,
            so row norms match the dense RF at every t.

Both curves are built from the SAME Stein/Hermite covariance routine
(core.rf_gmm_estimators_torch.stein_covariances_t), so the only difference between them is
the law of Theta and the constraint on W. That is what makes the comparison meaningful.

TWO REPRESENTATIONS PER DATASET, and the contrast between them is the point
---------------------------------------------------------------------------
  'avgpool' (d=512): the ResNet18 global-average-pooled penultimate vector. This is
      literally the phi used in scripts/dnn_feature_mmse.py. Global average pooling has
      DESTROYED all spatial structure: the 512 coordinates are channels in an arbitrary,
      training-determined order. A circulant Theta imposes a cyclic shift on that ordering,
      and a banded Theta calls channels j and j+1 "neighbours". Neither is meaningful here.
      We include it precisely because it is the honest null: locality should buy nothing.

  'layer2' (d=784 = 28x28): the ResNet18 layer2 feature map (128 x 28 x 28) averaged over
      channels to a single spatial map. This DOES have genuine 2-D locality, so it is the
      representation on which a banded circulant Theta could in principle pay off.

Caveat carried over from the pixel experiments: the circulant construction uses the 1-D
cyclic shift on raster-flattened coordinates, so it treats a 2-D map as a 1-D ring. This is
the same approximation already used for raw pixels, and it makes the wrap-around rows of
each block slightly unnatural; it is not exact 2-D translation equivariance.

Scale convention: each representation is standardized so Tr(Sigma_p0)/d = 1. sigma is then
in units of the representation's own per-coordinate standard deviation and is comparable
across the four (dataset, representation) combinations.

    python scripts/rf_circulant_dnn_layers.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.rf_circulant import build_circulant_theta
from core.rf_circulant_torch import circulant_rf_mmse_t
from core.rf_gmm_estimators_torch import stein_finiteN_mmse_t

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
NIMG    = int(os.environ.get('NIMG', '10000'))
C_LIST  = [int(x) for x in os.environ.get('C_LIST', '1,2,3,4,6,8').split(',')]
SIGMAS  = [float(x) for x in os.environ.get('SIGMAS', '0.5,1.0').split(',')]
T_BAND  = int(os.environ.get('T_BAND', '8'))
N_SEED  = int(os.environ.get('N_SEED', '6'))   # Theta draws per point -- NOT optional, see below
LAM     = float(os.environ.get('LAM', '1e-6'))
SEED    = 0
CIFAR_ROOT = '/n/home12/binxuwang/.keras/datasets'
MNIST_ROOT = os.path.expanduser('~/.keras/datasets')


# ---------------------------------------------------------------------------
# DNN layer representations
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract(dataset, n_img=NIMG, batch=256):
    """Return {'avgpool': (N,512), 'layer2': (N,784)} ResNet18 representations.

    Pre-processing matches scripts/dnn_feature_mmse.py exactly: raw [0,1] -> (1ch expanded
    to 3ch) -> bilinear resize to 224 -> ImageNet normalisation.
    """
    import torchvision, torchvision.transforms as T
    tf = T.ToTensor()
    ds = (torchvision.datasets.MNIST(MNIST_ROOT, train=True, download=False, transform=tf)
          if dataset == 'MNIST' else
          torchvision.datasets.CIFAR10(CIFAR_ROOT, train=True, download=False, transform=tf))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=4)

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(DEV).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).reshape(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEV).reshape(1, 3, 1, 1)

    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                         model.layer1, model.layer2)
    rest = nn.Sequential(model.layer3, model.layer4, model.avgpool, nn.Flatten())

    ap, l2, seen = [], [], 0
    for xb, _ in dl:
        xb = xb.to(DEV)
        if xb.shape[1] == 1:
            xb = xb.expand(-1, 3, -1, -1).contiguous()
        z = F.interpolate(xb.float(), size=224, mode='bilinear', align_corners=False)
        z = (z - mean) / std
        h2 = stem(z)                          # (B,128,28,28)
        ap.append(rest(h2).double().cpu())    # (B,512)
        l2.append(h2.mean(1).flatten(1).double().cpu())   # channel-mean -> (B,784)
        seen += xb.shape[0]
        if seen >= n_img:
            break
    out = {'avgpool': torch.cat(ap)[:n_img], 'layer2': torch.cat(l2)[:n_img]}
    del model, stem, rest; torch.cuda.empty_cache()
    return out


def standardize(X):
    """Center, then scale so Tr(Sigma)/d = 1 (sigma becomes representation-relative)."""
    Xc = X - X.mean(0, keepdim=True)
    return Xc / torch.sqrt((Xc ** 2).sum(1).mean() / Xc.shape[1])


# ---------------------------------------------------------------------------
# Losses at one (representation, sigma, c)
# ---------------------------------------------------------------------------

def losses_at(x0, sigma, c, t_band, seed=SEED):
    """L^dense, L^circ(full width), L^circ(banded t) at k = c*d, for ONE Theta draw.

    WHY THE CALLER MUST AVERAGE OVER DRAWS. A banded block-circulant Theta is generated by
    c filters of t taps, so the whole k x d matrix is a function of c*t independent random
    numbers (16 at c=2, t=8) against c*d^2 for dense. There is therefore essentially no
    concentration over Theta: measured on CIFAR-10/layer2 at sigma=0.5, band-dense is
    +22.8 +- 4.0 at k/d=2 and +24.7 +- 37.0 at k/d=3 across draws, while L^dense is stable
    to +-0.1. A single draw can and did produce a spurious "win" of -4.9. Any statement
    about L^circ must be about E_Theta[L^circ], with the spread reported.
    """
    N, d = x0.shape
    k = c * d
    rng = np.random.default_rng(seed + 1000 * c)
    U = torch.zeros(N, 1, dtype=DT, device=DEV)
    G = torch.zeros(k, 1, dtype=DT, device=DEV)

    Th_dense = rng.standard_normal((k, d)) / np.sqrt(d)
    L_dense = stein_finiteN_mmse_t(x0, U, Th_dense, G, sigma, LAM, conditional=False,
                                   device=DEV, dtype=DT)
    del Th_dense; torch.cuda.empty_cache()

    Th_full = build_circulant_theta(k, d, np.random.default_rng(seed + 7 + 1000 * c), w=None)
    L_circ = circulant_rf_mmse_t(x0, U, Th_full, G, sigma, LAM, conditional=False,
                                 device=DEV, dtype=DT)
    L_circ = float(L_circ['loss'] if isinstance(L_circ, dict) else L_circ)
    del Th_full; torch.cuda.empty_cache()

    Th_band = build_circulant_theta(k, d, np.random.default_rng(seed + 7 + 1000 * c), w=t_band)
    L_band = circulant_rf_mmse_t(x0, U, Th_band, G, sigma, LAM, conditional=False,
                                 device=DEV, dtype=DT)
    L_band = float(L_band['loss'] if isinstance(L_band, dict) else L_band)
    del Th_band; torch.cuda.empty_cache()

    return float(L_dense), L_circ, L_band


def main():
    reps = ['avgpool', 'layer2']
    res = {}
    for dataset in ('CIFAR-10', 'MNIST'):
        print(f"\n########## {dataset}: extracting ResNet18 representations ##########")
        feats = extract(dataset)
        for rep in reps:
            X = standardize(feats[rep].to(DEV, DT))
            N, d = X.shape
            print(f"\n=== {dataset} / {rep}: N={N}, d={d}, Tr(Sigma)/d="
                  f"{float((X**2).sum(1).mean()/d):.4f} ===")
            for sg in SIGMAS:
                key = (dataset, rep, sg)
                res[key] = {'c': [], 'dense': [], 'circ': [], 'band': [], 'd': d,
                            'dense_sd': [], 'circ_sd': [], 'band_sd': []}
                print(f"  sigma={sg}")
                print(f"    {'k/d':>4} {'k':>7} {'L_dense':>15} {'L_circ':>15} "
                      f"{'L_band(t=%d)' % T_BAND:>15} {'band-dense':>9}   "
                      f"(mean +- sd over {N_SEED} Theta draws)")
                for c in C_LIST:
                    trials = [losses_at(X, sg, c, T_BAND, seed=SEED + 137 * s_)
                              for s_ in range(N_SEED)]
                    A = np.array(trials)                      # (n_seed, 3)
                    m, sd = A.mean(0), A.std(0, ddof=1)
                    res[key]['c'].append(c)
                    for q, col in (('dense', 0), ('circ', 1), ('band', 2)):
                        res[key][q].append(m[col]); res[key][q + '_sd'].append(sd[col])
                    print(f"    {c:>4} {c*d:>7} {m[0]:>9.2f}+-{sd[0]:<5.2f} "
                          f"{m[1]:>9.2f}+-{sd[1]:<5.2f} {m[2]:>9.2f}+-{sd[2]:<5.2f} "
                          f"{m[2]-m[0]:>+8.2f}", flush=True)
        del feats; torch.cuda.empty_cache()

    np.savez('tables/rf_circulant_dnn_layers.npz',
             **{f"{k[0]}|{k[1]}|{k[2]}|{q}": np.array(v[q])
                for k, v in res.items() for q in ('c', 'dense', 'circ', 'band',
                                                  'dense_sd', 'circ_sd', 'band_sd')},
             **{f"{k[0]}|{k[1]}|{k[2]}|d": v['d'] for k, v in res.items()})

    # ---- figure: 2 rows (dataset) x 2 cols (representation), one line style per sigma ----
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 12.5})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    styles = {SIGMAS[0]: '-', SIGMAS[-1]: '--'}
    for i, dataset in enumerate(('CIFAR-10', 'MNIST')):
        for j, rep in enumerate(reps):
            ax = axes[i][j]
            for sg in SIGMAS:
                r = res[(dataset, rep, sg)]
                ls = styles.get(sg, '-')
                for q, col, mk, nm in (('dense', '#1f4e9c', 'o', r'\rm dense'),
                                       ('circ', '#7b3fa0', 's', r'\rm circ\ (full)'),
                                       ('band', '#c0392b', '^', rf'\rm circ\ (t={T_BAND})')):
                    mu = np.array(r[q]); sd = np.array(r[q + '_sd'])
                    ax.plot(r['c'], mu, ls, color=col, marker=mk, lw=2.2, ms=5,
                            label=rf'$\mathcal{{L}}^{{{nm}}}$, $\sigma$={sg}')
                    ax.fill_between(r['c'], mu - sd, mu + sd, color=col, alpha=.16, lw=0)
            d = res[(dataset, rep, SIGMAS[0])]['d']
            note = ('GAP: no spatial order — locality is meaningless here'
                    if rep == 'avgpool' else 'conv map 28x28 — genuine 2-D locality')
            ax.set_title(f'{dataset} / {rep}  ($d={d}$)\n{note}', fontsize=11)
            ax.set_xlabel('$k/d$'); ax.set_ylabel(r'denoiser loss  $\mathcal{L}_\sigma$')
            ax.set_xticks(C_LIST); ax.grid(True, alpha=.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8.2, ncol=2)
    fig.suptitle('Circulant vs dense random-feature denoiser on ResNet18 layer '
                 'representations, at matched $k/d$\n'
                 r'closed form $\mathcal{L}^{\rm circ}=\mathrm{Tr}\,\Sigma_{p_0}'
                 r'-\sum_f \mathbf{q}_f^H\mathbf{P}_f^{-1}\mathbf{q}_f$; '
                 'both use the same Stein covariances, so only the law of '
                 r'$\Theta$ and the readout constraint differ.'
                 '\nBands = $\\pm1$ s.d. over %d $\\Theta$ draws: a banded circulant '
                 r'$\Theta$ is generated by only $c\,t$ random numbers, so it barely '
                 'concentrates.' % N_SEED, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs('figures', exist_ok=True)
    out = 'figures/rf_circulant_dnn_layers.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
