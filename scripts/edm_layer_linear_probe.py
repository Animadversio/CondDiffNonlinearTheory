"""
EDM UNet intermediate-layer linear denoiser.

Extract activations h from a SongUNet layer, fit a ridge-regression
linear denoiser, and compare against the full EDM denoiser / linear Wiener.

EDM preconditioning (EDMPrecond, sigma_data=0.5):
  c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
  c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
  c_in    = 1 / sqrt(sigma_data^2 + sigma^2)
  c_noise = log(sigma) / 4
  D(y)    = c_skip * y + c_out * F(c_in * y, c_noise)

Two linear-denoiser variants from extracted h:
  no-skip:   D = W h + b            (fit x0 ← W h + b)
  with-skip: D = c_skip*y + c_out*(W h + b)
             (fit (x0 - c_skip*y)/c_out ← W h + b)

Image convention: x0_edm in [-1,1], sigma_edm = 2 * sigma_pixel.
MSE reported in pixel [0,1] units (divide EDM MSE by 4).

Evaluation: train x0 + fresh Z (finite-dataset MMSE standard).
"""

import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/edm')

import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
STORE    = os.environ.get('STORE_DIR',
           '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
CKPT_DIR = os.path.join(STORE, 'Datasets/EDM_datasets/edm_ckpts')
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

N        = int(os.environ.get('N', '5000'))
N_NOISE  = int(os.environ.get('N_NOISE', '3'))
BATCH    = int(os.environ.get('BATCH', '256'))
LAM      = float(os.environ.get('LAM', '1e-2'))

SIGMA_PIXEL_GRID = np.logspace(np.log10(0.02), np.log10(5.0), 20)

# Each entry: (enc/dec, block_name, short_label)
PROBE_LAYERS = [
    ('enc', '8x8_block3',   'enc-8x8 (bottleneck in)'),
    ('dec', '8x8_in1',      'dec-8x8-in1 (bottleneck)'),
    ('dec', '8x8_block4',   'dec-8x8-blk4 (bottleneck out)'),
    ('dec', '16x16_block4', 'dec-16x16 (256×16×16=65k)'),
    # dec.32x32_block4 omitted from default run: 256×32×32=262k dims,
    # storing full (N*N_NOISE, 262k) tensor requires ~15GB. Enable via env
    # INCLUDE_32x32=1 with small N.
]
if os.environ.get('INCLUDE_32x32'):
    PROBE_LAYERS.append(('dec', '32x32_block4', 'dec-32x32 (256×32×32=262k)'))


# ---------------------------------------------------------------------------
def load_cifar10(n, device):
    ds = torchvision.datasets.CIFAR10(
        root=os.path.join(STORE, 'Datasets'), train=True, download=False,
        transform=T.ToTensor())
    xs = []
    for img, _ in ds:
        xs.append(img.flatten())
        if len(xs) >= n:
            break
    x0_pixel = torch.stack(xs).to(device)   # (N, 3072) in [0,1]
    x0_edm   = x0_pixel.reshape(n, 3, 32, 32) * 2.0 - 1.0  # (N,3,32,32) [-1,1]
    return x0_pixel, x0_edm


def edm_coeffs(sigma_pixel, sigma_data=0.5):
    sigma = 2.0 * sigma_pixel   # EDM noise scale
    sd2   = sigma_data ** 2
    s2    = sigma ** 2
    c_skip  = sd2 / (s2 + sd2)
    c_out   = sigma * sigma_data / (s2 + sd2) ** 0.5
    c_in    = 1.0 / (sd2 + s2) ** 0.5
    c_noise = float(np.log(sigma) / 4.0)
    return c_skip, c_out, c_in, c_noise, sigma


@torch.no_grad()
def run_forward_with_hook(unet, y_edm, c_in, c_noise, dict_name, block_name,
                          batch_size=BATCH):
    """
    Run SongUNet (unet.model) forward with a hook on one block.
    y_edm: (N,3,32,32) in EDM pixel space [-1,1].
    Returns h: (N, k) float32 where k = C*H*W of the hooked block.
    """
    model = unet.model
    block = getattr(model, dict_name)[block_name]
    buf   = []
    def _hook(m, inp, out):
        buf.append(out.detach().float())
    handle = block.register_forward_hook(_hook)

    N_ = y_edm.shape[0]
    c_n_v = torch.full([1], c_noise, device=DEVICE, dtype=torch.float32)
    for s in range(0, N_, batch_size):
        e   = min(s + batch_size, N_)
        y_b = y_edm[s:e]
        c_n_b = c_n_v.expand(len(y_b))
        model(c_in * y_b, c_n_b, class_labels=None)

    handle.remove()
    h = torch.cat(buf, dim=0)       # (N, C, H, W)
    return h.reshape(h.shape[0], -1)  # (N, k)


def ridge_solve(H, T, lam):
    """
    Fit T ≈ H W (+ intercept via centering).
    Uses dual form when M < k, primal when M >= k.
    Returns W (k, d), b (d,) — all float32.
    """
    H = H.double(); T = T.double()
    M, k = H.shape
    mu_h = H.mean(0); mu_t = T.mean(0)
    Hc = H - mu_h; Tc = T - mu_t
    if M < k:
        G = Hc @ Hc.T / M; G.diagonal().add_(lam)
        alpha = torch.linalg.solve(G, Tc / M)
        W = Hc.T @ alpha
    else:
        A = Hc.T @ Hc / M; A.diagonal().add_(lam)
        W = torch.linalg.solve(A, Hc.T @ Tc / M)
    b = mu_t - mu_h @ W
    return W.float(), b.float()


@torch.no_grad()
def probe_one_sigma(net, x0_pixel, x0_edm, sigma_pixel, dict_name, block_name):
    """
    Collect features over N_NOISE draws, fit W (no-skip and with-skip),
    evaluate on one fresh draw.
    Returns pixel MSE for 'noskip' and 'skip'.
    """
    N_ = len(x0_pixel)
    c_skip, c_out, c_in, c_noise, sigma_edm = edm_coeffs(sigma_pixel, net.sigma_data)
    x0_flat = x0_edm.reshape(N_, -1)   # (N, 3072) EDM scale

    # --- collect training features ---
    hs, x0s, ys = [], [], []
    for _ in range(N_NOISE):
        Z    = torch.randn_like(x0_edm) * sigma_edm
        y    = x0_edm + Z
        h    = run_forward_with_hook(net, y, c_in, c_noise, dict_name, block_name)
        hs.append(h)
        x0s.append(x0_flat)
        ys.append(y.reshape(N_, -1))
    H    = torch.cat(hs,  dim=0)   # (M, k)
    X0   = torch.cat(x0s, dim=0)   # (M, 3072)
    Y    = torch.cat(ys,  dim=0)   # (M, 3072)

    # --- fit ---
    W_ns, b_ns = ridge_solve(H, X0, LAM)

    T_sk = (X0 - c_skip * Y) / c_out         # (M, 3072)
    W_sk, b_sk = ridge_solve(H, T_sk, LAM)

    # --- fresh evaluation ---
    Z_f  = torch.randn_like(x0_edm) * sigma_edm
    y_f  = x0_edm + Z_f
    h_f  = run_forward_with_hook(net, y_f, c_in, c_noise, dict_name, block_name)
    yf_flat = y_f.reshape(N_, -1)

    pred_ns = h_f @ W_ns + b_ns                          # (N, 3072) EDM scale
    pred_sk = c_skip * yf_flat + c_out * (h_f @ W_sk + b_sk)

    # convert EDM → pixel, compute MSE (sum over pixels, mean over images)
    def edm_to_pixel_mse(pred_edm, x0_edm_flat):
        pred_pix = (pred_edm + 1.0) / 2.0
        x0_pix   = (x0_edm_flat + 1.0) / 2.0
        return float(((pred_pix - x0_pix) ** 2).sum(dim=1).mean())

    return {
        'noskip': edm_to_pixel_mse(pred_ns, x0_flat),
        'skip':   edm_to_pixel_mse(pred_sk, x0_flat),
    }


@torch.no_grad()
def linear_wiener_mse(x0_pixel, sigma_pixel):
    N_, d = x0_pixel.shape
    x0_c  = x0_pixel - x0_pixel.mean(0)
    cov   = x0_c.T.double() @ x0_c.double() / N_
    ev    = torch.linalg.eigvalsh(cov).clamp(min=0)
    s2    = sigma_pixel ** 2
    return float((s2 * ev / (ev + s2)).sum())


@torch.no_grad()
def edm_full_mse(net, x0_pixel, x0_edm, sigma_pixel, n_noise=1):
    N_  = len(x0_pixel)
    sigma_edm = 2.0 * sigma_pixel
    total, count = 0.0, 0
    for _ in range(n_noise):
        Z = torch.randn_like(x0_edm) * sigma_edm
        y = x0_edm + Z
        for s in range(0, N_, BATCH):
            e   = min(s + BATCH, N_)
            y_b = y[s:e]
            sig_v = torch.full([len(y_b)], sigma_edm, device=DEVICE, dtype=torch.float32)
            D_b = net(y_b, sig_v, class_labels=None)   # (B,3,32,32) EDM scale
            D_pix = (D_b + 1.0) / 2.0
            x0_b  = x0_pixel[s:e].reshape(len(y_b), 3, 32, 32)
            total += float(((D_pix - x0_b) ** 2).sum())
            count += len(y_b)
    return total / count


# ---------------------------------------------------------------------------
def main():
    print(f"Device={DEVICE}  N={N}  N_NOISE={N_NOISE}  LAM={LAM}")
    print("Loading CIFAR-10 ...")
    x0_pixel, x0_edm = load_cifar10(N, DEVICE)
    print(f"  x0_pixel {x0_pixel.shape}, mean={x0_pixel.mean():.3f}")

    print("Loading EDM checkpoint ...")
    with open(os.path.join(CKPT_DIR, 'edm-cifar10-32x32-uncond-ve.pkl'), 'rb') as f:
        net = pickle.load(f)['ema'].to(DEVICE).eval()
    print(f"  {type(net).__name__}, sigma_data={net.sigma_data}")

    all_results = {'sigma': SIGMA_PIXEL_GRID}

    for sigma in tqdm(SIGMA_PIXEL_GRID):
        sigma = float(sigma)

        all_results.setdefault('linear_wiener', []).append(
            linear_wiener_mse(x0_pixel, sigma))
        all_results.setdefault('edm_uncond', []).append(
            edm_full_mse(net, x0_pixel, x0_edm, sigma, n_noise=1))

        for dict_name, block_name, layer_label in PROBE_LAYERS:
            lk = f'{dict_name}.{block_name}'
            try:
                r = probe_one_sigma(net, x0_pixel, x0_edm, sigma, dict_name, block_name)
                all_results.setdefault(f'{lk}_noskip', []).append(r['noskip'])
                all_results.setdefault(f'{lk}_skip',   []).append(r['skip'])
            except Exception as ex:
                print(f"\n  ERROR {lk}: {ex}")
                for tag in ('noskip', 'skip'):
                    all_results.setdefault(f'{lk}_{tag}', []).append(float('nan'))

    for k in all_results:
        if k != 'sigma':
            all_results[k] = np.array(all_results[k])

    os.makedirs('tables', exist_ok=True)
    npz_path = 'tables/edm_layer_probe_cifar10.npz'
    np.savez(npz_path, **all_results)
    print(f"\nSaved {npz_path}")

    mid = len(SIGMA_PIXEL_GRID) // 2
    print(f"\n=== Summary at sigma={SIGMA_PIXEL_GRID[mid]:.3f} ===")
    print(f"  linear_wiener : {all_results['linear_wiener'][mid]:.5f}")
    print(f"  edm_uncond    : {all_results['edm_uncond'][mid]:.5f}")
    for dict_name, block_name, layer_label in PROBE_LAYERS:
        lk = f'{dict_name}.{block_name}'
        ns = all_results.get(f'{lk}_noskip', [np.nan]*20)[mid]
        sk = all_results.get(f'{lk}_skip',   [np.nan]*20)[mid]
        print(f"  {layer_label:40s}  ns={ns:.5f}  sk={sk:.5f}")

    # Plot
    sigma_arr = SIGMA_PIXEL_GRID
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'EDM Layer Linear Probe — CIFAR-10  N={N}, N_noise={N_NOISE}, λ={LAM}')
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(PROBE_LAYERS)))

    for ax_idx, ax in enumerate(axes):
        ax.plot(sigma_arr, all_results['linear_wiener'], 'k--', lw=2, label='Linear Wiener')
        ax.plot(sigma_arr, all_results['edm_uncond'],    'r-',  lw=2.5, label='Full EDM UNet')

        for ci, (dict_name, block_name, layer_label) in enumerate(PROBE_LAYERS):
            lk = f'{dict_name}.{block_name}'
            for ls, skip_tag, slabel in [('-', 'noskip', 'no-skip'),
                                          ('--', 'skip',   '+EDM-skip')]:
                key = f'{lk}_{skip_tag}'
                if key not in all_results:
                    continue
                vals = all_results[key]
                ref  = all_results['linear_wiener']
                y    = vals if ax_idx == 0 else ref - vals
                short = layer_label.split('(')[0].strip()
                ax.plot(sigma_arr, y, color=colors[ci], ls=ls, lw=1.5,
                        label=f'{short} [{slabel}]')

        if ax_idx == 1:
            ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_xlabel('sigma (pixel units)')
        ax.set_ylabel('MSE' if ax_idx == 0 else 'L_Wiener − L_probe')
        ax.set_title('Layer probe MSE' if ax_idx == 0 else 'Gain over linear Wiener')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = 'figures/edm_layer_probe_cifar10.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    main()
