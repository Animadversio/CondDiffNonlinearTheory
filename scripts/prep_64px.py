"""Decode the EDM 64x64 zips (FFHQ 70k, AFHQv2 15,803) to uint8 tensors cached in STORE_DIR.

The DiffusionLearningCurve U-nets at 64 px ("FFHQ" / "AFHQ" datasets) index EDM's
ImageFolderDataset over these same zips, i.e. the files in sorted-name order, so image i here
is image i of their dset_start:dset_end splits.  Cached as (N, 3, 64, 64) uint8 -- 4x smaller
than the float tensors, and lossless (the PNGs are 8-bit).

    python scripts/prep_64px.py            # both;  DATASETS=ffhq64 to pick one
"""
import os, io, zipfile
import numpy as np, torch, PIL.Image

STORE = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
ZIPS = {'ffhq64': 'ffhq-64x64.zip', 'afhq64': 'afhqv2-64x64.zip'}
CACHE = os.path.join(STORE, 'CondDiffNonlinearTheory', 'data')


def path(name):
    return os.path.join(CACHE, f'{name}_uint8.pt')


def load(name):
    """(N, 3, 64, 64) uint8, decoding and caching on first use."""
    if os.path.exists(path(name)):
        return torch.load(path(name))
    z = zipfile.ZipFile(os.path.join(STORE, 'Datasets/EDM_datasets/datasets', ZIPS[name]))
    names = sorted(n for n in z.namelist() if n.lower().endswith('.png'))
    X = np.stack([np.array(PIL.Image.open(io.BytesIO(z.read(n))).convert('RGB'))
                  for n in names])                                   # (N, 64, 64, 3)
    X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    os.makedirs(CACHE, exist_ok=True)
    torch.save(X, path(name))
    print(f'{name}: {tuple(X.shape)} -> {path(name)}', flush=True)
    return X


if __name__ == '__main__':
    for n in os.environ.get('DATASETS', 'ffhq64,afhq64').split(','):
        load(n)
