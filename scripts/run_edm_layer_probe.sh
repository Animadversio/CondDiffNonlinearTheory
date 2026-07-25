#!/bin/bash
#SBATCH -p kempner
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-01:00
#SBATCH -J edm_layer_probe
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/edm_layer_probe_%j.out

set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L 2>/dev/null || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"

N=5000 N_NOISE=3 python scripts/edm_layer_linear_probe.py

echo "Done. Saved to figures/edm_layer_probe_cifar10.png"
