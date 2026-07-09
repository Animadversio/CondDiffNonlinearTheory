#!/bin/bash
#SBATCH -p kempner
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-00:30
#SBATCH -J rf_largek
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/rf_largek_%j.out

set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not found"
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"

# Run with k=8192 (> d=3072) for the large-k comparison
K=8192 N_NOISE=5 N_SIGMA=25 \
    python scripts/rf_theory_vs_empirical.py

echo "Done. Figure saved to figures/rf_theory_vs_empirical_cifar10_k8192.png"
