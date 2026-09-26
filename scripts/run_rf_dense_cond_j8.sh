#!/bin/bash
#SBATCH -J rf_dense_cond_j8
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -o logs/dense_cond_j8_%j.out
#SBATCH -e logs/dense_cond_j8_%j.err

set -euo pipefail
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
export PYTHONNOUSERSITE=1
export J=8 NSEED=${NSEED:-1}
export SIGS=${SIGS:-0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0}
export OUT=${OUT:-tables/rf_pixel_dense_cond_heldout_vu_j8.npz}
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_dense_cond_heldout.py
