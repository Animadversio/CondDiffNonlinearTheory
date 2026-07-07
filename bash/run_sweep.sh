#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH --mem 32gb
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:0
#SBATCH -A kempner_binxuwang_lab
#SBATCH -o logs/sweep_%j.out
#SBATCH -e logs/sweep_%j.err

mkdir -p logs

mamba activate torch2

REPO=/n/home12/binxuwang/Github/CondDiffNonlinearTheory
cd $REPO

python scripts/sweep_sigma.py --d 16 --d_u 8 --k 256 --n_sigma 40 --omega tanh
python scripts/sweep_sigma.py --d 16 --d_u 8 --k 256 --n_sigma 40 --omega sigmoid
python scripts/verify_jointly_gaussian.py
python scripts/verify_hermite_coeffs.py
