#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -J cond_toll_B456
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/cond_toll_B456_%j.out
#
# Extend the held-out conditional linear tolls to B=5,6; repeat B=4 for comparison.
# All eight existing noise levels, same 10k training images and 10k test images.
# The S family (shared slope + class offsets) is the baseline for conditional RF MODE=vu.
# The existing driver also computes U and C and runs its consistency checks.
#
# Writes a separate table so the original B=0..4 ladder is preserved. This job computes
# linear tolls only: it neither runs RF experiments nor predicts their width dependence.
# Larger B uses bigger linear systems than the original launcher; at B=6 each frequency
# system is 507 x 507, compared with 243 x 243 at B=4.

set -euo pipefail
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:${PYTHONPATH:-}

export NIMG=10000
export NTEST=10000
export SIGS=0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0
export BS=4,5,6
export OUT=tables/rf_cond_toll2d_heldout_B456.npz

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" scripts/rf_cond_toll2d_heldout.py
