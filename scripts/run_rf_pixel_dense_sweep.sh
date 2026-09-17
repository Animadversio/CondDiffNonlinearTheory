#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -J densesweep
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/densesweep_%j.out

# Runs in ITS OWN allocation on purpose: the previous two attempts at this experiment were
# launched inside the long-lived nanoclaw allocation and were killed when that rotated.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
export JS="${JS:-3,4,6,8}"
export SIGS="${SIGS:-0.127,0.452,1.610}"
export NSEED="${NSEED:-2}"
echo "JS=${JS}  SIGS=${SIGS}  NSEED=${NSEED}"
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_dense_sweep.py
