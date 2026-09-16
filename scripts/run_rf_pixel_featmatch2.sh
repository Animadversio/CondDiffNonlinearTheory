#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00
#SBATCH -J featmatch2
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/featmatch2_%j.out

# Runs in ITS OWN allocation on purpose: the previous two attempts at this experiment were
# launched inside the long-lived nanoclaw allocation and were killed when that rotated.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
export JS="${JS:-1,0.5,2}"
export SIGS="${SIGS:-0.127,0.452,1.610}"
export NSEED="${NSEED:-2}"
echo "JS=${JS}  SIGS=${SIGS}  NSEED=${NSEED}"
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_featmatch2.py
