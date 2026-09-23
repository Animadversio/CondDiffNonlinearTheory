#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -J rf_band2d_c512_rest
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_c512rest_%j.out

# BAND B=1, c=512, THE REMAINING FOUR NOISES (michimin, 2026-09-23 08:53: "finish running
# band B=1 c=512 against the remaining noises").  Job 47935090 did 0.127/0.452/1.61/2.212;
# this completes the 8-sigma grid that every other held-out arm already has.
#
# Writes into the SAME tables/rf_pixel_band2d_heldout.npz -- the driver skips any cell
# already present, so the four done sigmas cost nothing and the grid ends up in one file.
#
# COST: measured 362-366 s per sigma at c=512 on the completed run, all four identical to
# within 4 s (the work is sigma-independent -- same K, same chunking).  So ~25 min.
# 4 h budget is ~10x slack.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_band2d_heldout.npz
export SIGS=0.621,0.853,1.172,5.0
export CS=512
export BS=1
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=34e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
