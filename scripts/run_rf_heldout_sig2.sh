#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#SBATCH -J rf_heldout_sig2
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/heldout_sig2_%j.out

# HELD-OUT EVALUATION, SECOND NOISE BLOCK: sigma = 0.621, 0.853, 1.172, 2.212
# (michimin, 2026-09-23 04:42).
#
# The first block (job 47897466) measured sigma = 0.127 / 0.452 / 1.610 / 5.0 and found the
# verdict FLIPS somewhere in between: at sigma <= 0.452 the 2-D RF beats the held-out Wiener
# outright (-1.61 at c=3072, sigma=0.127) and the 1-D arm ties it, while at sigma >= 1.61 both
# arms stay 5.6-8.6 ABOVE it.  Nothing was measured in the octave where that crossing happens.
# These four sigmas sit inside that gap (0.452 < 0.621 < 0.853 < 1.172 < 1.610 < 2.212 < 5.0)
# and should locate it.
#
# Same c grid, same arms, same seeds, same lam, same 10,000-image train split and the same
# 10,000 CIFAR TEST images as the first block, and it writes the SAME checkpoint table, so the
# result is one grid of 8 sigmas rather than two tables that have to be reconciled.  Existing
# cells are detected and skipped, so this is also safe to re-run.
#
# BOTH SIDES TRAIN ON THE SAME 10,000 IMAGES.  NIMG=10000 for the RF and for the Wiener
# baseline alike (linear_split estimates Sigma on the train split and scores it on test).  No
# number this job produces involves a model fitted on more than 10k images.
#
# ~3.5 h for the first block's 4 sigmas; the budget here is 8 h, and the run exits when done.
# Per-cell checkpointing means a death costs only the in-flight cell.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_heldout.npz
export SIGS=0.621,0.853,1.172,2.212
export T2=3
export T_BAND=8
export NIMG=10000
export NTEST=10000
export NSEED=2

echo "=============== both arms, c = 32 .. 1536 ==============="
CS=32,96,256,512,1536 ARMS=2d,1d \
  /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_heldout.py

echo "=============== 2-D only, c = 3072 ==============="
CS=3072 ARMS=2d \
  /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_heldout.py

echo "ALL DONE"
