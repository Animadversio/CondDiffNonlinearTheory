#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#SBATCH -J rf_band2d_c512
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_c512_%j.out

# BAND-MODULATED *NONLINEAR* RF, B = 1, c = 512, held out (michimin, 2026-09-23 06:36,
# narrowed 07:10 to "just do c=512 and do 4 noises only, the first 2 and 1.61 and 2.21").
#
# THE DECISIVE TEST.  Everything the project has on the band class is LINEAR -- the toll.  The
# load-bearing untested assumption behind michimin's section 5 is that the NONLINEAR gain
# survives when the linear class is enlarged 9x.  This measures it directly, paired on the
# SAME Theta draws as the plain 2-D held-out cells already in tables/rf_pixel_heldout.npz.
#
# HELD OUT, NOT IN SAMPLE, and that is not optional here: 14,155,776 trained parameters
# against 10,000 images is 3x the free Wiener's 9,437,184 and squarely the regime where the
# dense RF was caught memorising on 2026-09-21.
#
# SIGMA ORDER IS DELIBERATE -- most informative first, so a wall-clock death still leaves the
# answer.  0.127 is where the plain 2-D arm's nonlinear gain is largest (1.87 held out) and
# where the band's linear toll collapses hardest (0.441 -> 0.200), i.e. where absorption
# would show up most clearly.  1.61 is where the plain arm fails outright (+5.59 over Wiener)
# and where the band's linear toll is still worth 2.4.  Checkpointed per (sigma, c, B).
#
# COST anchored on a measured c=96 B=1 cell (29 s/seed held out, peak 17.3 GB): the K^2
# moment accumulation dominates and K = c(2B+1)^2, so c=512 is ~28x that plus a c^3 solve
# term => ~15-40 min per sigma, ~1-3 h for the four.  Budget 8 h.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_band2d_heldout.npz
export SIGS=0.127,0.452,1.610,2.212
export CS=512
export BS=1
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=34e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
