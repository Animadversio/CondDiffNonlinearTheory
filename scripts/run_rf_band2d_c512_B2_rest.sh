#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-12:00
#SBATCH -J rf_band2d_c512_B2r
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_c512_B2rest_%j.out

# BAND-MODULATED *NONLINEAR* RF, B = 2, c = 512, the FOUR REMAINING sigma, held out
# (michimin, 2026-09-23 09:26: "queue B=2 run for all the remaining noises too").
#
# Companion to scripts/run_rf_band2d_c512_B2.sh (job 47948435), which covers
# sigma = 1.610, 2.212, 0.127, 0.452.  Together the two jobs give the full 8-sigma B=2 grid,
# matching the B=1 c=512 grid cell for cell so every comparison is paired on sigma.
#
# *** WHY A SEPARATE OUT FILE -- THIS IS A CORRECTNESS REQUIREMENT, NOT TIDINESS. ***
# scripts/rf_pixel_band2d_heldout.py reads the whole table ONCE at startup into `store` and
# then rewrites ALL of it (`np.savez(OUT, **store)`) after every finished cell.  Two jobs
# pointed at the same npz therefore CLOBBER each other: whichever saves second writes its own
# startup snapshot plus its own new cell, silently deleting every cell the other job finished
# in the meantime.  The skip-if-present resume logic does not protect against this -- it only
# runs at startup.  So this job writes its own file and the two are merged afterwards with
#     python scripts/merge_band2d_tables.py
# which refuses to run if the two files disagree on any shared key.
# (The alternative was --dependency=afterok, but that serialises ~1.5 h of GPU work for no
# reason when the two sigma sets are disjoint and a merge is exactly verifiable.)
#
# SIGMA ORDER: 0.621, 0.853, 1.172 first, 5.0 last.  The mid-sigma octave is where the
# linear->EDM gap PEAKS (9.73 at sigma ~ 0.62-0.85) and where no class we have tested
# captures the available nonlinear gain, so those three cells carry the most information.
# sigma=5.0 goes last because it is the one cell where B=1 already told us the answer is
# unfavourable: retention BROKE there (80.7%, the only sigma below 100% apart from 0.127),
# the linear toll alone collapses 7.41 -> 3.20, and relu's contribution FELL 0.133 -> 0.107.
# A wall-clock death that loses only sigma=5.0 loses the least.
#
# COST: identical shape to the companion job (same c, same B => same K = 12800, same
# memory), so the same bracket applies: ~14-24 min per sigma, ~1-1.6 h for the four.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_band2d_heldout_B2rest.npz
export SIGS=0.621,0.853,1.172,5.0
export CS=512
export BS=2
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
