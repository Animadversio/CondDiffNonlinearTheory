#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -J rf_band2d_seed1
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_seed1_sig0127_%j.out

# SECOND THETA DRAW AT sigma=0.127, B=1 AND B=2, c=512 (michimin, 2026-09-23 19:08:
# "sure run a second seed at sigma=0.127 for both B").
#
# *** WHAT THIS MEASURES AND WHY ONLY HERE. ***  Every band cell in the project is NSEED=1.
# The band-minus-plain PAIRED differential is a within-draw quantity (filt2d(c,s) depends only
# on (c,seed), not on B or sigma, so plain / B=1 / B=2 at c=512 seed 0 share the IDENTICAL
# base filters), but a single draw gives no estimate of that differential's own spread.
# Against the plain arm's 2-seed spread at c=512 the B=2 differentials run 72-2930x at seven
# of the eight sigma -- those need nothing.  sigma=0.127 is the exception at 7.5x:
#
#     sigma=0.127   plain 2-seed spread 0.0076
#                   B=1 paired d  -0.0644   ( 8.5x)
#                   B=2 paired d  +0.0573   ( 7.5x)   <- the interior-optimum headline
#
# and that 7.5x is measured against the PLAIN arm.  We have no direct measurement of the BAND
# arm's Theta variance at any cell, and B=2 carries 2.8x the parameters (39.3M vs 14.2M).
# This job measures it directly at the only cell whose SIGN depends on it.
#
# BOTH B IN ONE JOB, B=1 FIRST: B=1 is ~6 min and B=2 ~28 min on the measured per-cell timing,
# so the cheap arm lands first and the whole job is ~35 min.
#
# *** REQUIRES THE SEED-GRANULARITY RESUME ADDED TO THE DRIVER TODAY. ***  The old skip rule
# was `if key in store: continue`, i.e. cell granularity -- with the 1-seed cells already
# stored, NSEED=2 would have skipped both cells and done nothing at all.  The driver now
# compares the number of stored ROWS against NSEED and appends only the missing draws, so
# seed 0 is NOT recomputed.  (With no stored rows the behaviour is bit-identical to before.)
#
# SAFE TO WRITE THE CANONICAL TABLE DIRECTLY: this is the ONLY job running against it, so the
# load-once/save-all race that forced the two B=2 jobs onto separate files does not apply.
# Backup at /tmp/band2d_backup_preseed1.npz.  The driver now also ASSERTS that the recomputed
# `linear|0.127` Wiener row reproduces the stored one (linear_split has no RNG) rather than
# silently overwriting it.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_band2d_heldout.npz
export SIGS=0.127
export CS=512
export BS=1,2
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=2
export BUDGET=70e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
