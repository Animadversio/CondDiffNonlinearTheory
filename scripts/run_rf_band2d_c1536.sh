#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 0-12:00
#SBATCH -J rf_band2d_c1536
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_c1536_%j.out

# BAND-MODULATED *NONLINEAR* RF, B = 1, c = 1536, held out (michimin, 2026-09-23 08:53:
# "queue another experiment band B=1 c=1536 for the first 2 noises and 1.61 and 2.21").
#
# WHAT THIS ADDS OVER THE c=512 GRID.  c=512 B=1 already answered the absorption question
# (gain ~90% retained at low sigma, GROWING at sigma>=1.61).  What it did NOT answer is
# whether the band's high-sigma deficit closes with WIDTH.  At sigma=1.61/2.212 the band at
# c=512 is still +3.19/+3.54 over the held-out Wiener -- roughly half the plain arm's
# +5.67/+6.62 -- while the PLAIN arm is totally floored in c there (it moves 0.065 from
# c=512 to c=3072 at sigma=2.212).  If the band moves with c where the plain arm does not,
# that is the cleanest evidence in the project that the binding constraint at high sigma is
# the MODEL CLASS and that the band is the right direction to enlarge it.
#
# *** WHY kempner_h200 AND NOT kempner_h100. ***  The Delta-resolved Stein lag tensor Bc is
#   nD * c^2 * 3*(2t-1)^2 * 16 B  =  13 * 1536^2 * 75 * 16  =  36.8 GB  PER SPLIT,
# and on the held-out path B_tr and B_te are LIVE SIMULTANEOUSLY (core/rf_circulant2d_band.py
# builds the test moments while the train moments are still resident).  Add the ~19 GB
# Tre/Tim transient that exists while the second Bc is being assembled and the peak is about
# *93 GB* before P/q are allocated at all -- over the 80 GB H100 we have used for every band
# cell so far.  The H200 is 141 GB.  This was computed analytically and then confirmed by a
# direct two-point probe at the real c (job 47946185) rather than extrapolated from c=512.
#
# BUDGET=125e9 leaves ~88 GB after the 2*36.8 GB Bc term, which `sizing()` spends on P for
# both splits: K = c*nG = 1536*9 = 13824, so each P is nf * K^2 * 16 B = nf * 3.06 GB.
#
# SIGMA ORDER: the two high sigma FIRST.  This inverts the c=512 ordering deliberately.  At
# c=512 I ordered low-sigma-first because absorption was the open question and sigma=0.127 was
# where it would show; it did not show, and the interesting cells turned out to be the high
# ones.  The open question at c=1536 is the high-sigma deficit, so 1.61 and 2.212 run first
# and a wall-clock death still leaves the answer.  Checkpointed per (sigma, c, B).
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv

export OUT=tables/rf_pixel_band2d_heldout.npz
export SIGS=1.610,2.212,0.127,0.452
export CS=1536
export BS=1
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=125e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
