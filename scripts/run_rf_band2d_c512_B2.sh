#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-12:00
#SBATCH -J rf_band2d_c512_B2
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_c512_B2_%j.out

# BAND-MODULATED *NONLINEAR* RF, B = 2, c = 512, held out (michimin, 2026-09-23 09:14:
# "hmm okay if it's K^3 then hold off on task 2. what about running band B=2?").
#
# *** WHY B=2 AT c=512 RATHER THAN B=1 AT c=1536. ***  K = c*(2B+1)^2, so B and c push into
# the SAME K^3 solve, and B=2 lands just BELOW the c=1536 cell michimin just cancelled:
#
#     config        K      params    peak     solve    moments
#     c=512  B=1   4608   14.2M      9.6 GB    1.0x      1.0x
#     c=512  B=2  12800   39.3M     30.3 GB   21.4x      3.2x
#     c=1536 B=1  13824   42.5M     86.4 GB   27.0x      9.0x
#
# So B=2 is ~79% of the cancelled job's solve cost, 1/3 of its memory, and -- decisively --
# it FITS ON THE H100.  The c=1536 probe (job 47946185) measured a 97.8 GB peak, genuinely
# over the 80 GB card, which is why that one needed the H200.  Params are near-matched
# (39.3M vs 42.5M), so this probes "a LARGER LINEAR CLASS" at essentially the same capacity
# rather than confounding class size with width.
#
# WHAT IT DECIDES.  The held-out LINEAR tolls say B=1 -> B=2 removes the most EXTRA toll at
# exactly the high sigma where the band is the only class that moves anything:
#     sigma      0.127   0.452   0.621   1.610   2.212
#     B=1        0.1998  1.5228  2.0768  3.6046  3.8333
#     B=2        0.0972  1.0836  1.5155  2.6928  2.8435
#     extra      +0.103  +0.439  +0.561  +0.912  +0.990
# and the c=512 B=1 run showed the nonlinear gain is NOT absorbed by a 9x larger linear class
# -- retention 88% / 102% / 111% / 118% as sigma rises.  If that holds through a 25x class,
# B=2 should land near +2.2 at sigma=1.61 and +2.5 at 2.212, vs B=1's +3.19/+3.54.
#
# HELD OUT, NOT OPTIONAL: 39,321,600 trained params against 10,000 images is 4.2x the free
# Wiener's 9,437,184.  An in-sample number here would be uninterpretable.
#
# NO NEW VALIDATION NEEDED: core/rf_circulant2d_band.py's brute-force reference was asserted
# on 5 configs covering B=1 AND B=2 (1e-16 ... 6e-15), in sample and held out.
#
# SIGMA ORDER: high first.  At c=512 B=1 I ordered low-first because absorption was the open
# question and sigma=0.127 was where it would show; it did not show, and the interesting
# cells turned out to be the high ones.  The open question now is the high-sigma deficit.
#
# SIZING: fixed (Bc, both splits) = 25.8 GB; BUDGET=70e9 leaves ~44 GB, giving nf=3 and
# P for both splits = 2*3*K^2*16 = 15.7 GB => ~42 GB resident, well inside the 80 GB card.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_band2d_heldout.npz
export SIGS=1.610,2.212,0.127,0.452
export CS=512
export BS=2
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
