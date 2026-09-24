#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 1-00:00
#SBATCH -J rf_band2d_B3
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_c512_B3_%j.out

# BAND B=3, c=512, HELD OUT, sigma = 0.127 / 0.452 / 1.610 / 2.212
# (michimin, 2026-09-23 21:28 "queue a job running B=2 on c=512 at the first 2 noises and
# 1.61 and 2.21", corrected 21:30 "my bad i meant B=3").
#
# *** WHY h200 AND NOT h100 -- I HAD THIS WRONG WHEN I OFFERED THE RUN. ***  The standing
# offer said B=3 "fits the H100".  It does not.  That estimate quoted 62.8 GiB, which is a
# RESIDENT figure, as if it were a peak.  Measured against the B=2 run (38.7 GiB resident
# predicted, 51.4 GiB peak observed) the transient factor is 1.33x, so:
#
#     B    K=c(2B+1)^2   nD=((4B+1)^2+1)/2   Bc both splits   nf   P both   resident   ~peak
#     1        4,608            13                7.6 GiB     16   10.1        17.7     23.5
#     2       12,800            41               24.0 GiB      3   14.6        38.7     51.4  (measured)
#     3       25,088            85               49.8 GiB      1   18.8        68.6    ~91
#
# THE BINDER IS NOT THE SOLVE, IT IS THE DELTA-RESOLVED STEIN LAG TENSOR, which carries one
# entry per Delta = s_r' - s_r and therefore scales as (4B+1)^2 / 2 -- 85/41 = 2.07x from B=2
# to B=3, on top of being held at c^2.  49.8 GiB of that is irreducible at c=512, T2=3, B=3.
# ~91 GiB does not fit the 79 GiB H100 and does fit the 141 GB H200.
#
# nf=1 IS DELIBERATE AND IS THE CONSERVATIVE CHOICE.  Raising BUDGET to 142e9 would buy nf=2,
# halving the number of frequency passes (544 -> 272) and saving an estimated ~1.7 h overall,
# at a predicted peak of ~116 GiB = 82% of the card.  The 1.33x transient factor is calibrated
# on B=2, not on B=3, so an 18% margin is not enough to bet four cells and ~9 h of H200 on.
# BUDGET stays at the 70e9 both B=2 jobs used.
#
# COST, extrapolated from the two measured points (B=1 360 s/cell at nf=16 = 34 passes;
# B=2 1680 s/cell at nf=3 = 182 passes).  Splitting those into a moment part that tracks the
# PASS COUNT (the (c,N,H,W) feature tensor is rebuilt per frequency chunk and its cost is
# B-independent) and a solve part that tracks K^3 gives ~1060 s + ~620 s at B=2, hence at
# B=3 with 544 passes and a 7.5x solve: ~3180 s + ~4650 s = ~2.2 h/sigma, ~9 h for four.
# The 24 h budget is ~2.6x headroom; h200's limit is 2 days if it ever needs resubmitting.
#
# SIGMA ORDER, most-informative first, checkpointed after every cell so a wall-clock death
# costs exactly one cell:
#   1.610, 2.212  the two cells that test the PRE-REGISTERED prediction below, and where the
#                 open question (the high-sigma deficit) actually lives.
#   0.127         confirmation that the interior optimum in B keeps going the wrong way.
#   0.452         last -- mid transfer, no sharp prediction, least informative if lost.
#
# *** PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (the habit that paid off at B=2,
# where I called +2.2 / +2.5 at sigma=1.61 / 2.212 and got +2.260 / +2.528). ***  Linear
# held-out tolls by class are already in hand at all four sigma, so each cell is scoreable the
# moment it lands.  predicted = L(B=2) - (toll removed) x (the B=1->B=2 transfer at that sigma):
#
#   sigma   L(B=2)    toll B=2 -> B=3   removed   transfer   PREDICTED L(B=3)
#   0.127    7.3342   0.0972 -> 0.0407   0.0565    -118.6%      7.401   <- predicted WORSE
#   0.452   28.4825   1.0836 -> 0.8166   0.2670     +64.0%     28.312
#   1.610   74.7207   2.6928 -> 2.0664   0.6264    +102.2%     74.081
#   2.212   89.8079   2.8435 -> 2.1670   0.6765    +102.4%     89.115
#
# READING FIXED IN ADVANCE.  If 1.610/2.212 land near 74.08/89.12 the ~102% transfer plateau
# extends to a 49x linear class and the band's high-sigma behaviour is fully described by the
# linear toll ladder.  If sigma=0.127 lands ABOVE 7.3342 the interior optimum is confirmed on
# a third value of B.  ⚠ EITHER WAY THIS DOES NOT CLOSE THE DEFICIT: 74.08 is still +1.62 over
# the held-out Wiener's 72.4610, and even B=4's ENTIRE remaining toll would leave ~+1.2.
# The band is a halving at high sigma, not a fix.  Say that when reporting it.
#
# VALIDATION DONE BEFORE SUBMITTING: B=3 had never been exercised by the estimator's brute
# force (the stored cases covered B=1 and B=2 only).  Two B=3 cases added permanently to
# core/rf_circulant2d_band.py::selftest_band and both pass against the explicit real
# constrained least squares -- 7x7 Cin=1 c=2 at 3.4e-14 and 7x7 Cin=2 c=1 at 2.0e-15, in
# sample and held out.  ⚠ The B=3 cases run at 7x7, NOT at the 4x4/5x5 used for B=1/B=2:
# box(B) offsets are taken mod (H,Wd), so on a grid smaller than 2B+1 two distinct s_r alias
# onto the same modulation and the design silently goes rank-deficient.
#
# SAFE TO WRITE THE CANONICAL TABLE DIRECTLY: squeue shows no other job writing it, so the
# load-once/save-all race that forced the two B=2 jobs onto separate files does not apply.
# Backup at /tmp/band2d_backup_preB3.npz.  The driver ASSERTS the recomputed `linear|{sigma}`
# rows reproduce the stored ones rather than overwriting them.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
nvidia-smi --query-gpu=memory.total --format=csv
cp tables/rf_pixel_band2d_heldout.npz /tmp/band2d_backup_preB3.npz

export OUT=tables/rf_pixel_band2d_heldout.npz
export SIGS=1.610,2.212,0.127,0.452
export CS=512
export BS=3
export T2=3
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
