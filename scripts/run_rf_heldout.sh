#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-12:00
#SBATCH -J rf_heldout
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/heldout_%j.out

# DOES THE Z_32 x Z_32 WIN SURVIVE A HELD-OUT SPLIT?
#
# Solve w_f = (P_f^train + lam I)^{-1} q_f^train, then score it against moments built from the
# 10,000 CIFAR-10 TEST images.  Both arms, same c grid as tables/rf_pixel_circ_csweep.npz and
# tables/rf_pixel_circ2d.npz, so every cell can be differenced against its in-sample twin.
#
# The linear baseline is held out TOO (Sigma estimated on train, scored on test).  That is not
# a detail: a d=3072 covariance carries 4.7M free parameters, which is more than the 2-D RF's
# 4.7M at c=1536 and fitted from the same 10^4 images, so the Wiener line moves under a split
# as well.  The c=32 smoke test already measured it moving by +0.459 at sigma=0.127 while the
# RF's own train->test gap was +0.024 -- i.e. comparing a held-out RF against an in-sample
# Wiener would invent a gap ~19x larger than the one that is really there, with the sign that
# flatters the baseline.  Every comparison in the driver is train-vs-train or test-vs-test.
#
# The train column must reproduce the stored in-sample tables cell for cell (same images, same
# seed bases 800/900, same lam).  The driver ASSERTS this, so a refactor that quietly moved
# the estimator stops the run instead of producing a plausible wrong table.
#
# c ASCENDING, checkpointed per (sigma, c, arm): if the allocation dies only the in-flight
# cell is lost.  1-D stops at c=1536 (its curve is flat in c from c~32, so larger c buys
# nothing); 2-D goes on to 3072 alone, where the low-sigma curve is still falling.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export`.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_heldout.npz
export SIGS=0.127,0.452,1.610,5.0
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
