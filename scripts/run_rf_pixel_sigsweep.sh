#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00
#SBATCH -J sigsweep
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/sigsweep_%j.out

# NOISE SWEEP at matched free parameters, k/d in {0.5, 1, 2} only (michimin 2026-09-16):
# below k/d=0.5 the comparison is unfair to dense -- the c=2..256 points in
# tables/rf_pixel_parammatch.npz show dense still at 45-190 while circ is already on its
# floor, so those rows say nothing about the model classes.
#
# SIGS order is deliberate: sigma=1.610 first with JS=2 first, so the explicitly requested
# c=6144 circulant lands in the first ~43 min.  Then sigma=0.452 (only k/d=2 is missing;
# the other two cells are already in the npz and get skipped).  Then the new noises.
# sigma=5.0 is the cap michimin asked for; the next point on the project's 16-point
# logspace grid is 5.736, which is why 5.0 itself is slightly off-grid.
#
# NSEED_CIRC=1: circ cells cost 250-2561 s each and their seed spread is +-0.0001 at
# c=6144, so a second draw buys nothing.  Dense keeps 2 seeds -- it costs 1-5 s.
# Budget at NSEED_CIRC=1: ~3471 s per full sigma row, ~6.5 h total.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
# comma-valued vars MUST be set here, not via `sbatch --export` (sbatch splits on commas)
export JS="${JS:-2,1,0.5}"
export SIGS="${SIGS:-1.610,0.452,2.459,3.756,5.0,1.054,0.69}"
export NSEED="${NSEED:-2}"
export NSEED_CIRC="${NSEED_CIRC:-1}"
echo "JS=${JS}  SIGS=${SIGS}  NSEED=${NSEED}  NSEED_CIRC=${NSEED_CIRC}"
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_featmatch2.py
