#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00
#SBATCH -J circ2d
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/circ2d_%j.out

# IS Z_32 x Z_32 THE RIGHT GROUP FOR THE NONLINEAR RF?
#
# Same estimator, same data, same trained-parameter count (3072c), same free per-position
# bias; the ONLY change is the group the design is equivariant under.  The linear shadow of
# this re-index is already measured (tables/rf_equivariance_toll.npz): the toll falls
# +1.273 -> +0.889, +4.070 -> +3.082, +8.473 -> +6.469, +9.005 -> +7.782.  This run asks how
# much of that the nonlinear model collects.
#
# michimin asked for c = 256, 512, 1536.  The grid below ALSO covers c = 32, 96 (so the
# curve lands on exactly the c-axis of tables/rf_pixel_circ_csweep.npz and the two models
# can be differenced cell for cell) and extends to 3072 / 6144.  That extension is not
# optional decoration: the 2026-09-22 c-sweep showed the 1-D circulant is still moving
# at c = 6144 at sigma = 0.127 (the "+0.129" margin extrapolates to +0.071), so a
# three-point low-sigma curve cannot be read as a floor.  Measured cost is 2 s/seed at
# c=32, 15 s at c=512, so everything through c=1536 is minutes.
#
# c ASCENDING and 6144 LAST: the driver checkpoints after every (sigma, c) cell and skips
# what is already on disk, so if the allocation dies the only thing lost is the most
# expensive cell, and every cheap cell is already banked.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` -- sbatch splits the
# value on commas and the driver then sees a truncated list.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_circ2d.npz
export SIGS=0.127,0.452,1.610,5.0
export T2=3
export NIMG=10000

run () {   # run <c-list> <nseed>
  echo "=============== c=$1  seeds=$2 ==============="
  CS="$1" NSEED="$2" \
    /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_circ2d.py
}

run 32,96,256 8
run 512       4
run 1536      2
run 3072      1
run 6144      1

echo "ALL DONE"
