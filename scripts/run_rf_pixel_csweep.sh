#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-08:00
#SBATCH -J circ_csweep
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/circ_csweep_%j.out

# WHERE DOES THE BLOCK-CIRCULANT RF ACTUALLY FLOOR IN c?
#
# docs/rf_circulant_relaxations.md sec.1 claims "circ floors by c ~ 8-32".  That claim is
# what makes the band readout relaxation look affordable: the band multiplies the
# per-frequency block by |G| = (2B+1)^2, so it is only runnable at a c where the plain
# circulant has already saturated.  If the floor is really at c ~ 32 the band is cheap; if
# the model is still moving at c = 1536 it is not.
#
# This sweeps c = 32, 96, 256, 512, 1536 through the SAME code path (circulant_rf_mmse_lag2)
# and the SAME table, with the seed budget michimin specified: 8 seeds at c <= 256, 4 at
# c = 512, 2 at c = 1536.  Separate table so nothing in tables/rf_pixel_featmatch2.npz moves.
#
# c ascending on the outside so every cheap cell is on disk before the expensive ones start;
# the driver checkpoints after each (sigma, c) and skips what is already present, so a kill
# costs at most one cell.  Measured on an H100: 8 s/seed at c=32, 30 s at 256, 61 s at 512.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` -- sbatch splits the
# value on commas and the driver then sees a truncated list.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export OUT=tables/rf_pixel_circ_csweep.npz
export SIGS=0.127,0.452,1.610,5.0
export T_BAND=8
export NIMG=10000

# j = c/3072, written to full precision so int(round(j*d)) lands exactly on c.
run () {   # run <j> <c> <nseed>
  echo "=============== c=$2  seeds=$3 ==============="
  JS="$1" NSEED="$3" NSEED_CIRC="$3" \
    /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_featmatch2.py
}

run 0.010416666666666666   32 8
run 0.03125                96 8
run 0.08333333333333333   256 8
run 0.16666666666666666   512 4
run 0.5                  1536 2

echo "ALL DONE"
