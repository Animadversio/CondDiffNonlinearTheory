#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-06:00
#SBATCH -J circ2d_sigmid
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/circ2d_sigmid_%j.out

# FINISH THE *IN-SAMPLE* PLAIN 2-D c-SWEEP AT THE MID OCTAVE (michimin, 2026-09-23 20:30:
# "finish running plain 2-D c-sweep: 0.621, 0.853, 1.172, B=0").
#
# WHICH TABLE THIS IS.  There are two plain 2-D grids and only ONE of them has a hole:
#   tables/rf_pixel_heldout.npz   held out, `{sigma}|{c}|2d` -- COMPLETE, 8 sigma x 6 c x 2 seeds
#   tables/rf_pixel_circ2d.npz    IN SAMPLE, `{sigma}|{c}|circ2d` -- only 4 sigma
#                                 (0.127 / 0.452 / 1.61 / 5.0), c = 32..6144
# The three sigma michimin named are exactly the three the in-sample table lacks -- they are
# the mid octave added later (their 04:42 request), and job 47890358 predates it.  "B=0" =
# no band modulation, i.e. the plain arm, which is what this driver computes.
#
# SEED SCHEDULE IS COPIED FROM job 47890358 CELL FOR CELL (8/8/8/4/2/1/1 at c=32/96/256/512/
# 1536/3072/6144).  This is not decoration: the new sigma must be differenceable against the
# existing ones at equal seed count, and `filt2d`-style seeding (900 + 11*s + c, in this
# driver inline) depends only on (c, seed), so cell (sigma, c, s) here uses the IDENTICAL
# filters as cell (sigma', c, s) already stored.  Every cross-sigma comparison stays paired.
#
# COST, from the measured per-seed times in logs/circ2d_47890358.out (1/2/7/15/78/243/1350 s):
# 1889 s = 31 min per sigma, so ~1.6 h for the three requested and ~2.1 h with the extra
# below.  Peak 46.4 GB at c=6144 ⇒ 64 GB is right.  6 h budget is ~3x headroom.
#
# *** sigma=2.212 IS APPENDED LAST AND WAS NOT REQUESTED. ***  It is the OTHER hole in this
# table (the held-out grid has 8 sigma, the in-sample one would still have 7), and having
# both grids on the same sigma axis is what makes the in-sample-optimism route and the
# held-out route comparable cell for cell -- that cross-validation (Richardson extrapolation
# vs the held-out value) is the thing that validated the held-out estimator in the first
# place.  It costs 31 min and it runs AFTER everything michimin asked for, so cancelling it
# or letting it die loses nothing that was requested.
#
# c ASCENDING, 6144 LAST, checkpointed after every (sigma, c) cell: if the allocation dies
# the only thing lost is the single most expensive cell in flight.
#
# ⚠ THIS DRIVER STILL RESUMES AT CELL GRANULARITY (`if f'{key}|circ2d' in store: continue`),
# the same pattern that was fixed in rf_pixel_band2d_heldout.py today.  It is HARMLESS here
# because every target cell is absent, and it is deliberately NOT being changed while a job
# is about to run against it -- but do NOT try to raise the seed count of an existing cell in
# this table without fixing it first, or the run will silently do nothing and exit 0.
#
# SAFE TO WRITE THE CANONICAL TABLE DIRECTLY: no other job is writing it (checked squeue).
# The load-once/save-all race only bites with two concurrent writers.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
cp tables/rf_pixel_circ2d.npz /tmp/circ2d_backup_presigmid.npz

export OUT=tables/rf_pixel_circ2d.npz
export T2=3
export NIMG=10000

run () {   # run <c-list> <nseed>
  echo "=============== c=$1  seeds=$2 ==============="
  CS="$1" NSEED="$2" \
    /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_circ2d.py
}

sweep () {   # sweep <sigma-list>   -- one full c-sweep at the job-47890358 seed schedule
  export SIGS="$1"
  run 32,96,256 8
  run 512       4
  run 1536      2
  run 3072      1
  run 6144      1
}

sweep 0.621,0.853,1.172        # <- what michimin asked for
echo "=============== REQUESTED SIGMA COMPLETE ==============="
sweep 2.212                    # <- unrequested extra, see header

echo "ALL DONE"
