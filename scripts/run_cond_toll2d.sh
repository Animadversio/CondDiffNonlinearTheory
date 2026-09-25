#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -J cond_toll2d
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/cond_toll2d_%j.out
#
# DOES CONDITIONING SHRINK THE 2-D EQUIVARIANCE TOLL?  (michimin, 2026-09-25)
# scripts/rf_cond_toll2d_heldout.py -- linear, closed form, held out.  WRITTEN, NOT YET RUN.
#
# Held-out band-B tolls for U (unconditional), S (class mean known, pooled within-class) and
# C (per class), B = 0..4, all eight sigmas.  Fits on train[:10000], scores on the 10,000 CIFAR
# test images with their own labels.  Its U column must reproduce the stored held-out ladder
# (asserted), so a wrong image/label pipeline stops the job instead of printing a table.
# Expected: a few minutes, ~6 GB.  Writes tables/rf_cond_toll2d_heldout.npz, which
# scripts/rf_pixel_band2d_cond_heldout.py reads for its band_S(B) baseline.

set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

export NIMG=10000
export NTEST=10000
export SIGS=0.127,0.452,0.621,0.853,1.172,1.610,2.212,5.0
export BS=0,1,2,3,4
export OUT=tables/rf_cond_toll2d_heldout.npz

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/rf_cond_toll2d_heldout.py
echo "=== DONE ==="
