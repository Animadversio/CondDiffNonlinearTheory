#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 0-06:00
#SBATCH -J rf_band2d_cond
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_cond_%j.out
#
# LABEL-CONDITIONED band RF, held out -- scripts/rf_pixel_band2d_cond_heldout.py.
# WRITTEN, NOT YET RUN.  A TEMPLATE: edit the cell list below before submitting.
#
# Order that wastes the least GPU:
#   1. scripts/run_selftest_cond.sh      must PASS (minutes)
#   2. scripts/run_cond_toll2d.sh        linear answer first: if toll_S/toll_U ~ 1 at the
#                                        sigma you care about, the band RF has little extra
#                                        to find there (minutes)
#   3. this file, cheap cells first      (the defaults below: 3x3, c=96, B=0,1)
#   4. the 7x7 cells                     e.g. T2=7 CS=256 BS=0,3 SIGS=1.610 -- the unconditional
#                                        t7 table has c=256 B=0 AND B=3 there, so both pair.
#
# Defaults: MODE=vu (free V U + label shift in phi).  Also worth one cheap cell each:
#   MODE=vu0  (readout-only: what V U alone buys)   MODE=feat (phi-only, the dense-RF convention).
# Pairing: at T2=3, c=96 the unconditional tables hold B=0 at every sigma (plain 2-D table)
# and B=1 at sigma 0.621 and 1.61 (t3fill), so those cells print a paired cond - uncond.
#
# Sizing is the unconditional driver's sizing() (BUDGET below); class_centre adds ~2 GB.

set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=60e9

export MODE=vu
export GSCALE=1.0
export T2=3
export CS=96
export BS=0,1
export SIGS=0.621,0.853,1.172,1.610

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/rf_pixel_band2d_cond_heldout.py
echo "=== DONE ==="
