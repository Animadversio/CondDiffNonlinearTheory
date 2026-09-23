#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-00:30
#SBATCH -J band_lin_fill
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band_lin_fill_%j.out

# HELD-OUT *LINEAR* BAND/COMB TOLLS at the three sigma the stored grid lacks.
#
# WHY: the nonlinear band cells at sigma = 0.621 / 0.853 / 1.172 (job 47946168) are useless
# on their own.  The headline quantity is the NONLINEAR GAIN
#     gain(B) = L[best LINEAR map in the band-B class] - L[the relu RF in that same class],
# so each nonlinear cell needs its linear partner at the SAME sigma.  The stored grid is
# [0.127 0.452 1.61 5.0] plus a separate [2.212] file -- the three new sigma are absent.
# ~1 min on GPU, closed form, no sampling.
#
# Writes a SEPARATE file: scripts/rf_band_relaxation_heldout.py rebuilds whatever sigma grid
# it is given, and pointing it at the main table would rewrite that table on the new 3-point
# grid, destroying the 4 stored sigma.  scripts/rf_band2d_report.py already merges any number
# of these files by matching on the physical sigma VALUE, so a separate file costs nothing.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

export SIGS=0.621,0.853,1.172
export OUT=tables/rf_band_relaxation_heldout_sigmid.npz

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_band_relaxation_heldout.py

echo "ALL DONE"
