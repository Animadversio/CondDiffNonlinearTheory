#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -J circ2d_N
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/circ2d_nsweep_%j.out

# IS THE Z_32 x Z_32 WIN OVER LINEAR A PROPERTY OF THE MODEL CLASS, OR OF N = 10^4?
#
# The c-sweep (job 47890358) puts the re-indexed nonlinear RF BELOW the Wiener optimum at
# sigma = 0.127 (-1.040 at c = 1536) and nearly at it at sigma = 0.452 (+0.270).  That is
# the first outright win over linear this project has measured on raw pixels, so it has to
# survive the check that killed the last one.
#
# On 2026-09-21 the dense RF's advantage over linear at FIXED k collapsed as N grew
# (k/d = 4, sigma = 0.127: 0.825 -> 0.336 -> 0.039 at N = 10k/20k/40k), and what had looked
# like a property of the model class was memorisation: 24,576 features against 10^4 atoms,
# with lam = 1e-6 not stopping it.  The 2-D model here has 4.7M trained parameters and
# 1.57M features against the same 10^4 atoms, and -- unlike the 1-D circulant, which is flat
# in c at low sigma -- its loss is still falling steeply in c.  A steadily falling curve at
# fixed N is precisely the signature that turned out to be overfitting last time.
#
# The 1-D control matters as much as the 2-D arm: the 1-D circulant's flatness has been
# quoted as evidence that its function class is too small to overfit, but that was never
# actually measured against N.  If BOTH arms are N-flat the win is real; if only the 2-D arm
# moves, the extra freedom (3-channel mixing + 27 taps) is being spent on memorisation.
#
# Wiener is recomputed on the SAME N images inside each driver, so every "excess over
# Wiener" below is self-consistent -- no cross-N baseline mixing.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export`.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L

SG=0.127,0.452

for NI in 10000 20000 40000; do
  echo "############### N=${NI}  2-D Z_32xZ_32 ###############"
  OUT=tables/rf_circ2d_nsweep_${NI}.npz SIGS=$SG T2=3 NIMG=$NI CS=512,1536 NSEED=2 \
    /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_circ2d.py
done

for NI in 10000 20000 40000; do
  echo "############### N=${NI}  1-D Z_3072 control ###############"
  OUT=tables/rf_circ1d_nsweep_${NI}.npz SIGS=$SG T_BAND=8 NIMG=$NI JS=0.5 \
    NSEED=1 NSEED_CIRC=1 \
    /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/rf_pixel_featmatch2.py
done

echo "ALL DONE"
