#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 0-02:00
#SBATCH -J probe_band_c1536
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/probe_band_c1536_%j.out

# TIMING + FEASIBILITY PROBE for band B=1 at c=1536 (michimin asked for the ETA before I
# queue the real run).  Measures, it does not extrapolate.
#
# WHY A PROBE AND NOT ARITHMETIC.  The c=96 -> c=512 pair gives 29 s -> 362 s, which is
# c^1.5 -- but those two cells ran at different freq_chunk, so the exponent is confounded
# and the cost is a SUM of terms with different c-scalings (the per-frequency solve is c^3,
# the moment accumulation c^2, pass 1 is c^2 and linear in N).  Extrapolating one power law
# through it would be guessing.
#
# DESIGN: two N points at the REAL c=1536, so time(N) = fixed + slope*N separates the
# N-independent solve from the N-linear accumulation and predicts N=10000 by interpolation
# rather than by a fitted exponent.  It also measures PEAK MEMORY at the real c, which is
# the actual open question:
#   Bc = nD * c^2 * 3nL * 16 B = 13 * 1536^2 * 75 * 16 = 36.8 GB PER SPLIT, both live at
#   once = 73.6 GB, plus ~19.1 GB of transient Tre/Tim while the second split's pass 1
#   runs => ~93 GB peak.  That is over the 80 GB H100 and is why this is on kempner_h200
#   (141 GB).  If the probe OOMs, c=1536 needs the Delta loop restructured, not a bigger
#   budget.
#
# Separate OUT files: a reduced-N cell must never land in the real table.
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv

export SIGS=0.127
export CS=1536
export BS=1
export T2=3
export NSEED=1
export BUDGET=125e9

for NP in 2500 5000; do
  echo "=========== PROBE N=$NP ==========="
  export NIMG=$NP
  export NTEST=$NP
  export OUT=tables/_probe_band_c1536_N$NP.npz
  rm -f $OUT
  /usr/bin/time -v /n/home12/binxuwang/.conda/envs/torch2/bin/python \
      scripts/rf_pixel_band2d_heldout.py 2>&1 | grep -vE "^\s*(Command|User|System|Percent|Average|Major|Minor|Voluntary|Involuntary|Swaps|File|Socket|Signals|Page size|Exit)"
done

echo "PROBE DONE"
