#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 0-02:00
#SBATCH -J selftest_band
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/selftest_band_%j.out
#
# Run core/rf_circulant2d_band.py::selftest_band (the brute-force validation suite) on an
# H200, and EXIT NONZERO IF IT FAILS.
#
# *** WHY THIS SCRIPT EXISTS AT ALL: THE SUITE NO LONGER FITS ON AN H100, MEASURED. ***
# The B=4 case added for the 7x7 c=128 B=4 production run (michimin, 2026-09-24 23:20) OOMs
# on the 80 GB card, and it misses by about ONE GiB:
#
#   core/rf_circulant2d_band.py:495, A = torch.einsum('pri,ij,qrj->pq', E, Sig, E)
#   torch.OutOfMemoryError: Tried to allocate 25.98 GiB.  GPU 0 has a total capacity of
#   79.17 GiB ... this process has 54.20 GiB memory in use.        (logs/selftest_band_B4.log)
#
# => 54.20 + 25.98 = ~80.2 GiB needed vs 79.17 available.  THAT IS A MEASUREMENT, NOT AN
# ESTIMATE.  The H200's 140.4 GiB clears it with real margin (~57%).
#
# *** THE REFERENCE IS NOT RESTRUCTURED TO MAKE IT FIT. ***  Chunking that einsum over `p`
# would halve the peak, but it would mean editing the very object every production number in
# this project is validated against, on the same day it is used to clear a new band order.
# Moving the job to a bigger card costs nothing and leaves the reference untouched.
#
# *** AND THE WRAPPER IS NOT COSMETIC: `python -m core.rf_circulant2d_band` EXITS 0 EVEN ON
# FAIL. ***  `selftest_band()` RETURNS a bool and the module only prints PASS/FAIL, so a
# `--dependency=afterok` chain hung off the bare module would happily launch a production
# cell on top of a FAILED validation.  The -c wrapper below turns the bool into an exit code
# (and prints the measured peak, so the next person sizing this suite does not have to guess).
#
# Cost: the eleven pre-existing cases run in ~1 min; the B=4 case is the expensive one.
# Budgeted 2 h, expected well under.

set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -c "
import torch, sys
from core.rf_circulant2d_band import selftest_band
ok = selftest_band()
print('selftest_band:', 'PASS' if ok else 'FAIL')
print('peak GiB %.2f' % (torch.cuda.max_memory_allocated() / 2**30))
sys.exit(0 if ok else 1)
"

echo "=== DONE ==="
