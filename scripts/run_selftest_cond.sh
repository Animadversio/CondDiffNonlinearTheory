#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-01:00
#SBATCH -J selftest_cond
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/selftest_cond_%j.out
#
# Validation for the label-conditioned 2-D circulant RF (michimin, 2026-09-25).  WRITTEN, NOT
# YET RUN.  Exits NONZERO on any failure, so production cells can hang off it with
# --dependency=afterok:<jobid>.
#
#   selftest2d          the pre-existing unconditional suite of core/rf_circulant2d.py.  The
#                       edit touched that file, so it must still PASS; its printed numbers
#                       should match any earlier log digit for digit (lab=None never enters
#                       the new code).
#   selftest2d_cond     conditional plain estimator vs an explicit-bias brute force, plus the
#                       gam=0 / one-class / lam=0 identities.
#   selftest_band_cond  the same for the band estimator (B = 1, 2) and B=0 vs plain.
# All cases are toy-sized; the whole job should take minutes on any GPU.
#
# FULL=1 also re-runs core/rf_circulant2d_band.py::selftest_band, the unconditional band
# suite, whose B=4 case needs ~106 GiB => submit with -p kempner_h200 in that case.  Its
# printed numbers should match logs/selftest_band_48364740.out digit for digit.

set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -c "
import os, sys, torch
from core.rf_circulant2d import selftest2d, selftest2d_cond
from core.rf_circulant2d_band import selftest_band_cond, selftest_band
res = {}
print('--- selftest2d (unconditional, pre-existing)', flush=True)
res['selftest2d'] = selftest2d()
print('--- selftest2d_cond', flush=True)
res['selftest2d_cond'] = selftest2d_cond()
print('--- selftest_band_cond', flush=True)
res['selftest_band_cond'] = selftest_band_cond()
if os.environ.get('FULL') == '1':
    print('--- selftest_band (unconditional, pre-existing; needs the H200)', flush=True)
    res['selftest_band'] = selftest_band()
for k, v in res.items():
    print(f'{k}: {\"PASS\" if v else \"FAIL\"}')
print('peak GiB %.2f' % (torch.cuda.max_memory_allocated() / 2**30))
sys.exit(0 if all(res.values()) else 1)
"

echo "=== DONE ==="
