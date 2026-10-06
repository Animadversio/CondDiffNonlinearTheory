#!/bin/bash
# Heun-30 sampling at 64 px, 256 paired seeds (michimin 2026-10-01): Wiener (pool and full-rank
# B), LS, Lukoianov, global softmax vs U-nets -- FFHQ64: EDM, 30000_split1, 10000_split1;
# AFHQ64: EDM, full_longtrain (both trained on all of AFHQ).  Cache in STORE_DIR.
#   ssh <node> "cd <repo> && setsid nohup bash scripts/run_local_softmax_sampling_64.sh \
#       > logs/local_softmax_sampling_64.log 2>&1 < /dev/null &"
set -e
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONNOUSERSITE=1
for ds in ffhq64 afhq64; do
  echo "[$(date +%T)] $ds"
  DATASET=$ds NSAMP=256 $PY -u scripts/local_softmax_sampling.py
done
echo ALLDONE
