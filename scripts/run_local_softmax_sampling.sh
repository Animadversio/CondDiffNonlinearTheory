#!/bin/bash
# Heun-30 sampling with Wiener / LS / Lukoianov / global softmax vs U-nets, 512 paired seeds,
# CIFAR-10, FFHQ32, AFHQ32 (michimin 2026-09-29).  Cache in STORE_DIR, tables in the repo.
#   ssh <node> "cd <repo> && setsid nohup bash scripts/run_local_softmax_sampling.sh \
#       > logs/local_softmax_sampling.log 2>&1 < /dev/null &"
set -e
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONNOUSERSITE=1
for ds in cifar10 ffhq32 afhq32; do
  echo "[$(date +%T)] $ds"
  DATASET=$ds NSAMP=512 $PY -u scripts/local_softmax_sampling.py
done
echo ALLDONE
