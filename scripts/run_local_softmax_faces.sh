#!/bin/bash
# FFHQ32 / AFHQ32 repeat of the CIFAR local-softmax comparison (michimin 2026-09-28).
# Same contract as CIFAR: pool = [0:10000] for Wiener / LS / Lukoianov / global softmax,
# 1000 held-out test images, 200 val images (the dataset's last 200), 19 sigma 0.0002 .. 40.
# U-nets from DL_Projects/DiffusionSpectralLearningCurve (same SongUNet architecture):
#   FFHQ32 (70k):  10000_split1 [0:10k] and 30000_split1 [0:30k] never see test [60000:61000];
#                  full_longtrain = all 70k, 250k steps (test images SEEN; no held-out FFHQ exists)
#   AFHQ32 (15.8k): 10000_split1 never sees test [10000:11000]; full = all images, 50k steps.
# Pilot 2026-09-28 (FFHQ, sigma=0.127, per image): Wiener 9.97, 10k 21.5 (memorises),
#   30k 7.99, full 5.22, full_longtrain 4.13.
#   ssh <node> "cd <repo> && setsid nohup bash scripts/run_local_softmax_faces.sh ffhq32 \
#       > logs/local_softmax_ffhq32.log 2>&1 < /dev/null &"
set -e
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONNOUSERSITE=1
SIGS=0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.127,0.452,0.621,0.853,1.172,1.61,2.212,5.0,10.0,20.0,40.0
case "$1" in
  ffhq32) TS=60000; NETS=10000_split1,30000_split1,full_longtrain ;;
  afhq32) TS=10000; NETS=10000_split1,full ;;
  *) echo "usage: $0 ffhq32|afhq32"; exit 1 ;;
esac
DATASET=$1 TEST_START=$TS UNETS=$NETS ARMS=wiener,insample,ls,luk,edm NTEST=1000 NVAL=200 \
  SIGS=$SIGS KS_LS=1,3,5,7,9,11,13,15,19,23,31,63 \
  TAUS=0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0 \
  $PY -u scripts/local_softmax_heldout.py
echo ALLDONE
