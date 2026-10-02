#!/bin/bash
# 64 px repeat of the local-softmax held-out comparison (michimin 2026-10-01).
# Pool = images [0:10000] for Wiener / LS / Lukoianov / global softmax; 500 test, 100 val (the
# dataset's last 100); 19 sigma 0.0002 .. 40.  Data: scripts/prep_64px.py (EDM zips, uint8).
#   ffhq64: test [60000:60500], unseen by every 10k / 30k U-net (50k- and 250k-step twins);
#           full_longtrain (all 70k, ckpt ~221k steps) and the official EDM saw it.
#           + variant B (lukB masks, wienerB) from the covariance of [0:60000] (full rank).
#   afhq64: test [10000:10500]; only full-data U-nets exist (DLC longtrain ckpt, EDM afhqv2);
#           variant B from [0:10000]+[10500:15703] (15.2k images > d, full rank).
# Pilot (FFHQ64, sigma=0.127, 50 imgs): Wiener 31.0, Luk A 30.8, Luk B 30.5; U-net 30k 14.5
# (50k steps) vs 23.1 (250k steps) -- longer training on 30k images generalises WORSE.
#   ssh <node> "cd <repo> && setsid nohup bash scripts/run_local_softmax_64.sh ffhq64 \
#       > logs/local_softmax_ffhq64.log 2>&1 < /dev/null &"
# 2026-10-01 17:20: a duplicated launch ran FFHQ64 twice on the GPU (3 processes, 3x slower,
# one AFHQ64 OOM).  Relaunched one process per dataset; sigma >= 0.02 then used the reduced
# grid TAUS=0.02,0.05,0.1,0.2,0.3 (every tau validation picked at 32 px in that range, bar
# noise-level picks at sigma >= 20).  sigma <= 0.01 keep the full 9-tau sweep (resumed).
set -e
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONNOUSERSITE=1
SIGS=0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.127,0.452,0.621,0.853,1.172,1.61,2.212,5.0,10.0,20.0,40.0
case "$1" in
  ffhq64) EXTRA="TEST_START=60000 LUKB_COV=0:60000 UNETS=10000_split1,10000_longtrain_split1,30000_split1,30000_longtrain_split1,full_longtrain,edm" ;;
  afhq64) EXTRA="TEST_START=10000 LUKB_COV=0:10000,10500:15703 UNETS=full_longtrain,edm" ;;
  *) echo "usage: $0 ffhq64|afhq64"; exit 1 ;;
esac
env $EXTRA DATASET=$1 ARMS=wiener,insample,ls,luk,edm NTEST=500 NVAL=100 SIGS=$SIGS \
  TAUS=${TAUS:-0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0} \
  $PY -u scripts/local_softmax_heldout.py
echo ALLDONE
