#!/bin/bash
# WIDEN the local-softmax sigma range (michimin 2026-09-28: "更大的range ... when UNet and Linear
# start to converge to each other again, and with global softmax train = eval").
#
# New sigma (pixel units) on top of the project grid 0.127 .. 5.0:
#   low  0.005 0.01 0.02 0.05      (EDM valid down to sigma_edm = 0.002)
#   high 10 20 40                  (sigma_edm = 80 is EDM's sigma_max)
#
# WIDER HYPER-PARAMETER GRIDS.  At sigma = 0.005 the Wiener scores ~0.07 per image while the
# best LS (k=3) / Lukoianov (tau=0.2) scored ~6: both were stuck at their smallest window.  So
# every sigma also gets LS k=1 and Lukoianov tau in {0.3, 0.5, 1.0} (tau=1 keeps only
# coordinates whose normalised Wiener weight equals the global max, i.e. ~the diagonal), and the
# (dropped) ELS runs included k=1.  The plot only needs the BASE grid present to draw a sigma;
# extras enter the val selection wherever they exist.
#
#   0. (already running when this was written) cheap arms on the new sigma, base grids,
#      python pid CHEAP_PID -> tables/local_softmax_heldout.npz
#   1. cheap extras (LS k=1, tau 0.3/0.5/1.0) on ALL 15 sigma -> same npz, after CHEAP_PID
#   2. global softmax, held out AND train = eval, via scripts/bayes_oracle_heldout.py.  That
#      script rewrites its OUT from scratch, so it writes a SEPARATE file; the plot reads both.
#   ELS DROPPED (michimin 2026-09-28 21:50: "stop doing ELS, just do LS and Lukoianov").  The
#   8-sigma ELS job was killed after sigma = 0.127 and 0.452; those two cells stay in
#   tables/local_softmax_heldout_els.npz.  ELS costs ~24 min per sigma (10^10 candidate
#   patches per image), ~50x Lukoianov per image per hyper-parameter.
#
#   ssh <node> "cd <repo> && CHEAP_PID=<pid> setsid nohup bash \
#       scripts/run_local_softmax_wide.sh > logs/local_softmax_wide2.log 2>&1 < /dev/null &"
set -e
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONNOUSERSITE=1
LOW=0.005,0.01,0.02,0.05
HIGH=10.0,20.0,40.0
GRID=0.127,0.452,0.621,0.853,1.172,1.61,2.212,5.0

waitpid() { while [ -n "$1" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done; }

echo "[$(date +%T)] 1. waiting for cheap arms pid $CHEAP_PID, then extras on all sigma"
waitpid "$CHEAP_PID"
ARMS=ls,luk KS_LS=1 TAUS=0.3,0.5,1.0 NTEST=1000 NVAL=200 SIGS=$LOW,$GRID,$HIGH \
    OUT=tables/local_softmax_heldout.npz $PY -u scripts/local_softmax_heldout.py

echo "[$(date +%T)] 2. global oracle, held out + train = eval"
SIGS=$LOW,$HIGH OUT=tables/bayes_oracle_heldout_wide.npz $PY -u scripts/bayes_oracle_heldout.py

echo "[$(date +%T)] done"
