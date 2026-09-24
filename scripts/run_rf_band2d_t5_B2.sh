#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 1-00:00
#SBATCH -J rf_band2d_t5B2
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_t5B2_%j.out
#
# =====================================================================================
# 5x5 TAPS AT B=2, c=512, HELD OUT -- michimin 2026-09-23 23:48:
#   "run B=2 at c=512 for kxk=5x5 on 4 noises, first 2 and 1.61 and 2.21"
#
# Third concurrent job.  48071356 (B=3 c=512, 3x3) and 48084263 (7x7 at B=1) are both
# live on their own h200s; this takes a third.  SEPARATE OUT npz -- see the race note.
#
# -------------------------------------------------------------------------------------
# WHAT THIS ADDS THAT THE 7x7 RUN DOES NOT
# -------------------------------------------------------------------------------------
# The 7x7 arm (48084263) answered "is 27 px enough?" at B=1 and sigma=0.127: NO, wider is
# WORSE, +0.32..+0.41 held out and +0.24..+0.30 IN SAMPLE, at c=32/96/256.  Three things
# were left open and this run moves all three:
#
#   (1) *** IS THE PENALTY MONOTONE IN WINDOW SIZE, OR IS 3x3 AT AN INTERIOR OPTIMUM? ***
#       We have exactly two tap sizes.  Two points cannot tell a monotone slope from a
#       minimum at one end.  5x5 (75 px) sits between 27 and 147 and decides it.
#   (2) *** DOES THE TAP-SIZE PENALTY SURVIVE AT B=2? ***  Every tap-size number so far is
#       at B=1.  B=2 is the cell where the band class TURNED AROUND at sigma=0.127 (paired
#       differential +0.0573, own gap +0.6462 vs the free Wiener's +0.4593 -- the most
#       overfit object in that comparison).  If a wider window inflates own gaps the way
#       7x7 did at B=1, B=2 is where it should show up worst.
#   (3) *** THE SIGMA TREND, WHICH IS ENTIRELY UNMEASURED. ***  All four requested sigma
#       have a stored 3x3 B=2 c=512 partner, so every cell here is scoreable on arrival.
#
# *** AND IT CLOSES THE GRID: 5x5 AT B=2 IS THE LAST FEASIBLE CELL IN THIS DIRECTION.
# 7x7 at B=2 needs ~224 GiB peak and fits NOTHING (h200 is 141 GB).  See the table below. ***
#
# -------------------------------------------------------------------------------------
# THE FRAMING (unchanged from the 7x7 run, and now partly FALSIFIED by it -- say both)
# -------------------------------------------------------------------------------------
# TAP SIZE MOVES ONLY THE *RANDOM* PARAMETERS.  Trained params = Cin*c*H*W*(2B+1)^2 -- no
# T2 in it.  c=512 B=2 is 39,321,600 trained params at 3x3 AND at 5x5.  Only the design
# draw changes: c*Cin*T2^2 = 27c -> 75c (2.78x).
# NORM-MATCHED: filt2d draws N(0, I/(Cin*T2^2)) => E||row||^2 = 1 at every tap size.  5x5
# is the SAME energy spread over 75 px, not more energy.
# ⚠ I ARGUED FROM THAT FRAMING THAT A WIDER WINDOW COULD NOT BUY MEMORISATION, AND THE 7x7
# RUN KILLED IT: own gaps went +0.0643->+0.0962, +0.1096->+0.1910, +0.1843->+0.3577 at
# c=32/96/256 = 1.50/1.74/1.94x, at an IDENTICAL trained-parameter count.  The
# generalisation gap is not determined by the trained count alone.  Carried here as a
# prediction to test, not as a premise.
#
# -------------------------------------------------------------------------------------
# *** PRE-REGISTERED, BEFORE ANY 5x5 NUMBER EXISTS (the habit that paid at B=2 and B=3) ***
# -------------------------------------------------------------------------------------
# (A) DIRECTION AND SIZE at sigma=0.127: 5x5 WORSE than 3x3 by *+0.13 to +0.20*, i.e. test
#     ~7.46-7.53 against the stored 3x3 B=2 c=512 value of *7.3300*.  Basis: the measured
#     7x7-vs-3x3 penalty at B=1 is +0.32..+0.41, and 5x5 is 40% of the way in pixels
#     (75 of 27->147) or 50% in T2.  A HALF-PENALTY IS THE MONOTONE PREDICTION.
# (B) THE FALSIFIER IS THE SHAPE, NOT THE SIGN:
#       * 5x5 ~ half the 7x7 penalty  => MONOTONE in window size.  3x3 is then on a slope,
#         not at an optimum, and the natural follow-up is 2x2 -- with the caveat that the
#         2026-07-25 1-D locality sweep found the optimum at w=2,3,4 and w=1 DEGENERATE
#         (Theta_a = g_a*I, no spatial mixing, span collapses to {y, |y|}), so the 2-D
#         analogue would be 2x2 near-optimal and 1x1 degenerate.
#       * 5x5 ~= 3x3 or better  => NOT monotone; there is an interior optimum between 3 and
#         7 and the 7x7 result is a far-field statement, not a local slope.  Different
#         picture, and it would need a finer sweep.
#       * 5x5 WORSE than 7x7 => implausible; would mean a maximum at 5 and I would re-check
#         the draw before believing it.
# (C) OWN GAP: if the 7x7 own-gap inflation is real and scales with window size, B=2's own
#     gap at sigma=0.127 should go from *+0.6462* to roughly *+0.9 to +1.2*.  This is the
#     sharpest available test of the point that killed the trained-vs-random argument.
# (D) THE SIGMA TREND I DO *NOT* PREDICT.  There is no tap-size measurement at any sigma
#     other than 0.127 yet.  48084263 will deliver the B=1 sigma-trend at 7x7 within ~2 h,
#     independently -- so the two runs cross-check rather than one assuming the other.
#
# -------------------------------------------------------------------------------------
# ⚠ UNPAIRED ACROSS TAP SIZES, PAIRED WITHIN ONE (same caveat as 7x7, state both halves)
# -------------------------------------------------------------------------------------
# filt2d draws randn(c, CIN, T2, T2); at a different T2 the same seed gives a DIFFERENT
# tensor and the draws are not even nested (the row stride changes).  So 5x5-vs-3x3 is a
# difference of two independent Theta draws.  Measured absolute Theta spread at c=512
# sigma=0.127 is 0.0076 (plain) / 0.0080 (band B=1) => anything above ~0.02 is readable at
# NSEED=1, and the predicted +0.13..+0.20 is 16-25x that.
# BS=0,2 => the B=0 arm gives the PLAIN 5x5 reference, and B=0 vs B=2 at the same T2 DO
# share filters, so the band-vs-plain differential inside 5x5 is properly PAIRED.  That is
# what produced the "the band is orthogonal to the window" finding at 7x7 and it costs
# ~2 GiB and a few minutes here (nD=1, K=512 => the Delta-tensor is 41x smaller).
#
# -------------------------------------------------------------------------------------
# SIZING -- the binder is the Delta-resolved Stein lag tensor, NOT the solve
# -------------------------------------------------------------------------------------
# fixed = 2*nD*c^2*3*(2*T2-1)^2*8*2 bytes, nD = ((4B+1)^2+1)/2.  (2*T2-1)^2 = 25 -> 81 is
# *3.24x* over 3x3, held at c^2.  K = c*(2B+1)^2 = 12800 is UNCHANGED by T2 -- the solve
# does not feel the tap size at all.
#     c    B  T2 |    K  nD  nf     ns | Bc GiB  P GiB  resid  ~peak   card
#   512    2   3 | 12800  41   3   4882 |   24.0   14.6   39.8   53.0   h100   (measured 51.4)
#   512    2   5 | 12800  41   1  10000 |   77.8    4.9   84.0  111.8   h200   <- THIS RUN
#   512    0   5 |   512   1  16  10000 |    1.9    0.1    5.6    7.5   h100
#   512    3   3 | 25088  85   1   7473 |   49.8   18.8   69.7   92.7   h200   (measured 88.8)
#   512    2   7 | 12800  41   1  10000 |  162.4    4.9  168.6  224.2  -- none --
# Transient factor 1.33x calibrated on the B=2 3x3 run; it has since been checked twice and
# runs 3-4% CONSERVATIVE (53.0 predicted / 51.4 measured; 92.7 / 88.8).  111.8 GiB is ~80%
# of the h200's 140.4 GiB, so the margin is real but it is NOT the 50% the B=3 run had.
# nf is already pinned at its floor of 1, so there is no knob left to trade if it grows.
#
# COST = *1.5-2.5 h/sigma, 6-10 h for the four -- QUOTE THE BRACKET.*  The solve is
# unchanged (same K, same 544 frequencies); what grows is the pass count (nf 3 -> 1, so
# 3x more frequency chunks, each rebuilding the feature tensor) and the lag Gram (3.24x).
# Two measured points cannot separate those terms -- the same reason the B=3 bracket was
# 2.2-3.1 h and the truth was 1.1 h.  24 h budget leaves room for the model being wrong.
#
# -------------------------------------------------------------------------------------
# SIGMA ORDER = 0.127, 1.610, 2.212, 0.452 -- most informative first, cheapest loss last
# -------------------------------------------------------------------------------------
# sigma=0.127 FIRST: it is the ONLY sigma with an existing tap-size measurement to compare
# against, AND the cell where B=2 turned around, AND where the own-gap prediction (C) bites.
# Then the high pair where the band actually earns its keep (transfer plateaus at 101-102%).
# sigma=0.452 LAST -- it is the crossing octave, interesting but the least load-bearing of
# the four, so a wall-clock death there costs least.
# B=0 runs before B=2 at each sigma (loop order is sigma -> c -> B) => the cheap plain arm
# banks first and every sigma has its paired partner even if the job dies mid-B=2.
#
# -------------------------------------------------------------------------------------
# ⚠⚠ SEPARATE OUT npz -- NEVER THE CANONICAL TABLE
# -------------------------------------------------------------------------------------
# rf_pixel_band2d_heldout.py reads the whole table ONCE at startup and rewrites ALL of it
# after every cell, so two concurrent jobs sharing one OUT silently delete each other's
# work.  48071356 is writing the canonical table right now.  Merge later with
#   APPLY=1 python scripts/merge_band2d_tables.py
# Keys here are namespaced automatically: cell_key() suffixes |t5 whenever T2 != 3, so
# these cells can never be confused with the 3x3 ones even after a merge.
#
# VALIDATION DONE BEFORE SUBMITTING: selftest_band gained a *B=2 x t=4* case (7x7 grid,
# clearing both the >= 2B+1 and >= 2t-1 rules) and passes at 1.0e-15 / 1.5e-15 / 2.9e-16 in
# sample / residual / held out.  ⚠ THE INTERACTION HAD NEVER BEEN TESTED: every B>1 case in
# the suite ran at t=2 and every t>2 case ran at B=1, and the Delta-resolved Gram
# T^n_ab(m, Delta) is indexed by the lag m (set by t) AND by Delta (set by B) -- an error in
# combining the two index sets is invisible to both single-knob cases.  A suite that moves
# one parameter at a time validates a diagonal, not a grid.
# =====================================================================================
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory

PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

export T2=5
export OUT=tables/rf_pixel_band2d_heldout_t5.npz
export CS=512
export BS=0,2
export SIGS=0.127,1.610,2.212,0.452

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/rf_pixel_band2d_heldout.py

echo "=== DONE ==="
