#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#SBATCH -J rf_band2d_t2B2
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_t2B2_%j.out
#
# =====================================================================================
# 2x2 TAPS AT B=2, c=512, HELD OUT -- michimin 2026-09-24 00:18:
#   "run held-out 2d circulant at B=2 at c=512 at kxk=2x2"
#
# This is the NARROW end of the tap-size sweep, requested ~5 min after the 5x5 B=2 job
# (48095626) was cancelled.  It is the follow-up I offered at 00:10 when the 5x5 plain
# arm landed and settled the SHAPE of the window penalty.
#
# -------------------------------------------------------------------------------------
# *** WHY 2x2: THE PENALTY IS LINEAR IN k AND THIS IS THE LAST STEP DOWN THAT EXISTS ***
# -------------------------------------------------------------------------------------
# Plain arm (B=0), c=512, sigma=0.127, held out, all three tap sizes measured:
#
#    taps    px     test     penalty vs 3x3    own gap
#    3x3     27    7.2765         --           +0.0511
#    5x5     75    7.5084       +0.2319        +0.0736
#    7x7    147    7.7258       +0.4493        +0.0899
#
# Per-step (k -> k+2): +0.2319 then +0.2174.  A STRAIGHT LINE IN k.  And the scaling
# discriminates mechanisms: 75 px is 40% of the way 27->147 by AREA but the measured
# penalty is 51.6% of 7x7's = exactly the T2 fraction, so whatever hurts scales with the
# RADIUS of the window, not its area.
# => 3x3 is ON A SLOPE, not at an interior optimum, and the line says SMALLER IS BETTER.
# 2x2 is the only remaining step: *** 1x1 IS DEGENERATE -- Theta_a = g_a * I, no spatial
# mixing at all, and relu(g*y) = |g| relu(+-y) so the span collapses to {y, |y|} at ANY c.
# The 2026-07-25 1-D locality sweep put the optimum at w = 2,3,4 with w=1 degenerate, so
# 2x2 is the 2-D analogue of the last non-degenerate point. ***
#
# -------------------------------------------------------------------------------------
# *** PRE-REGISTERED, BEFORE ANY 2x2 NUMBER EXISTS ***
# -------------------------------------------------------------------------------------
# (A) PLAIN ARM (B=0), sigma=0.127: test ~= *7.164*, i.e. *-0.11 BETTER than 3x3*.
#     Arithmetic: mean per-step is +0.2247 over a step of 2 in k; 3 -> 2 is half a step.
#     Bracket 7.13-7.20.  This is the first PREDICTED IMPROVEMENT in the whole tap-size
#     sweep -- every measured point so far has been a penalty.
# (B) B=2 (the cell actually requested): if the plain-arm offset carries, *~7.22* against
#     the stored 3x3 B=2 c=512 value of *7.3342* (2 seeds).
# (C) *** THE PAIRED BAND DIFFERENTIAL IS THE SHARPEST TEST, because it tests the claim I
#     replaced the retracted one with. *** At 3x3, d(B=2) = *+0.0577* (the band HURTS --
#     the interior optimum in B).  The 7x7 run showed window size and band order push the
#     SAME knob: the d(B=1) shift 3x3 -> 7x7 at c=512 was *+0.0725* over two steps of 2,
#     i.e. ~+0.036/step, and a wider window moved the turnaround from B=2 down to B=1.
#     Running that backwards, 3 -> 2 is half a step => predicted d(B=2) at 2x2 ~= *+0.040*:
#     STILL POSITIVE, BUT SHRINKING.
#     FALSIFIERS, stated now:
#       * d goes NEGATIVE  => the coupling is STRONGER on the narrow side than the linear
#         transfer predicts; the interior optimum in B moves back up past 2 at 2x2.
#       * d unchanged at ~+0.058 => the coupling is ONE-SIDED (only wide windows move it)
#         and "same knob" is too strong a statement.
#     ⚠ The +0.036/step transfer is measured at B=1 and applied at B=2.  That cross-B step
#     has no error bar attached to it -- treat the number as a direction with a magnitude,
#     not as a tight prediction.
# (D) OWN GAP at B=2: 3x3 gives +0.6462.  The 7x7-vs-3x3 own-gap inflation at B=1 c=512 was
#     2.11x over two steps => ~1.45x/step => 2x2 predicted ~*+0.54* (bracket 0.50-0.58).
#     Plain-arm own gaps are near-linear in k (+0.0511/+0.0736/+0.0899) => 2x2 plain ~+0.041.
#
# -------------------------------------------------------------------------------------
# ⚠ 2x2 IS THE FIRST *EVEN* TAP SIZE MEASURED -- FLAG IT BEFORE THE NUMBER LANDS
# -------------------------------------------------------------------------------------
# 3, 5, 7 are all odd and have a centre pixel; 2 does not.  filt2d writes the taps into
# h[:, :, :T2, :T2], i.e. a corner, at EVERY T2, so the filter is never centred anyway, and
# for the plain arm a global shift of the filter is a pure phase in Fourier which the
# full-width circulant readout absorbs => parity should NOT matter structurally.
# But that is an argument, not a measurement.  *** IF 2x2 BREAKS THE LINE, PARITY IS THE
# FIRST CANDIDATE TO CHECK BEFORE CONCLUDING THE LINE BENDS.  The clean control would be
# 4x4 (even, between 3 and 5) -- cheap, and worth running before re-theorising. ***
#
# -------------------------------------------------------------------------------------
# ⚠ UNPAIRED ACROSS TAP SIZES, PAIRED WITHIN ONE (state both halves, as always)
# -------------------------------------------------------------------------------------
# filt2d draws randn(c, CIN, T2, T2) / sqrt(CIN*T2^2); at a different T2 the same seed gives
# a DIFFERENT tensor and the draws are not even nested (the row stride changes).  So every
# 2x2-vs-3x3 delta is a difference of two independent Theta draws.  Measured absolute Theta
# spread at c=512 sigma=0.127 is 0.0076 (plain) / 0.0080 (band) => the predicted -0.11 is
# ~14x the noise and is readable at NSEED=1.  The predicted differential SHIFT (-0.018) is
# NOT -- but the differential itself is PAIRED inside one tap size (B=0 and B=2 share base
# filters at fixed T2) and its own error bar is 0.0004-0.0008, so d is resolved even though
# the cross-tap-size shift in d is not.  BS=0,2 is what buys that.
# NORM-MATCHED: E||row||^2 = 1 at every T2.  2x2 is 12 random numbers per feature (Cin=3 x
# 4 px) against 27 / 75 / 147 -- the SAME energy in fewer pixels, not less energy.
# TRAINED PARAMS ARE UNCHANGED AT 39,321,600 (= Cin*c*H*W*(2B+1)^2, no T2 in it).
#
# -------------------------------------------------------------------------------------
# SIZING -- this run is CHEAPER THAN THE 3x3 ONE, which is why it fits the h100
# -------------------------------------------------------------------------------------
# The binder is the Delta-resolved Stein lag tensor,
#   fixed = 2 * nD * c^2 * 3 * (2*T2-1)^2 * 8 * 2,  nD = ((4B+1)^2+1)/2 = 41.
# (2*T2-1)^2 = 9 at T2=2 vs 25 at T2=3 => *0.36x*, and K = c(2B+1)^2 = 12800 is UNCHANGED
# by T2 (the solve does not feel the tap size at all).  The freed budget goes into nf:
#     c    B  T2 |     K  nD  nf     ns | Bc GiB  P GiB  resid   ~peak   card
#   512    2   2 | 12800  41   5   2929 |   8.65  24.41  35.86   47.7    h100   <- THIS RUN
#   512    2   3 | 12800  41   3   4882 |  24.02  14.65  41.47   55.2    h100  (measured 51.4)
#   512    2   5 | 12800  41   1  10000 |  77.84   4.88  84.63  112.6    h200  (cancelled)
#   512    0   2 |   512   1  16  10000 |   0.21   0.12   1.56    2.1    h100
# 47.7 GiB on a 79 GiB card = a real margin, and nf=5 means FEWER frequency passes than the
# 3x3 run's nf=3 (544/5 = 109 chunks vs 182).
# COST: the 3x3 B=2 run measured ~1680 s/sigma.  Fewer passes and a 0.36x lag Gram both cut
# it; the solve is identical.  *** QUOTE 15-30 min/sigma, 1-2 h for the four *** -- two
# measured points still cannot separate pass count from Gram size, which is exactly why the
# B=3 bracket (2.2-3.1 h) came in at 1.1 h.  8 h budget.
# ⚠ THE B=0 CELL IS IN THE nf=16 BUFFER-DOMINATED REGIME WHERE THIS MODEL IS NOT CALIBRATED
# -- it missed the 5x5 B=0 cell by 2.4x (7.5 predicted / 18.0 measured).  Irrelevant here
# (2.1 GiB predicted, so even 3x over is nothing), but do not trust 1.33x at nf >> 1.
#
# -------------------------------------------------------------------------------------
# SIGMA SET = the same four as michimin's last two tap-size requests ("the first 2 and
# 1.61 and 2.21"); they did not name a set this time.  Order 0.127, 1.610, 2.212, 0.452.
# -------------------------------------------------------------------------------------
# sigma=0.127 FIRST: the ONLY sigma with an existing tap-size measurement to compare against,
# AND where B=2 turned around, AND where (A)-(D) all bite.  Then the high pair where the band
# earns its keep (transfer plateaus at 101-102%).  sigma=0.452 LAST -- the crossing octave,
# interesting but least load-bearing, so a wall-clock death there costs least.
# B=0 runs before B=2 at each sigma (loop order sigma -> c -> B) => the cheap plain arm banks
# first.  That ordering is what saved the 5x5 result when 48095626 was cancelled mid-B=2.
#
# -------------------------------------------------------------------------------------
# ⚠⚠ SEPARATE OUT npz -- NEVER THE CANONICAL TABLE
# -------------------------------------------------------------------------------------
# rf_pixel_band2d_heldout.py reads the whole table ONCE at startup and rewrites ALL of it
# after every cell, so two concurrent jobs sharing one OUT silently delete each other's work.
# 48071356 (B=3) is writing the canonical table right now and 48084263 (7x7) is writing _t7.
# Merge later with:  APPLY=1 python scripts/merge_band2d_tables.py
# cell_key() suffixes |t2 automatically whenever T2 != 3, so these cells can never collide
# with the 3x3 ones even after a merge.
#
# VALIDATION: *** NO NEW CASE NEEDED -- t=2 x B=2 IS ALREADY THE MOST-VALIDATED CORNER. ***
# selftest_band case (5x5, Cin=1, c=2, t=2, B=2) is EXACTLY this knob combination and passes
# at 1.2e-15 / 2.5e-15 / 1.3e-15 (in sample / residual / held out); re-run 2026-09-24 before
# submitting, full suite PASS.  Contrast the 5x5 run, which needed a new (t=4, B=2) case
# because the B x t interaction had never been exercised off BOTH defaults.  Here t=2 IS the
# default that every original case used.  Grid rules are clear too: 5 >= 2B+1 = 5 and
# 5 >= 2t-1 = 3.
# =====================================================================================
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory

PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

export T2=2
export OUT=tables/rf_pixel_band2d_heldout_t2.npz
export CS=512
export BS=0,2
export SIGS=0.127,1.610,2.212,0.452

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/rf_pixel_band2d_heldout.py

echo "=== DONE ==="
