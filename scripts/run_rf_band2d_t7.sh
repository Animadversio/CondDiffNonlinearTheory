#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 1-00:00
#SBATCH -J rf_band2d_t7
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_t7_%j.out

# 7x7 TAPS, BAND B=1, HELD OUT, c = 32 / 96 / 256 / 512.
#
# michimin, 2026-09-23 22:30: "i think that we also should increase what a nonlinear window
# can see, i.e. is a 27-dimensional window enough for a useful nonlinear feature. can you do
# the held out experiment sweeping c up to 512 but with kxk=7x7 instead of 3x3 as of right
# now? for B=1"  --  asked to run PARALLEL TO job 48071356 (B=3 c=512, on its own h200).
#
# *** WHY THIS IS A CLEANER KNOB THAN c OR B: IT MOVES ONLY THE *RANDOM* PARAMETERS. ***
# Trained parameters = Cin * c * H * W * (2B+1)^2.  There is no tap size in that expression:
# c=512 B=1 is 14,155,776 trained parameters at 3x3 AND at 7x7.  What changes is the size of
# the random design draw, c*Cin*T2^2 = 27c -> 147c, a factor 5.44.  Every previous enlargement
# in this project (c, B, k/d) bought expressivity by buying trained parameters, and at
# sigma=0.127 that is exactly what turned around at B=2 -- the band became the most overfit
# object in the comparison (own gap +0.6462 against the free Wiener's +0.4593).  A wider
# random window has no such mechanism available to it.  This is also the direct follow-up to
# michimin's own 2026-09-17 split of the parameter count into trained vs random ("dense draws
# k*d = 18,874,368 iid numbers, banded circ draws only c*t = 49,152, 384x fewer").
#
# IT STAYS NORM-MATCHED.  filt2d draws N(0, I/(Cin*T2^2)), so E||row||^2 = 1 at both tap
# sizes -- the 7x7 window is not simply more filter energy, it is the same energy spread over
# 147 pixels instead of 27.
#
# *** PRE-REGISTERED PREDICTION, WRITTEN BEFORE THE RUN.  I EXPECT 7x7 TO BE NEUTRAL-TO-WORSE
# AT FIXED c, NOT BETTER, AND THE PROJECT ALREADY HAS EVIDENCE FOR THAT. ***
#   (a) The 2026-07-25 locality sweep (GMM, d=32, 1-D group) found the circ-vs-dense gap grew
#       MONOTONICALLY with the band width w and that FULL WIDTH WAS THE WORST CASE; the only
#       genuine wins were at w=2,3,4.
#   (b) The 2026-08-17 permutation test isolated the mechanism: banding buys FEATURE
#       DIVERSITY, because D translates of a width-t filter read D different t-subsets,
#       whereas D rotations of a full-width vector are badly redundant.  Two translates of a
#       3x3 filter offset by >=3 pixels share NO support and are nearly independent features;
#       at 7x7 translates share support out to offset 6, so at the SAME c the 1024 translates
#       are MORE correlated with each other and the effective feature count is lower.
#   (c) ⚠ BOTH are 1-D / GMM measurements.  Neither has ever been checked on a 2-D group at
#       d=3072, and 27 of 3072 pixels is a very small window in absolute terms, so this is
#       genuinely open.  That is why it is worth 1-2 h of h200.
#   ⇒ THE FALSIFIABLE PART IS THE c-TREND, WHICH IS WHY THE SWEEP IS THE EXPERIMENT AND NOT
#     JUST THE c=512 CELL.  If the mechanism is (b), the 7x7 penalty is a feature-redundancy
#     penalty and MUST SHRINK AS c GROWS -- more planes buy back the diversity the wider
#     window spent.  A penalty that is FLAT or GROWING in c falsifies (b).
#   ⇒ READING FIXED IN ADVANCE.  7x7 better at c=512 => 27 pixels really was the binding
#     constraint, and the follow-ups are 5x5 (locate the optimum) and 7x7 at B=2.  7x7 worse
#     => the answer to michimin's question is "yes, 27 is enough, and more is actively worse",
#     and the reason is redundancy between translates, not expressivity.
#   ⚠ UNPAIRED, UNLIKE EVERY BAND-VS-PLAIN NUMBER WE QUOTE.  filt2d draws
#     randn(c, CIN, T2, T2); at a different T2 the same seed gives a DIFFERENT tensor and the
#     draws are not even nested (the row stride changes).  So this is a difference of two
#     independent Theta draws.  Measured absolute Theta spread at c=512, sigma=0.127 is 0.0076
#     (plain) / 0.0080 (band B=1), so anything above ~0.02 is readable at NSEED=1 and anything
#     smaller needs a second draw -- the driver now resumes at SEED granularity, so raising
#     NSEED later APPENDS rather than skipping.
#
# SIZING, from the driver's own sizing() at BUDGET=70e9, transient factor 1.33x calibrated on
# the measured B=2 run (38.7 GiB resident predicted -> 51.4 GiB peak observed).  The binder is
# the Delta-resolved Stein lag tensor, which carries 3*(2*T2-1)^2 lag entries: 75 -> 507, i.e.
# *** 6.76x, and it is held at c^2. ***
#
#    c   B  T2 |  nf     ns |  Bc GiB   P   buf  resident  ~peak   card
#   512  1   3 |  16   2543 |     7.6  10.1  2.8      20.5   27.3   h100
#   512  1   7 |   9   4521 |    51.5   5.7  2.8      60.0   79.8   h200  <- this run
#   256  1   7 |  16   5086 |    12.9   2.5  2.8      18.2   24.2   h100
#    96  1   7 |  16  10000 |     1.8   0.4  2.1       4.2    5.6   h100
#    32  1   7 |  16  10000 |     0.2   0.0  0.7       0.9    1.2   h100
#   512  0   7 |  16  10000 |     4.0   0.1  1.2       5.3    7.1   h100
# ⇒ ONLY the c=512 B=1 cell needs the h200: 79.8 GiB predicted peak against the h100's 79 GiB
#   is not a margin, it is a coin flip.  Everything else would fit an h100 comfortably.
#
# COST: *** QUOTE THE BRACKET, 1-4 h FOR THE WHOLE GRID, NOT THE LOW END. ***  The two
# measured points (B=1 T2=3 c=512 = 360 s at nf=16 = 34 frequency passes; B=2 = 1680 s at
# nf=3) cannot separate the moment cost from the solve cost, and the last time I extrapolated
# this estimator with a single term I budgeted 8 h for a job that took 25 min.  What is known:
# the solve is UNCHANGED (K = c(2B+1)^2 = 4608 either way, T2 does not enter it), the pass
# count goes 34 -> 61 (1.8x, nf 16 -> 9) and the lag Gram goes 6.76x.  That brackets the
# c=512 B=1 cell at roughly 9-50 min, hence 0.5-2.5 h for its three sigma.  24 h is >=6x
# headroom on the pessimistic end.
#
# ORDER: c ASCENDING so the cheap cells land first and give a MEASURED per-cell time before
# the c=512 cells run, and sigma most-informative-first.  Checkpointed after every cell.
#   0.127  the only sigma where a 3x3 B=1 c-SWEEP already exists to compare against
#          (c=96/256/512), and where the nonlinear gain is largest and B=2 turned around --
#          i.e. where a zero-trained-parameter enlargement has the most to prove.
#   0.621  the mid-sigma hole: linear->EDM peaks at 9.73 there and NO class we have tested
#          captures it.  Highest upside if a wider window helps at all.
#   1.610  last: the band's high-sigma behaviour is already fully described by the linear toll
#          ladder (transfer plateaus at 101-102%), so this is the least informative of the
#          three and costs least if the wall clock kills it.
#
# *** TWO TRAPS FOUND WHILE READING THE DRIVER, BOTH FIXED BEFORE THIS WAS SUBMITTED. ***
# (1) THE CELL KEY HAD NO TAP SIZE.  It was `{sigma}|{c}|{B}`, which is not an identifier of
#     the model once T2 is a knob.  Pointed at the canonical table this job would have either
#     seen `have >= NSEED` for every stored 3x3 cell, printed "already done" and EXITED 0
#     HAVING COMPUTED NOTHING, or (at NSEED>=2) APPENDED a 7x7 draw as "seed 1" of a 3x3 cell
#     so that every mean over that cell silently averaged two different models.  Neither fails
#     loudly.  cell_key() now suffixes `|t{T2}` whenever T2 != 3, and T2=3 keeps the bare key
#     so the 27 existing cells and the two scripts that parse them are untouched.
#     (Same error family as the cell-vs-seed resume bug fixed this morning: a resume rule is
#     only as safe as the identity encoded in its key.)
# (2) THE BRUTE FORCE HAD NEVER BEEN RUN ABOVE t=2 -- all seven stored cases were t=2, so the
#     lag machinery was validated only at its smallest non-trivial setting.  Exactly the hole
#     B=3 was in before job 48071356.  Two cases added permanently to
#     core/rf_circulant2d_band.py::selftest_band and both pass against the explicit real
#     constrained least squares: 13x13 Cin=1 c=1 t=7 B=1 at 5.1e-16 / 0.0e+00 / 6.2e-16 and
#     7x7 Cin=2 c=2 t=4 B=1 at 4.0e-16 / 0.0e+00 / 0.0e+00, in sample, residual and held out.
#     ⚠ The t=7 case runs at 13x13 because the lag set is m in [-(t-1),t-1]^2 taken mod (H,W):
#     on a grid smaller than 2t-1 two distinct lags ALIAS and the Delta-resolved Gram
#     double-counts.  13 = 2*7-1 exactly, so it is the tightest case available and any
#     off-by-one in the wraparound shows up as a hard disagreement.  (The estimator raises on
#     2t-1 > min(H,W) rather than aliasing silently; on CIFAR 13 <= 32, so 7x7 is legal.)
#
# *** SEPARATE OUT FILES, NOT THE CANONICAL TABLE -- REQUIRED TWICE OVER. ***  Job 48071356 is
# concurrently writing tables/rf_pixel_band2d_heldout.npz, and these drivers read the whole
# table once at startup and rewrite ALL of it after every cell, so the second job to save
# would delete every cell the other finished in between.  Merge afterwards with
# `APPLY=1 python scripts/merge_band2d_tables.py`, which VERIFIES shared keys and refuses on
# mismatch rather than overwriting.
#
# NOTE comma-valued env vars are set HERE, not via `sbatch --export` (sbatch splits on commas).
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
mkdir -p logs tables
nvidia-smi -L
nvidia-smi --query-gpu=memory.total --format=csv

PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

# ---------------------------------------------------------------------------------------
# STEP 1 -- COMPLETE THE 3x3 PARTNER GRID (cheap, ~20-30 min, and it must run FIRST because
# without it the 7x7 numbers have nothing to be compared against at sigma=0.621 and 1.610).
# The stored 3x3 B=1 c-sweep exists ONLY at sigma=0.127 (c=96/256/512); at the other two sigma
# only c=512 exists.  This fills all 12 (sigma, c) cells into a SIDE table.
# The three cells that already exist in the canonical table are DELIBERATELY recomputed: same
# seeds, same deterministic estimator, so they must come back equal, and the merge script
# checks exactly that.  An independent recomputation is a real cross-check, not a formality
# (it is what caught nothing and would have caught a moved split instantly, twice already).
# ⚠ They may differ in the last few digits rather than exactly: nf/ns depend on BUDGET, and a
# different chunking changes the floating-point summation order.  Agreement at ~1e-12 is the
# expected outcome, not 0.0e+00.
# ---------------------------------------------------------------------------------------
export T2=3
export OUT=tables/rf_pixel_band2d_heldout_t3fill.npz
export CS=32,96,256,512
export BS=1
export SIGS=0.127,0.621,1.610
$PY scripts/rf_pixel_band2d_heldout.py

# ---------------------------------------------------------------------------------------
# STEP 2 -- THE RUN michimin ASKED FOR.  B=0 is included at negligible cost (K=512, and the
# Delta tensor is 13x smaller at nD=1) and is worth having: it gives the PLAIN 7x7 arm, which
# is what makes "does the band still buy anything once the window is wider?" answerable at
# 7x7 instead of only "is 7x7 better than 3x3 inside the band class".
# ---------------------------------------------------------------------------------------
export T2=7
export OUT=tables/rf_pixel_band2d_heldout_t7.npz
export CS=32,96,256,512
export BS=0,1
export SIGS=0.127,0.621,1.610
$PY scripts/rf_pixel_band2d_heldout.py

echo "ALL DONE"
