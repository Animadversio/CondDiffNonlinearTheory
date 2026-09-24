#!/bin/bash
#SBATCH -p kempner_h200
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 1-00:00
#SBATCH -J rf_band2d_t7B3
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_t7B3_%j.out
#
# ============================================================================
# 7x7 TAPS AT BAND ORDER B=3, HELD OUT, sigma = 1.610 and 2.212.
#
# michimin, 2026-09-24 19:11: "run c=512 B=3 at kxk=7 at noise sigma 1.61 and
# 2.21 only".
#
# *** c=512 IS NOT RUNNABLE AND THIS SCRIPT RUNS c=256 INSTEAD.  DO NOT SUBMIT
# IT UNTIL michimin HAS SAID WHICH c THEY WANT. ***  At B=3 the Delta index set
# doubles relative to B=2 (nD = ((4B+1)^2+1)/2 = 41 -> 85), and 7x7 B=2 c=512
# was ALREADY the "-- none --" row of run_rf_band2d_t5_B2.sh's sizing table:
#
#    7x7, c=512    nD    Bc/split   both splits   ~peak (2.12x)
#       B=2        41     81.2 GiB   162.4 GiB      204 GiB
#       B=3        85    168.3 GiB   336.7 GiB      357 GiB   <- as requested
#    h200 = 140.4 GiB
#
# ONE SPLIT'S Delta-TENSOR ALONE (168.3 GiB) IS LARGER THAN THE WHOLE CARD, and
# nf is already pinned at its floor of 1 at every c >= 224 here, so there is no
# knob left to trade.  The two-pass restructure that would have halved it is
# CLOSED -- michimin killed it 2026-09-24 06:26 ("we might need to prioritize
# runtime over memory at any point in time"), and that decision is recorded in
# sizing()'s docstring.  So the choice is which c, not how to fit c=512.
#
# FEASIBLE FRONTIER AT 7x7 B=3 (Bc = nD*c^2*3*(2t-1)^2*16 B per split, nL = 169;
# peak factor 2.12x MEASURED on 2026-09-24, not estimated):
#
#      c      K     nf   Bc/split   ~peak    card
#    128    6272    16     10.5      22      h100
#    192    9408     3     23.7      50      h100
#    224   10976     1     32.2      68->73  h100
#    256   12544     1     42.1      89->94  h200 (67%, real margin)   <- THIS
#    288   14112     1     53.3     113->119 h200 (85%)
#    320   15680     1     65.8     139->147 -- over --
#    384+                                    -- none --
#    max c: h100 241, h200 321.  ^ UPPER BOUNDS: the peak factor was measured at
#    small nf/P, and the production P term adds ~5 GiB at these K.
#
# *** WHY c=256 AND NOT c=288: IT IS THE ONLY FEASIBLE c THAT ALREADY HAS 7x7
# PARTNERS ON DISK. ***  tables/rf_pixel_band2d_heldout_t7.npz holds B=0 and B=1
# at c = 32/96/256/512.  `filt2d(c, s)` seeds on 900 + 11*s + c -- it depends on
# (c, seed) ONLY, not on B and not on sigma -- so at c=256, T2=7 the B=0/1/3
# ladder is a PAIRED comparison on one fixed feature map.  c=288 would be
# unmoored: a single unpaired cell with nothing at its own tap size and width to
# difference against.  The margin is the second reason, not the first.
#
# ============================================================================
# ⚠⚠ sigma=2.212 DOES NOT EXIST AT 7x7 AT ANY B.  The _t7 grid is
# 0.127/0.621/1.610 only.  So BS=0,3 below, NOT BS=3: the B=0 arm at sigma=2.212
# is what gives that cell a plain arm to difference against.  It is nearly free
# (nD=1 => Bc = 0.5 GiB, minutes) and without it the sigma=2.212 B=3 number can
# only be read against the held-out Wiener and against an UNPAIRED 3x3 cell.
# At sigma=1.610 the B=0 cell is already stored, so the driver's seed-granularity
# resume will skip it and go straight to B=3.
#
# sigma ORDER: 1.610 FIRST.  It is the sigma with the full 7x7 partner grid, the
# sigma where the pre-registration below is scoreable, and the sigma where the
# wide window earns its keep (the band x window coupling shift is -0.0445 at
# c=256 / -0.0708 at c=512 there, i.e. a wider window makes the band MORE useful
# -- the opposite of its sign at sigma=0.127).  A wall-clock death on the second
# cell therefore loses the less informative half.
# ============================================================================
# *** PRE-REGISTERED BEFORE ANY NUMBER EXISTS (sigma=1.610 only). ***
# Stored, held out, 7x7 c=256 sigma=1.610:  B=0 78.2223, B=1 75.7043, paired
# d = -2.5180.  The LINEAR band toll is TAP-SIZE-FREE (T2 lives only in the
# random draw; the best-linear-in-class calculation has no Theta in it), so the
# same held-out ladder applies at 7x7 as at 3x3: at sigma=1.610 the toll is
# B=1 3.6046 -> B=3 2.0664, i.e. *1.5382 removed*.  At 3x3 the incremental
# transfer at this sigma is PLATEAUED at 102.2% (B=1->B=2) and 98.7%
# (B=2->B=3).  At 100% transfer:
#
#     7x7 B=3 c=256, sigma=1.610  ~=  75.7043 - 1.5382  =  *74.17*
#                                     (bracket 73.9 - 74.5)
#
# ⇒ THE SHARP READING: that is essentially the 3x3 B=3 c=512 value (74.1024) at
# HALF THE WIDTH.  FALSIFIERS NAMED: lands ABOVE ~74.5 => transfer does NOT
# plateau at the wider window and the high-sigma toll ladder is a 3x3-only
# model; lands BELOW ~73.9 => the band x window coupling KEEPS COMPOUNDING with
# band order (the shift is -0.0445 at B=1 here; a B=3 shift near -0.15 would do
# it) and 7x7 B=3 is the best RF cell in the project at this sigma.
#
# ⚠ NO PREDICTION IS MADE AT sigma=2.212 -- no 7x7 measurement exists at that
# sigma at any B or c, so there is nothing to extrapolate from and a number
# invented here would only look like a result afterwards.
#
# ⚠ UNPAIRED vs 3x3: at a different T2 the same seed gives a different, NOT
# NESTED Theta (the row stride changes).  The absolute Theta spread at c=512
# sigma=1.61 is 0.0327 => cross-tap-size deltas below ~0.03 are NOT readable at
# NSEED=1.  The B=0/1/3 ladder INSIDE 7x7 c=256 is paired and carries the much
# smaller 0.0004-0.0008 paired error bar.
# ============================================================================
# VALIDATION DONE BEFORE THIS SCRIPT WAS WRITTEN (logs/selftest_band_B3t4.log):
# `selftest_band` now carries a (7x7, Cin=1, c=1, t=4, B=3) case -- 14/14 PASS,
# the new one at 7.2e-16 / 2.9e-15 / 1.6e-15 in sample / residual / held out,
# and all ten prior brute-force cases reproduce their stored values exactly.
# It exists because the ONLY B x t interaction point the suite had was (B=2,
# t=4), and this run is (B=3, t=7) -- further off default on both knobs than
# anything the reference had ever seen.
# ⚠⚠ (B=3, t=7) ITSELF CANNOT BE BRUTE-FORCED: the 2t-1 rule forces a 13x13 grid
# and the reference's basis tensor E is then 8281 x 8281 x 169 x 8 B = *86 GiB*
# even at Cin = c = 1.  So the t=7 axis is validated at B=1, the B=3 axis at t=2
# and t=4, and the far corner is covered by those three and NOT directly.  SAY
# SO when reporting these numbers.
# ============================================================================
# COST = *1.5-4 h/sigma -- QUOTE THE BRACKET, NOT A POINT.*  K = c(2B+1)^2 =
# 12544 puts the solve within 2% of the 3x3 B=2 c=512 cell (measured ~1680 s per
# sigma), but nf is at its FLOOR of 1 => 544 frequency chunks instead of 182,
# and the Delta-resolved lag Gram is 6.76x larger at 7x7 than at 3x3.  Two
# measured points cannot separate a per-pass term from a per-Gram term; every
# previous attempt to quote a single number from two points has been wrong in
# both directions (B=3 3x3 came in at 1.1 h against a 2.2-3.1 h bracket; 5x5 was
# quoted 1.5-2.5 h/sigma).  24 h budget covers the bracket 3x over.
#
# OUT = tables/rf_pixel_band2d_heldout_t7.npz, the tap-size side table that
# ALREADY holds the B=0/B=1 partners and that `rf_band2d_report.py` reads
# directly (cells are namespaced |t7 by cell_key(), so there is no collision
# with the 27 stored 3x3 cells and nothing to merge, ever).
# ⚠ THE DRIVER LOADS THAT TABLE ONCE AND REWRITES ALL OF IT AFTER EVERY CELL =>
# NOTHING ELSE MAY WRITE TO IT WHILE THIS RUNS.  Check squeue first.  It is
# committed at 5956ff9, so a truncated save is recoverable from git.
# ============================================================================
set -e
cd /n/home12/binxuwang/Github/CondDiffNonlinearTheory
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PYTHONPATH=/n/home12/binxuwang/Github/CondDiffNonlinearTheory:$PYTHONPATH

export NIMG=10000
export NTEST=10000
export NSEED=1
export BUDGET=70e9

export T2=7
export OUT=tables/rf_pixel_band2d_heldout_t7.npz
export CS=256
export BS=0,3
export SIGS=1.610,2.212

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/rf_pixel_band2d_heldout.py
echo "=== DONE ==="
