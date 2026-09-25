#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 1-00:00
#SBATCH -J rf_band2d_t7B4
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/band2d_t7B4_%j.out
#
# ============================================================================
# 7x7 TAPS, BAND ORDER B=4, c=128, sigma = 1.610 and 2.212 -- HELD OUT
# ============================================================================
#
# REQUEST, VERBATIM (michimin, 2026-09-24 23:20):
#     "run c=128 at B=4 at kxk=7x7 at noise sigma=1.61 and 2.212"
#
# Run as asked: c=128, B=4, T2=7, sigma in {1.610, 2.212}.  No substitution of a
# "better" c or B.  The one addition is the B=0 arm -- see PAIRING below; it costs
# 0.3 GiB and is the only way either sigma gets a paired plain arm at c=128.
#
# ----------------------------------------------------------------------------
# VALIDATION: B=4 HAD NEVER BEEN EXERCISED BY THE BRUTE FORCE.  CLOSED FIRST.
# ----------------------------------------------------------------------------
# Every stored selftest_band case ran at B <= 3.  From B=3 to B=4 BOTH index sets
# grow: nD = ((4B+1)^2+1)/2 goes 85 -> 145 and nR = (2B+1)^2 goes 49 -> 81, so an
# error combining the Delta index set with the lag index set is invisible to every
# prior case.  New case (9, 9, 1, 1, 4, 300, 0.85, 4) = B=4, t=4 on the MINIMAL
# legal grid (box(4) offsets are mod (H, Wd) => both dims need >= 2B+1 = 9; a
# rectangular grid does not help.  9 also clears >= 2t-1 = 7).
#
#   job 48364740, logs/selftest_band_48364740.out:  selftest_band: PASS
#     new case  9x9 Cin=1 c=1 t=4 B=4 : train 0.0e+00  resid 2.5e-15  test 2.2e-16
#     all eleven prior cases reproduce their stored values DIGIT-FOR-DIGIT
#     => the edit is both validated AND inert.   measured peak 105.56 GiB.
#
# *** !! THE TRUE PRODUCTION CORNER (B=4, t=7) CANNOT BE BRUTE-FORCED, AND THAT IS
# A HARD LIMIT, NOT A CHOICE. ***  The 2t-1 rule forces a 13x13 grid, and E is
# (Cin*c*nR*D) x (c*nR*D) x d doubles with nR = 81, D = 169 => 13689 x 13689 x 169
# x 8 B = *253 GiB* even at Cin = c = 1 (the already-impossible B=3 corner was
# 86 GiB).  So t=7 is validated at B=1, B=4 is validated at t=4, and this cell is
# covered BY THOSE TWO AND NOT DIRECTLY.  *** SAY THIS when reporting any 7x7 B=4
# number. ***
#
# ----------------------------------------------------------------------------
# SIZING -- from the driver's own sizing() executed at T2=7, BUDGET=70e9, not from
# memory.  Bc = nD*c^2*3*(2*T2-1)^2*16 per split; peak model = 2.12x ONE split's Bc
# (measured) plus the P term 2*nf*K^2*16.
#
#      c   B       K    nD   nf  Bc/split   x2.12      P     tot    card
#    128   0     128     1   16      0.12     0.3    0.0     0.3    free
#     96   4    7776   145   11     10.10    21.4   19.8    41.2    h100
#    128   4   10368   145    4     17.95    38.0   12.8    50.9    h100 (64%)  <- THIS
#    160   4   12960   145    1     28.04    59.5    5.0    64.5    h100 (81%)
#    192   4   15552   145    1     40.38    85.6    7.2    92.8    h200 only
#    256   4   20736   145    1     71.79   152.2   12.8   165.0    -- none --
#
# => c=128 B=4 lands at ~51 GiB = 64% of the h100's 79.17.  Real margin.
# ! THE 2.12x FACTOR IS CALIBRATED AT SMALL nf AND THE LAST BIG CELL OVERRAN IT:
#   the c=256 B=3 job peaked at 105.7 GB against ~94 predicted (~11% optimistic).
#   Treat 51 GiB as a LOWER bound; the 28 GiB of headroom here is why this is an
#   h100 job and not a bet.
#
# COST: *quote a bracket, 40 min - 2 h per sigma.*  K = 10368 is below the c=256
# B=3 solve (K = 12544, measured 1h04m/sigma) but nf = 4 not 1 => 136 frequency
# chunks not 544, while nD is 145 not 85 (1.7x the Delta-Gram).  Two measured
# points cannot separate those terms and every previous point estimate has missed
# in both directions.
#
# ----------------------------------------------------------------------------
# PAIRING -- WHY BS=0,4 AND NOT BS=4.
# c=128 EXISTS AT NO BAND ORDER AND NO SIGMA ANYWHERE IN THIS PROJECT (verified
# against _t7.npz: the 7x7 grid is c = 32/96/256/512, and the 3x3 tables have no
# c=128 either).  filt2d seeds on 900 + 11*s + c => the B ladder at a FIXED (T2, c)
# shares one Theta draw and is properly paired; across c it is NOT.  So without the
# B=0 arm this is a lone unpaired point with nothing at its own (T2, c) to
# difference against.  The B=0 cell is nD=1 => Bc 0.12 GiB, 0.3 GiB peak, and runs
# in well under a minute.  Direct precedent: on job 48316205 the same BS=0,3
# decision produced the 53 s B=0 arm that was the ONLY thing giving sigma=2.212 a
# paired differential at 7x7.
#
# SIGMA ORDER: 1.610 first.  It is the sigma with a measured 7x7 c-sweep at B=0/B=1
# (c = 32/96/256/512), so its prediction is an interpolation; sigma=2.212 has a
# single c (256) at B=0/B=3 and nothing else, so its prediction borrows the c-shape
# from 1.610.  Running 1.610 first means a wall-clock death loses the weaker cell.
#
# ----------------------------------------------------------------------------
# *** PRE-REGISTERED BEFORE ANY B=4 NUMBER EXISTS.  Every input below was read from
# the npz by execution, not quoted from memory. ***
#
# Held-out LINEAR tolls vs Wiener test (rf_band_relaxation_heldout{,_sig2212}.npz,
# keyed on the SIGMA VALUE and column [1] = test -- these files store one ROW PER
# sigma and each row is a [train, test] PAIR):
#     sigma=1.610  B0 6.0196  B1 3.6046  B2 2.6928  B3 2.0664  B4 1.6610
#     sigma=2.212  B0 6.8849  B1 3.8333  B2 2.8435  B3 2.1670  B4 1.7316
#   => toll removed B0->B4 = 4.3585 (1.610) and 5.1533 (2.212).
#
# Measured transfer at 7x7 c=256, B=0 -> B=3 (paired, one Theta draw):
#     sigma=1.610  RF gain 4.0588 / toll removed 3.9532 = *102.7%*
#     sigma=2.212  RF gain 4.7999 / toll removed 4.7179 = *101.7%*
#
# B=0 at c=128, sigma=1.610, by log-interpolation between MEASURED c=96 (78.4353)
# and c=256 (78.2223), weight log(128/96)/log(256/96) = 0.2933 => *78.3728*.  That
# is an INTERIOR interpolation, not an extrapolation off the end of a line (the
# 2x2 miss of 2026-09-24 was the latter).
#
#   sigma=1.610  ~= 78.3728 - 4.3585*1.027 + (c-shrink ~0.04) = *73.9*  [73.6 - 74.3]
#   sigma=2.212  ~= 94.09   - 5.1533*1.017 + (c-shrink ~0.05) = *88.9*  [88.5 - 89.3]
#
# (The c=128 B=0 anchor at sigma=2.212 borrows the 1.610 c-shape: c=128 sits +0.1505
# above c=256 there, so 93.9391 + 0.15 ~= 94.09.  The c-shrink term is the measured
# d(B=1) c-trend at sigma=1.610 -- -2.4654/-2.4726/-2.5180/-2.5486 at c=32/96/256/512,
# i.e. the band buys slightly LESS at smaller c -- scaled by the B=4/B=1 toll ratio.)
#
# => THE SHARP READING IF IT LANDS: *73.9 at c=128 B=4 would beat the B=3 c=256
# value 74.1636 on HALF THE WIDTH and 0.83x the trained parameters* (3*128*1024*81
# = 31,850,496 vs 3*256*1024*49 = 38,535,168).  Same at 2.212: 88.9 vs 89.1392.
#
# FALSIFIERS, NAMED:
#   > 74.1636 at sigma=1.610 => halving c costs MORE than B=3->B=4 buys, i.e. at
#     7x7 the binding axis is WIDTH, not band order.  That would be the first time
#     the two axes traded against each other in this direction.
#   < 73.6 => transfer EXCEEDS 100% into a fourth band order at a smaller width,
#     i.e. the plateau keeps paying past B=3.
#
# ! THIS PREDICTION CROSSES BOTH THE B AND c AXES AT ONCE and is therefore STRICTLY
#   WEAKER than the 74.17 pre-registration (a pure B-extrapolation at fixed c, which
#   hit to 0.006).  B=3 -> B=4 transfer is UNTESTED AT ANY TAP SIZE.
# ! NSEED=1 (one Theta draw) and there is no paired differential ACROSS tap sizes
#   or across c -- only within (T2=7, c=128).
#
# ----------------------------------------------------------------------------
# OUT = the EXISTING 7x7 side table.  cell_key() namespaces every key as |t7 so
# there is no collision with the canonical table, and rf_band2d_report.py reads it
# directly => NOTHING TO MERGE, EVER.
# *** !! NOTHING ELSE MAY WRITE _t7.npz WHILE THIS JOB LIVES. *** The driver loads
# the whole table once at startup and rewrites all of it after every cell, so a
# second writer silently deletes this job's cells.  Queue was verified clean of any
# other writer at submit.
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
export CS=128
export BS=0,4
export SIGS=1.610,2.212

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/rf_pixel_band2d_heldout.py
echo "=== DONE ==="
