# Round 050 - the non-GEMV work is 15 ms, not 44: TILE=4, and three refuted hypotheses

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=**TILE=4 +1.1% median over 8 pairs** G4=**accept.sh 12/12 ACCEPTED**
- outcome: TILE=4 shipped (bytes per token down 21%), the round-048 cost model corrected, and three
  attempts to beat the GEMV shape measured and rejected.

## CORRECTION to round 048: the non-GEMV part is 15 ms, not 44 ms

Round 048 reported "the non-GEMV work of a full pass is 44-52 ms (28-33%)" - and round 047's plan
rested on it.  That number came from subtracting the bench's pure-GEMV sweep (113.15 ms) from the
full verify (157 ms) **measured in different thermal states**.  Measured instead inside one run, by
skipping every GEMV dispatch and timing what is left:

```
full verify (1634 dispatches)      109.37 / 109.82 ms
GEMV-skipped (1137 dispatches)      15.15 /  14.88 ms
```

The non-GEMV part is **15.0 ms, 14% of the pass, at 13 us per dispatch**.  The GEMV is **94.4 ms
(86%)**, and within it the 14.41 GB weight stream is 69.6 ms at the plateau's 207 GB/s.  The 86%
figure is what makes the rest of this round's negative results unsurprising.

## Three hypotheses, all measured, all rejected

**The encoder split costs nothing on a real forward.**  Round 006 had only tested this on the bench
sweep, where no two dispatches depend on each other.  The verify batch with its 1090 encoder
boundaries against one with a single encoder:

```
split (1090 encoders)  110.92 / 110.71 ms
one encoder            111.63 / 113.98 ms
```

Splitting is, if anything, faster.  Not a lever.

**More memory-level parallelism does not help.**  A lane owns only `n_groups/32` groups (K=5120 is 80
groups over 32 lanes, so two or three) and walks them one at a time, so unrolling the group loop by
two - both groups' loads in flight, two independent fp32 accumulators - should have overlapped more
of the 24.8 ms of ALU work with the 69.6 ms stream:

```
k3 + u4 + half dots    median 0.8472  calibrated 0.7992  wins 12/12
k3 + u4 + half4 acc    median 0.8493  calibrated 0.8012  wins 11/12
k3 + u4 + 2grp unroll  median 0.9273  calibrated 0.8748  wins 11/12
k3 (baseline dup)      median 1.0601  (CALIBRATION: instrument bias)
```

The unroll is 9.5% *slower* than the plain form.  Register pressure costs more than the extra loads
in flight buy.

**And the element-wise half4 accumulator, from round 049, is 1.2% slower** - `w0*x0 + w1*x1` cannot
fuse into an FMA chain the way `dot()` does.

Three structural changes to the same kernel, three regressions.  The shape is at a strong local
optimum: the extract-and-convert is ~16 of ~40 instructions per group/word/lane and irreducible, the
x loads are at their minimum for 3 rows, and the accumulator variants are both worse.

## TILE=4

With the per-row cost now cheaper, TILE=4 was re-measured end to end rather than estimated.  Two
independent sets of four alternating pairs:

```
set 1 (cooler, ~19.9 tok/s):  t3 19.98/19.83/19.95/19.80   t4 20.52/20.93/20.72/17.41*
                              median paired ratio 0.9737 -> +2.7% (pairs 1-3: +2.6, +5.2, +3.9%)
set 2 (hotter, ~16.0 tok/s):  median paired ratio 0.9938 -> +0.63%
* pair 4 of set 1 is a thermal spike (17.41 against 19.80 for the same binary)
```

Steady-state tokens per pass: **2.69 (TILE=3) against 3.41 (TILE=4)**, which implies the same
per-draft acceptance p = 0.895 in both - the drafting is unchanged, as it should be.  The 27% more
tokens per pass cost 22-26% more per pass, so the **true marginal cost of one more verified row is
~30 ms, not the ~17 ms the components predict** (GEMV row 8.3 + non-GEMV row 4 + draft 4.65).  That
gap is why TILE=4 is worth about +1%, and not the +12% a component model had suggested - the same
kind of model error round 048 made.

Shipped anyway, on the measure that does not depend on the throttle: **bytes of weight streamed per
token falls from 5.36 GB to 4.23 GB, -21%**.  The weight stream is the one term that is at a hard
floor (DRAM peak) whenever the machine is not throttled, and the machine is unthrottled at mains.

TILE=5 is not worth measuring: with p = 0.895 the fourth draft adds only 0.64 expected tokens against
another ~30 ms row.
