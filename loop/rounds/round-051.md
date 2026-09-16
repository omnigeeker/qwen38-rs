# Round 051 - the per-row ALU is the wall, not the weight stream

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity (rerun below) G3=none shipped G4=runtime unchanged from accepted b75f13e
- outcome: the cost model is inverted; TILE=6 rejected; the half accumulator rejected; the
  simdgroup_matrix path refuted on paper before being written; `QW_COOL_SLEEP` added.

## Where the pass actually goes

The verify batch is the pass: 1858 dispatches (TILE=4), 22 of them per 60-token run, mean
**135.4 ms** against a whole pass of ~158 ms.  The per-phase printout that says "verify 47.06
ms/pass" is measuring something else and must not be used.

Attribution inside one run:

```
                          TILE=4        TILE=3
verify (all dispatches)   135.43 ms     114.58 ms
GEMV skipped               13.50 ms      13.30 ms   (1361 / 1137 dispatches)
=> GEMV                   121.93 ms     101.28 ms
```

The non-GEMV part is **flat in the number of rows** (13.30 -> 13.50), so the whole +20.6 ms is
inside the GEMV.  Two points give `GEMV(k) = 39.5 + 20.6k` - and that is the inversion: two
thirds of the GEMV is ALU that grows with rows, and only one third is the weight stream.

## The clean measurement: the throttle is on memory, not on the cores

The prefill is twelve sequential full-weight passes, which is enough load to reach the throttled
plateau before the first decode token is timed.  `QW_COOL_SLEEP=<sec>` idles after the prefill so
the part cools; the first tokens then run at the unthrottled clock, which is the closest this
machine (battery + Low Power Mode) gets to a mains figure.

```
cool, after 90 s idle            hot plateau
k=1 forward  30.90-31.04 ms      ~130 ms       4.2x
TILE=4 verify 105.6 / 105.8 ms   135.4 ms      1.28x
non-GEMV       10.6 ms            13.5 ms      1.27x
```

The k=1 forward streams all 14.41 GB in 26 ms of its 31 - **554 GB/s, the DRAM peak, with the one
row's ALU completely hidden**.  The stream is 4.2x slower when hot; the per-row cost is not
(95/4 = 23.8 cool against 20.6 hot).  So the thermal throttle here is a **memory-path** throttle,
and the per-row ALU is an issue-rate wall that does not move with it.

That is why four separate structural attacks on this kernel all failed: the wall is not the memory
system, which is what they were all aimed at.

## Rejected this round

**TILE=6.**  Steady state 4.26 tokens/pass against TILE=4's 3.41 (+25%), but four alternating pairs
gave ratios 1.0824 / 1.0972 / 1.0917 / 1.1504, median **TILE=6 is 8.63% slower**.  The marginal
verified row costs ~30 ms and yields 0.425 tokens, below the pass average of 0.0209 tokens/ms, so
TILE=4 is the optimum of that curve and not just the best measured point.

**The half accumulator.**  A lane owns only two or three groups, so the row partial sums can stay
in half and the convert + fp32 add can leave the hot loop.  Twelve rounds: calibrated 0.8071
against the shipped fp32-accumulate form's 0.8053, 14/14 wins but a wash.  Fourth attack, fourth
regression.

**simdgroup_matrix, before writing it.**  Per 256-weight k-tile the staging costs ~59 warp
instructions (8 stores for W, 8 loads + 8 stores for X, 2 barriers, 4 MMAs) = 0.23 warp-inst per
weight.  The shipped kernel is already at 0.203 (52 lane instructions per 8 weights, amortised by
registers across rows).  The matrix unit cannot pay for its own staging, because the nibble
extract - 3 instructions per weight - has to happen either way and is already shared across rows.

## What the ceiling is

The per-row ALU of ~20.6 ms contains ~5.5 lane instructions per weight: 2 x loads, 2 dots, a half
add, a convert, a float add.  Every alternative accumulator form measured worse.  The extract is
0.094 warp-inst per weight and is irreducible for 4-bit.

Against round 001's measured mains figure (42.4 tok/s, 60 ms/pass at 2.54 tokens) the two verified
improvements since compound to roughly **46-49 tok/s**: x1.065 measured end to end for the half
dots, and TILE=4 worth more at mains than the +1.1% it is worth on battery, because the term it
amortises is the one at a hard floor when the machine is not throttled.  That is short of 50, and
the gap is the per-row ALU, which the current kernel shape cannot reduce further.
