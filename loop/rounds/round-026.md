# Round 026 — the GEMV is NOT at the hardware limit, but three shapes failed to capture it

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6) G3=no change (all three experiments reverted) G4=pass

## First, a process failure worth recording

Round 25 ended by proposing "R output rows per threadgroup" as the lever.  That idea
had **already been implemented and measured in round 004** (rows 1/2/4 = 46.58 / 60.88
/ 62.84 ms - rows=1 wins) and the uint4-widening follow-up in round 005 was recorded
as a wash.  Both are in `loop/state.json` and `loop/rounds/`, and I re-derived them
from first principles instead of reading them first.  The cost was a round.  Grep the
loop state before proposing a lever.

## The measurement that matters

The engine's kernel achieves ~360 GB/s (14.41 GB in 40 ms).  A pure streaming read of
the same volume on this machine:

```
14 GB streaming read:  47.1 ms = 298 GB/s    (cold)
14 GB streaming read:  28.1 ms = 499 GB/s
14 GB streaming read:  28.8 ms = 486 GB/s
```

**~490-500 GB/s is real**, so the GEMV's 360 GB/s is a genuine 1.4x gap, not a hardware
ceiling.  That single number justifies continuing to work on the kernel - and it is
the most useful thing this round produced.

## Why the traffic model says the gap is activations

Each lane fetches 8 activations (16 bytes) to use 8 four-bit weights (4 bytes): the
activation:weight byte ratio is 4:1, and unlike the weights, `x` is re-read by every
one of the `out_f` output rows.  That is consistent with the k=2 measurement (the
second row costs ~8 ms even though the weights are read once for both) and with the
~360 GB/s plateau: the DRAM stream is a fifth of the load traffic.

## Three shapes tried, all reverted

| shape | result |
|---|---|
| shared-memory staged x, 4 rows/threadgroup | bit-exact, **2.2x slower** (109 vs 49 ms) |
| coalesced (one lane per word, not per group) | bit-exact, **identical** (41.0/49.2 vs 40.6/48.9 ms) |
| R rows per *lane*, x fetched once per R weight rows | **wrong results** (zeros) and 5x slower |

The coalesced variant is worth a note for the future: it proved the access pattern is
*not* the limit.  Making every warp load a single 128-byte transaction changed nothing,
and it stayed bit-exact, so it is a clean negative.

The third shape is the one that should work on paper - no barriers, no shared memory,
weight loads still coalesced, activation traffic divided by R - so it is worth one more
attempt with two specific suspicions to check first:

* `float acc[R][NK]` indexed inside runtime loops needs both loops genuinely unrolled;
  a 5x slowdown is the signature of the array spilling to thread-local memory.  Consider
  R separate scalars, or `#pragma unroll` on a fully literal loop nest.
* a collective (`simd_sum`) must be executed by every lane.  The first version called it
  inside an `if (lane == 0)` branch; that was fixed and the kernel was still wrong, so
  something else - most likely the same spill - is also wrong.

## Where the objective stands

The k=2 pass measures 40.7 tok/s, so two accepted tokens per weight sweep already reach
the target; the problem is that a real MTP (alpha = 56%) plus its ~5.5 ms draft turns
that into ~28 tok/s.  Both ends have to move:

* the sweep: 360 -> ~490 GB/s would put a k=2 pass near 33 ms (60 tok/s of pass
  throughput) and make the extra rows much cheaper;
* the head: alpha 56% -> 85%, and the draft ~5.5 ms -> ~2.5 ms (its bandwidth floor is
  1.6 ms for 841 MB).

Gates: fmt clean, clippy `-D warnings` clean, 13 test binaries pass, parity 6/6,
`forward2` bit-exact.  Nothing from this round's experiments remains in the tree.
