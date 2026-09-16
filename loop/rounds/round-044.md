# Round 044 - 16-byte weight loads: the first measured sweep win, 7.0%

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=pass G4=accept.sh 12 passed 0 failed ACCEPTED
- outcome: `q4_gemv_k3_u4` wired into the model.  Paired sweep at a reproducible clock plateau:
  **median ratio 0.9304, 76/80 wins** - the first kernel change since round 040 that measures
  faster rather than slower.

## Why this variant and not the others

Round 043 settled that the sweep is 77% of a pass and that at the plateau it saturates two
throttled resources: memory bandwidth (207 GB/s for the fixed 14.41 GB weight stream) and ALU
issue (a flat 14.5 ms per extra row).  Row blocking attacks the per-row term by removing x loads
and loses at every factor.  The untested direction is the other way to buy memory-level
parallelism: **more bytes in flight per instruction**.  `Q4_GEMV_KS_U4` loads weights as `uint4`
(16 bytes = 16 nibbles) instead of `uint` (8 bytes).

Measured at the plateau (30 s burn-in first, 80 rounds, order alternating, one process):

```
k3 (baseline)        grid_rows=1  (paired baseline)
k3 + u4 (16B) loads  grid_rows=1  median ratio 0.9304  wins 76/80  (faster)
k3 + u4, 2 rows/tg   grid_rows=2  median ratio 1.0495  wins 14/80  (slower)
```

76 of 80 rounds is not a marginal signal, and it is the strongest result this session.  It also
confirms the read of round 043: adding rows on top of u4 loses again, so the two effects are
additive negatives, not substitutes.

## Correctness

The u4 kernel changes the accumulation order within a group, so parity was checked before
anything else:

```
parity: 6/6 cases
tools/accept.sh: 12 passed, 0 failed  ACCEPTED
```

## What it is worth

The sweep is 77% of a pass, so 7.0% off the sweep is ~5.4% off a pass: on battery 18.4 -> ~19.4
tok/s, and against the mains baseline of 42.4 tok/s a projected ~44.8 - not the target, but the
first real movement toward it since the 40 tok/s goal was met.

## Not yet confirmed end to end

The interleaved steady-state A/B was set up (binaries at `/tmp/qwen38_base` and
`/tmp/qwen38_u4`) but not completed: after ~40 minutes of continuous load the machine was reading
8.74 tok/s steady-state against 18.4 earlier in the same session - a 2x swing that is exactly why
the interleaved protocol exists, and why a same-state pair is required rather than two sequential
runs.  The A/B is the first thing to run next round, on mains if available.  (Two harness
failures along the way were both my own extraction regex, not the engine: the steady-state line
carries `tokens/pass` between the seconds and the ms/token field, and `tail -4` cuts stderr
diagnostics that are flushed before the buffered stdout text.)

## CORRECTION (round 045): the 7.0% was an uncalibrated instrument, the real win is ~1%

Round 045 added a duplicate of the baseline to the paired sweep as a calibration arm - the same
kernel, same grid, measured in the same rounds.  It read **0.9649** and **0.9559** against the
primary baseline in two independent runs, i.e. the method itself prefers whatever is not the
baseline by ~3.5%.  Order alternation does not cancel it.

Correcting the u4 number with that bias:

```
raw        0.9535   1.2% "faster"  after dividing by the 0.9649 bias
raw        0.9521   0.4% "faster"  after dividing by the 0.9559 bias
```

and the end-to-end interleaved A/B on the real model (4 pairs, orders alternating) gives
**0.9831, i.e. +1.7% tok/s**, which agrees.  **So `q4_gemv_k3_u4` is worth about +1.7%, not the
7.0% claimed above.**  The change is kept - it is a real, reproducible improvement and it costs
nothing - but the claim in this file is wrong and the calibrated column is the one to read.
