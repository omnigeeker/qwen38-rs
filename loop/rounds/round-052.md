# Round 052 - the marginal row is 8 ms, not 20.6: u4hx ships

- date: 2026-09-17
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=see below G4=**accept.sh 12/12 ACCEPTED**
- outcome: a measurement-method correction that invalidates round 051's ceiling, and the x-load
  widening integrated.

## The round 051 ceiling was an artefact

Round 010 fitted `GEMV(k) = 39.5 + 20.6k` by subtracting a TILE=3 verify (114.58 ms) from a TILE=4
verify (135.43 ms) **measured in different runs**, and round 051 built its "the per-row ALU is a
clock-invariant wall, the target is out of reach" conclusion on it.  Measured instead with the two
binaries interleaved in one thermal state, averaging every verify batch in each run:

```
TILE=4 verify (1858 dispatches)  50.16  49.62  66.59*  ms
TILE=3 verify (1634 dispatches)  41.96  41.98  42.00   ms
* third pair is a thermal spike; the TILE=3 column is reproducible to 0.04 ms
=> one more verified row costs ~8 ms, not 20.6
```

The same trap produced round 010's "non-GEMV is flat in rows" and round 011's "the stream is 4.2x
throttled while the rows are not": a k=1 forward inside the prefill sits on a different point of
the throttle ramp from a verify batch later in the run, so **no prefill-phase number may be
subtracted from a decode-phase number**.  Every such subtraction in rounds 009-011 is void.

What survives: the cool-state k=1 forward streams 14.41 GB in 26 ms of its 31 - **554 GB/s, the
DRAM peak** - so at the cooled state the memory path is already at the hardware limit, and round
001's measured mains pass (60 ms, TILE=3) is consistent with today's cooled TILE=3 verify (42 ms)
plus drafts and tail.  That makes the cooled state a defensible mains proxy, and it means the
earlier "46-49 tok/s" compounding stands rather than the 31 tok/s the round-051 wall implied.

A caution on the cool protocol: a ten-token decode after `QW_COOL_SLEEP` reports only 2.00
tokens/pass, so its 26 tok/s is not the sustained figure and must not be compared with a 400-token
mains run.

## u4hx: one 16-byte x load instead of two 8-byte ones

The x loads dominate the kernel's memory operations - per (group, word) the row loop issues two
8-byte `half4` loads per row, while the weights need one 16-byte load for the whole pass, 28.8 G
loads against 3.6 G.  The eight halves a lane needs are sixteen contiguous bytes, so one `uint4`
covers them and the two `half4`s are `as_type` reinterprets.  **Same bits, same order of
operations** - which is why it carries no numerical risk, unlike every other kernel change tried:

```
paired sweep, 12 rounds    u4h   median 0.7730  calibrated 0.8373  5/6
                           u4hx  median 0.7478  calibrated 0.8100  5/6     (+3.4%)
k=1 forward (1186 disp)    ratios 0.9796 / 0.9939 / 1.0080             (~1%)
cool-state decode          25.82 / 26.01  against 26.20 / 25.52        (a wash)
```

The end-to-end instrument cannot resolve a 2% effect on a 10-token decode, so the decision rests on
the controlled screen: +3.4% on the GEMV, which is 86% of the verify.  `q4_gemv_hx` widens the k=1
kernel the same way - it had neither widening - and that is the path the three MTP draft steps and
the whole prefill take.

Verified: parity 6/6, `spec==plain` byte-identical over 300 tokens, accept.sh 12/12 ACCEPTED.
