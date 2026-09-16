# Round 041 - row blocking was never implemented; implementing it correctly says no

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte (300 tokens), plain identical to round 037 G4=pass
- outcome: **no net change retained**.  Row blocking was implemented, measured, and rejected;
  the model is back to the round-040 kernel.  What this round buys is the reason *why*, plus a
  correction to two earlier claims.

## The marginal row is not a FLOP wall - correcting round 040

Round 040 concluded the three-row GEMV was near-optimal because a marginal row looked like
~16.8 TFLOP/s.  That used `bench --rows 1`, i.e. the right (unrolled) kernels, but only two
points.  Four points on the same harness:

```
k=1  27.94 ms
k=2  29.14 ms   +1.20
k=3  32.97 ms   +3.83
k=4  38.97 ms   +6.00
```

The marginal cost is **superlinear**, which a FLOP wall is not.  At k=3 the sweep moves
86.4 GFLOP in 32.97 ms = 2.6 TFLOP/s, an order of magnitude below what the k=2 marginal row
implies the chip can do.  So the sweep is memory/latency limited, and there was room to look for.

## `q4_gemv_kr` never implemented row blocking

`crates/qw-metal/src/msl.rs:145` is `(void)R;`.  The `R` parameter - the entire point of the
kernel - is discarded.  `q4_gemv_kr` is just `q4_gemv_k` with 16-byte weight loads.  So round
039's "row-blocking rows>1 was falsified" note rested on a kernel that does not block, and the
lever had never actually been tried.  Worth stating plainly: that was a wrong conclusion, not a
wrong measurement.

## So: implement it properly

Added `Q4_GEMV_KS_R(NAME, NK, R)`: `R` output rows per threadgroup, x loaded once per group and
held across all R rows, so the x load traffic - which is `out_dim` times the input size, far more
than the 4-bit weights - is divided by R, while each row's weights are still read exactly once.
The accumulation order per output row is bit-identical to `q4_gemv_k3`, so nothing numeric moves.
Also `Q4_GEMV_KS_U4` (16-byte weight loads) and a `bench --rows 9` mode that round-robins
candidates against the baseline inside one process.

In the isolated sweep it looked like a win, and then refused to stay won:

```
run A: R2 0.9396 (36/40)  R3 0.9068 (37/40)  R4 0.8866 (35/40)
run B: R2 0.9985 (21/40)  R4 0.9516 (32/40)  R6 1.3283 (1/40)  R8 1.2058 (2/40)
run C: R2 0.9422 (56/60)  R3 0.9121 (57/60)  R4 0.8734 (59/60)
run D: R2 1.0660 (11/60)  R3 1.0439 (17/60)  R4 1.0340 (18/60)
```

Runs C and D are the same binary, same command, ~40 s apart, and disagree in sign with a strong
signal either way.  R6/R8 are genuinely bad (they spill), and their inclusion poisons the round,
but that does not explain C vs D, which contain only R2/R3/R4.

**End to end, which is the measurement that counts, row blocking loses.**  8 tightly interleaved
pairs against the round-040 binary, alternating order:

```
0.9823  1.7597  1.0465  1.1875  0.9040  1.0559  1.0478  1.0483
median 1.0481, 6 of 8 slower
```

Every pair where the baseline itself was not an outlier (58-66 ms/token) has R=4 slower by
4-19%; the two pairs where R=4 "won" are the two where the baseline read 120.9 and 129.9
ms/token.  Reverted.  The likely reason it helps in the sweep and hurts in the model is that the
sweep reuses **one** input buffer for all 497 linears, so x stays cache-hot and the traffic row
blocking removes is understated, while in the model the same kernel also has to cover linears
with tiny `out_f` - the GDN a/b projections are 48 rows - where blocking leaves the grid nearly
empty.

## The measurement environment, stated plainly

The machine is on battery and in Low Power Mode for this whole round:

```
Now drawing from 'Battery Power' ... discharging;  powermode 1
```

Under that, the *same binary doing the same sweep* read 27.9, 33.0, 58.5, 70.6 and 113.5 ms
across this session, and a single variant's samples inside one run spanned 65 to 294 ms.  Two
consequences:

1. **Absolute tok/s from this session cannot be compared to anything**, including the 40 tok/s
   baseline and any future 50 tok/s claim.  A 50 tok/s verification needs mains power with Low
   Power Mode off.
2. **Relative comparisons need >= 6 tightly interleaved pairs at model level.**  The
   best-of-N sweep that looked so convenient is not trustworthy here: it flipped uint4 from
   1.097 (slower) to 0.558 (faster) to 0.978 across three runs, and the paired sweep flipped R4
   in the same way.  Only the interleaved full-model A/B produced a consistent sign, and it is
   what this round was decided on.

By the same standard, round 040's 2.0% fold win (4/4 pairs) is the strongest evidence in this
session, but it is 4 pairs of a ~2% effect - not enough to settle a 50 tok/s question either.
