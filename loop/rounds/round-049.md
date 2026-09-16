# Round 049 - the GEMV was widening, not multiplying: half4 dots cut the sweep 20%

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=**+6.5% end to end, 4/4 paired** G4=**accept.sh 12/12 ACCEPTED**
- outcome: the first real win since round 002.  The fp32 inner loop spends more instructions
  widening operands than multiplying them; keeping the dequantised weights, the x loads and the
  dots in half4 removes the widening and makes the weight sweep **20.2% faster** (calibrated,
  9/10 paired rounds), worth **+6.5% end to end** on the throttled battery machine.

## Why the instruction count, not the FLOP count

Per (group, word, lane) the u4 tile kernel executed, for 3 rows x 8 weights = 24 MACs:

| ops | what |
|---|---|
| 8 | nibble extract: shift + and |
| 8 | nibble -> **float** convert |
| 8 | dequantise: `*s + bb` in float4 |
| 6 | x loads (3 rows x 2 x half4) |
| **24** | **x half4 -> float4 widen (3 rows x 8)** |
| 24 | dot in float4 |
| = ~78 | for 24 MACs |

The widen is the largest single group and it exists only because the accumulator is fp32.  Keep
everything in half4 and the same 24 MACs cost ~40: 8 nibble extracts, 8 -> half converts, 8 half4
FMAs for the scale/bias, 6 x loads (no conversion - the buffer is already half), 6 `dot(half4,half4)`
and 6 widenings of the group partial sum into the fp32 accumulator.

## What was added

- `q4_gemv_k3_u4h` - the k=3 (spec verify) kernel.
- `q4_gemv_h` - the k=1 kernel.  **Both paths had to move together**: the plain path and the spec
  path use different kernels for the same linear, and round 047 proved the 300-token `spec==plain`
  gate catches any arithmetic divergence between them.  The two halves compute
  `(float)(dot(w0,x0) + dot(w1,x1))` into an fp32 accumulator in the same group and word order, so
  they are bit-identical to each other.

The dequantised weight is now rounded to half before the dot.  That is a real numerical change, not
a free one - but the oracle was produced with mlx-lm, whose own 4-bit path dequantises in the
compute dtype, so it is not obviously further from the reference.  It passes both numerical gates
below.

## Evidence

Paired sweep, 10 rounds, `bench --tokens 3 --rows 9`, calibration arm last:

```
k3 (baseline)          (paired baseline)
k3 + u4 (16B) loads    median ratio 0.9569  calibrated 0.9335  wins 8/10
k3 + u4, 2 rows/tg     median ratio 1.1238  calibrated 1.0964  wins 1/10
k3 + u4 + half dots    median ratio 0.8179  calibrated 0.7980  wins 9/10   <- 20.2% faster
k3 (baseline dup)      median ratio 1.0250  (CALIBRATION: instrument bias)
```

End to end, `QW_SPEC=1 gen --max-tokens 400`, four pairs with the order alternating:

```
pair1  fp32 18.69  half 19.65  ratio 0.9511
pair2  fp32 18.37  half 20.06  ratio 0.9158
pair3  fp32 18.50  half 19.84  ratio 0.9325
pair4  fp32 18.62  half 19.69  ratio 0.9457
median 0.9391 -> half dots deliver 1/0.9391 = +6.5% tok/s
```

Within one run, the verify batch itself (1634 dispatches, `QW_ENCODE_TIME=1`), two rounds:
122.72 -> 118.82 ms and 123.68 -> 113.90 ms, i.e. -3.2% and -7.9%.

Gates: parity 6/6 against mlx-lm, and `tools/accept.sh` 12/12 ACCEPTED, which includes the
300-token `spec==plain` comparison and the 4-bit GEMV against the CPU reference on real weights (the
half rounding is inside that check's tolerance).

## Where this leaves the target

By the round-048 budget (26 ms weights + 19 ms ALU rows + 15 ms dispatch, = 60 ms at mains), a 20%
sweep win is ~9 ms and would land at ~51 ms = **49.8 tok/s**.  Measured on the throttled battery
machine, where the bandwidth term is 2.6x larger and the ALU term is relatively smaller, the same
change is +6.5%, which extrapolates to ~45 tok/s.  The two estimates bracket the target, and which
one holds is exactly what mains power would settle.

The same lever is not exhausted: the nibble extract + convert is now the largest remaining block
(16 of ~40 instructions), and accumulating a whole group in half before widening would remove 7 of
the 8 widenings per group.

## Follow-up in the same round: the half4 accumulator loses

`dot(half4,half4)` collapses four lanes to a scalar, twice per word per row, and a lane only owns
`n_groups/32` groups (K=5120 is 80 groups over 32 lanes), so the accumulator could in principle stay
a half4 and be collapsed once at the end.  Screened as `q4_gemv_k3_u4h4` against `u4h` in the same
paired sweep (10 rounds):

```
k3 + u4 + half dots   median 0.8319  calibrated 0.8257  wins 9/10
k3 + u4 + half4 acc   median 0.8417  calibrated 0.8354  wins 9/10
k3 (baseline dup)     median 1.0075  (CALIBRATION)
```

Element-wise accumulation is 1.2% *slower* than the scalar form it replaces: `w0*x0 + w1*x1` cannot
fuse into an FMA chain the way the reduction can.  Production keeps `u4h`; the arm stays in the bench
as a closed candidate.
