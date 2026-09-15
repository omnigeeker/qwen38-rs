# Round 004 — ruling out row blocking for the k>1 sweep

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6 unchanged)** G3=measured G4=pass

## Hypothesis under test

Round 3 left one number unexplained: one weight pass costs 27.0 ms at k=1 but
46.6 ms at k=2, i.e. the second token costs 72% of a full pass instead of
roughly nothing. The suspected cause was **x re-reads from L2**: with one
threadgroup per output row, the whole x vector is read once *per row*, so x
traffic is ~4x the weight traffic. Blocking R output rows per threadgroup with
the x slice held in registers should divide that traffic by R.

Implemented as `q4_gemv_kr`, a copy of `q4_gemv_k` with `R` as a **runtime
scalar**, so the blocking factor can be swept with one build
(`qwen38 bench --tokens k --rows R`; R=1 must reproduce `q4_gemv_k`).

## Result: the hypothesis is wrong, decisively

| k | rows=1 | rows=2 | rows=4 |
|---|---|---|---|
| 2 | **46.58 ms** (42.9 tok/s) | 60.88 ms (32.9) | 62.84 ms (31.8) |
| 3 | **60.42 ms** (49.7 tok/s) | 82.09 ms (36.5) | 83.74 ms (35.8) |

Row blocking is *strictly worse* at every point, and R=2 already loses 31% at
k=2. So the extra cost at k>1 is not L2 x traffic, or at least the traffic saved
is worth less than what the blocking costs (several weight streams in flight per
lane, more address arithmetic, higher register pressure). The k-blocked
one-row-per-threadgroup kernel from round 3 stands as the best known variant.

## Where the k>1 time really goes (next probes)

The remaining candidates, in order of suspicion:

1. **The x load inside the unrolled word loop.** Per weight word the kernel does
   k x-loads (two `half4` loads each) that are not needed for the weight, so at
   k=2 the instruction count per weight byte doubles. Hoisting the x slice for
   the group *outside* the word loop (a small, k-independent 16B/lane) would
   halve the x-load count without adding weight streams — unlike row blocking.
2. **Redundant work across linears.** Every linear re-derives the same x slice
   layout; a fused kernel that consumes one x row for several weight rows of the
   *same* linear (not different rows of the same kernel) is what vLLM/llama.cpp
   do (they batch the matmul, not the GEMV).
3. **The real answer may be a batched GEMM**, i.e. treat speculative
   verification as a small prefill (T=k) with `q4_gemm_t`, where x reuse is
   structural: each weight tile is loaded once for a whole tile of tokens and
   the accumulation is over K only.

## State

Rounds 1-3 stand: 6/6 token-for-token parity with mlx-lm, 27.9 tok/s end to end,
43 tok/s equivalent for the k=2 weight pass. This round adds no speed, but it
removes a plausible direction with a measured, reproducible counter-example
instead of another guess, and it leaves the A/B switch in the CLI.

## Next round

Implement (1) — hoist the x slice out of the word loop — and measure; if the
gain is small, go straight to (3), the tiled `q4_gemm_t` prefill kernel, which
M2 needs anyway and which is the structure that actually makes k tokens cheap.
