# Round 005 — the GEMV structure is exhausted; the tiled GEMM is the answer

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6 unchanged)** G3=no regression G4=pass

## Probes run this round (both cheap, both refuted)

**Probe 1 — hoist the x slice out of the word loop** (the `xv[4][8]` register
variant, reachable as `--rows 1` before this round's rewrite). If the k>1 cost
were the k extra x-loads inside the inner loop, this should halve them.

```
k=2  rows=0 (round-3 kernel) 46.52 ms | rows=1 (x hoisted) 63.12 ms   <- 36% WORSE
k=3  rows=0                 60.36 ms | rows=1             84.39 ms   <- 40% WORSE
```

**Probe 2 — fetch the 4-bit words as 16-byte `uint4` loads** instead of eight
4-byte loads per group (`q4_gemv_kr`, rewritten this round; `--rows 1`).

```
k=1  rows=0 26.81 ms | rows=1 27.08 ms
k=2  rows=0 47.15 ms | rows=1 46.31 ms   (within run-to-run noise)
k=3  rows=0 60.49 ms | rows=1 60.09 ms
```

Both variants keep 6/6 parity; neither is used by the engine's k=1 path.

## What the three structures together tell us

| structure | k=2 | verdict |
|---|---|---|
| one row/threadgroup, k in registers (round 3) | 46.5 ms | best |
| R rows/threadgroup (round 4, R=2/4) | 60.9-62.8 ms | worse |
| x hoisted to registers (this round) | 63.1 ms | worse |
| uint4 weight loads (this round) | 46.3 ms | wash |

Four structural variants, none better than the round-3 kernel. So the ~19 ms
marginal cost of a second token is **not** x traffic, **not** weight-load width,
and **not** address arithmetic. What is left is the shape of the kernel itself:
5120 threadgroups x 32 threads with ~2.5 group iterations per lane is very little
work per thread, and at k=2 each thread's dequantise->FMA chain doubles with no
extra threadgroups to hide it. A GEMV-shaped kernel has no room to fix that; more
parallelism per weight byte is needed, which means cooperation between threads
over a *tile* of tokens — i.e. a real tiled GEMM, exactly what the round-4 report
identified as the structural answer.

That kernel is also M2's prefill path, so the work is not speculative: it is
required by the objective either way.

## Next round

Write `q4_gemm_t`: a tiled 4-bit GEMM over a tile of T tokens (T = 8..32), with
x staged in threadgroup memory, weights dequantised per K-chunk, and fp32
accumulation, targeting the same 27-28 ms per weight pass at T = 8 (which would
put speculative verification at 40+ tok/s even with a mediocre acceptance rate).
Verify it against the k=1 GEMV output for every token in the tile.
