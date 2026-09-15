# Round 006 — the k-token sweep is fixed: two tokens for the price of one

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6 unchanged)** G3=**big win** G4=pass

## The bug behind rounds 3-5

Rounds 3-5 tried four kernel *structures* to make a multi-token sweep cheap and
all of them failed to beat the round-3 kernel. The real cause was not structural
at all: `q4_gemv_k` takes the token count `k` as a **runtime** scalar, so
`float acc[4]; ... acc[t] += ...` is dynamically indexed and the accumulators were
spilled to thread-local memory — every single accumulate became a local-memory
round trip, and the more tokens, the worse it got.

Fix: specialise the kernel on the token count with a macro, so the token loops
have literal bounds and fully unroll, keeping every accumulator in a register
(`Q4_GEMV_KS(q4_gemv_k2, 2)` / `k3` / `k4`, same ABI as `q4_gemv_k`). No structure
change, no extra memory traffic, ~30 lines of shader.

## Measured (497 linears, 14.41 GB per sweep)

| k | runtime k (round 3) | specialised | speedup |
|---|---|---|---|
| 2 | 46.63 ms (42.9 tok/s) | **28.64 ms (69.8 tok/s)** | 1.63x |
| 3 | 60.24 ms (49.8 tok/s) | **32.34 ms (92.8 tok/s)** | 1.86x |
| 4 | 74.64 ms (53.6 tok/s) | **38.41 ms (104.1 tok/s)** | 1.94x |

At k=2 a full weight pass over two tokens now costs 28.6 ms versus 26.8 ms for
one token: the second token is very nearly free, which is exactly the property
speculative decoding needs. Effective bandwidth at k=2 is 503 GB/s (97% of the
measured 519 GB/s device peak) *while doing twice the arithmetic*.

New test `q4_gemv_k_specialisations_match_single_token_kernel`: k2/k3/k4 must
equal the single-token GEMV token by token on real 4-bit weight layout (8 GPU
tests green).

## What this means for the 40 tok/s target

Verification of `1+k` tokens per pass, with a ~2.5 ms MTP draft step:

```
k=2: 28.6 + 2.5 = 31.1 ms for (1 + alpha) tokens   -> 48 tok/s at alpha=0.5
                                                     58 tok/s at alpha=0.8
```

Even with a mediocre acceptance rate the linear path now clears 40 tok/s, before
any of the non-linear work (attention / delta net / argmax, ~9 ms per k=1 step)
is optimised. The remaining risk moves from the kernels to the MTP head itself,
which is the next round's work.

## Next round

Quantise the local bf16 `mtp.*` shard (MLX affine, group 64) and wire the MTP
head, then measure a real draft/verify loop end to end at k=2 — with the 6/6
prefill-oracle parity re-checked every round.
