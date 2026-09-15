# Round 003 — M4 groundwork: weight-stationary multi-token GEMV

- date: 2026-09-15
- milestone: M4 (MTP / speculative decoding) — **first primitive in place**
- gates: G1=pass G2=**pass (6/6 unchanged)** G3=measured G4=pass

## Hypothesis

> A decode step is 92% weight bandwidth (14.41 GB / 518 GB/s = 27.8 ms), so the
> only way past ~33 tok/s is to serve several tokens per weight read. Speculative
> decoding with the MTP head supplies those tokens; this round builds and
> measures the kernel that consumes them.

## What was built

`q4_gemv_k` — the same 4-bit GEMV, but each threadgroup keeps its row's weight
words in registers and reuses them for `k` activation vectors
(`crates/qw-metal/src/msl.rs`), with `QLinear::encode_k` / `kernel_k` on the
model side and `qwen38 bench --tokens k` to measure it. A GPU test asserts the
multi-token output equals the single-token kernel **token by token**
(`q4_gemv_k_matches_single_token_kernel`, real weights layout, 3 tokens).

## Measured (497 linears, 14.41 GB per sweep, M5 Max)

| k | ms/sweep | equivalent tok/s | effective GB/s |
|---|---|---|---|
| 1 | 26.96 | 37.1 | 535 |
| 2 | 46.52 | **43.0** | 310 |
| 3 | 60.52 | 49.6 | 238 |
| 4 | 74.80 | 53.5 | 193 |

The linear path alone clears 40 tok/s at k=2, but the marginal cost of an extra
token is 18.8 ms — 68% of a full pass — so the amortisation is real but partial.

## What did not work (recorded so it is not retried)

Register-blocking 4 output rows per threadgroup (`#define Q4_ROWS 4`) to cut the
x re-reads from L2 **made k=2 slower (62.7 ms vs 46.5 ms)**: with several rows in
flight a lane touches several weight rows at once, and the weight loads stop
being coalesced 128-byte lines. The coalescing is worth more than the x reuse.
A correct fix has to keep one row per threadgroup and instead cut x traffic some
other way (e.g. staging x in threadgroup memory once per row block, or making the
row the *inner* grid dimension).

An earlier edit of this round also silently rewrote the single-token `encode`
grid to `out_f/4` threadgroups; the symptom was an impossible 1667 GB/s sweep and
parity dropping to 0/6. Reverted; k=1 is back to 26.96 ms / 535 GB/s.

## Budget arithmetic for the 40 tok/s target

End-to-end at k=1 is 35.9 ms/token (27.9 tok/s): 27.0 ms of weights + ~9 ms of
attention/delta-net/norms/argmax. Verifying `1+k` tokens per pass at k drafts:

```
tok/s = (1 + accepted) / (sweep(k) + k * draft_cost)
```

With one MTP draft token (k=2, 46.5 ms) and a ~2.5 ms draft step, 40 tok/s needs
an acceptance rate near 0.96 — unrealistic. Two drafts (k=3, 60.5 ms) with 0.8/0.64
acceptance gives 2.44 tokens / 65 ms = 37 tok/s. So **both** the sweep efficiency
and the ~9 ms non-linear overhead have to come down; the next rounds attack them
directly rather than hoping for a high acceptance rate.

## Next round

1. Cut the x re-reads without breaking weight coalescing (threadgroup-memory
   staging), target <= 32 ms at k=2.
2. Quantise the local bf16 `mtp.*` shard (MLX affine, group 64) and wire the MTP
   head so a real draft/verify loop can be measured end to end.
