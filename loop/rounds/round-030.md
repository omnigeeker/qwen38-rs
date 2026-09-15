# Round 030 — the embedding dequantiser was converting the whole table, every token

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6 + byte-identical) G3=**big win** G4=pass

## The bug

`embed_row` needs 80 scales and 80 biases for one token's embedding row.  It was
getting them like this:

```rust
let scales = self.embed_s.as_bf16_f32();   // the ENTIRE table
let biases = self.embed_b.as_bf16_f32();
let scales = &scales[row * groups..(row + 1) * groups];
```

`embed_s`/`embed_b` are `[vocab][hidden/64]` = 248320 x 80 = **19.9M entries each**, so
every call converted ~40M bf16 values and allocated ~160 MB of `f32` - to use 160 of
them.  And `embed_row` is on the hot path of **both** decoding and drafting, because
`set_token`/`set_tokens` and `mtp_step` all go through it.

## The fix

Read the row's metadata straight out of the tensor bytes: 80 `bf16_at` lookups into
`embed_s`/`embed_b`, no allocation, no table-wide conversion.

## Measured

| | before | after |
|---|---|---|
| draft, ms/pass | 6.35 | **2.60** |
| verify, ms/pass | 52.49 | **45.15** |
| plain decode | 24.06 tok/s | **27.5 tok/s** (+14%) |
| speculative decode | 32.29 tok/s | **35.61 tok/s** (+10%) |

The draft is now at its bandwidth floor (~1.1 GB of head weights at ~500 GB/s ~= 2.2 ms),
and the verify pass got faster too because it pays one `set_tokens` per pass.  Oracle
parity stays 6/6 and speculative output stays byte-identical to greedy.

The speculative decoder is now at **35.6 tok/s**, 89% of the 40 tok/s target.

## What is left, precisely

Per pass the loop accounts for 45.15 (verify) + 2.60 (draft) = 47.8 ms, but the measured
pass is ~51.9 ms.  The missing ~4 ms is host-side: the loop reads back `logits_row(0)`
and `logits_row(1)` - 497 KB of f16 each - converts 248K values to f32 and argmaxes on
the CPU, three times per pass.  A GPU argmax writing one `u32` per row would cut that to
a few bytes and is the next change; after it the remaining work is the dispatch-chain
fusion (rmsnorm/ewise_add/silu_mul/the GDN row loops).

Gates: fmt clean, clippy `-D warnings` clean, tests pass, parity 6/6, spec byte-identical.
