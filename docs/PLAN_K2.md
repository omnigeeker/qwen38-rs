# Plan v2: two tokens per weight pass, quantified

Revision 2 (round 18) after reading the whole layer body of `forward()`. The v1
plan was right about the shape and wrong about the arithmetic: it assumed the
per-token fixed cost would be amortised for free. It is not. This is the corrected
plan, with the numbers that change what has to be built.

## What the layer body actually contains

`forward()` (runner.rs:467) is one flat dispatch sequence per layer:

```
pre-norm(x -> h)                                     per row
Full: q, k, v projections (QLinear::encode)          LINEAR
      q_norm, k_norm, rope, kv_append                per row, pos-dependent
      attn_scores, attn_out, gate_mul(sigmoid)       per row, causal
      o projection                                   LINEAR
Gdn:  in_proj_qkv -> writes window[3*conv_dim]       LINEAR, but into the shared
                                                     conv window (NOT a tile)
      in_z, in_b, in_a                               LINEAR
      copy window->conv_hist, conv1d, copy back      per row, sequential
      rmsnorm (q,k, no weight), gdn_step, rmsnorm_gated  per row, sequential
      out projection                                 LINEAR
residual (ewise_add)                                 per row
post-norm                                            per row
gate, up projections                                 LINEAR
silu_mul                                             per row
down projection                                      LINEAR
residual, optional bf16 round, optional debug copy   per row
```

## The reorder, and why it is not optional

Because four projection groups read a 2-row tile, the per-row segments between
them must be wrapped in `for row in 0..TILE`. The gated-delta-net branch cannot be
syntactically wrapped: `in_proj_qkv` writes into the shared 4-row `window`, so
running row 1 before row 0's conv would clobber it. Giving the window a fifth row
is the fix, and that is a layout change, not a loop.

## The corrected arithmetic (measured inputs)

```
W  = 26.87 ms   k=1 weight sweep (536 GB/s)
W2 = 28.68 ms   k=2 specialised sweep  -> 1.067x for two rows in one pass
N  =  9.08 ms   per-token non-linear (35.95 - 26.87), of which ~1282 dispatches
lm_head = 636 MB of the 14.41 GB (4.4%)  -- read once per row unless batched
```

| scheme | 2 tokens | equivalent tok/s |
|---|---|---|
| two k=1 passes (today) | 71.9 ms | 27.8 |
| forward2, linear k=2, GDN per-row, non-linear per-row | ~51.8 ms | **38.6** |
| + lm_head k=2 | ~46.1 ms | **43.4** |
| + non-linear fused to one dispatch per op for both rows | ~40.9 ms | **48.9** |

The gap between rows 3 and 4 is the whole lesson of v1: **two per-row dispatches
cost two times the dispatch overhead**, and the dispatch overhead is what the
9.08 ms mostly is (rounds 7-12 eliminated encoder teardown, encoder churn, ~8000
objc bindings and the host logits readback; what remains is GPU-side scheduling of
~1282 small kernels). So batching the arithmetic without batching the dispatches
buys less than it looks like.

## What has to be built, in order

1. **`forward2(pos, &mut [u32; 2])`** with the row-wrapped body above: projections
   via `encode_k(..., TILE)` and `msl::K_Q4_GEMV_K2`, everything else per row via
   `buf_offset(i, buf, row * ROW_BYTES)`. Keep `forward` untouched so parity can
   always be re-checked. Verify: row-1 logits match a plain `forward(pos+1)`
   (identical argmax; `max|delta| < 0.5` is expected from accumulation order).
2. **lm_head + final norm for both rows** (single `encode_k` at the end) and, for
   speculative decoding, both rows' logits are needed anyway.
3. **2-row non-linear kernels**: add a row-stride scalar to `rmsnorm`, `rmsnorm_ws`,
   `silu_mul`, `ewise_add` and `gate_mul`, and give `attn_scores`/`attn_out` two
   query positions. This is what removes the second dispatch, and it is what turns
   38.6 into 48.9 tok/s.
4. **The MTP head**: quantise the local bf16 `mtp.*` shard (MLX affine, group 64),
   `mtp.fc` + the single full-attention layer, ~2.5 ms draft. Note the norm
   convention in `docs/M1_NOTES.md`: the bf16 repo's
   `mtp.layers.0.input_layernorm / post_attention_layernorm / q_norm` need `+1`,
   `mtp.norm` does not.

## Expected result

```
verification pass, 2 rows          ~41 ms  (after step 3)
MTP draft                          ~2.5 ms
-----------------------------------------------
alpha = 1.0  -> 2 tokens / 43.5 ms = 46.0 tok/s
alpha = 0.8  -> 1.8 tokens / 43.5 ms = 41.4 tok/s
alpha = 0.6  -> 1.6 tokens / 43.5 ms = 36.8 tok/s
```

40 tok/s therefore needs step 3 as well as step 1 - a mediocre draft alone will not
carry it, because the per-token non-linear work has to be shared across the two
rows, not just paid twice.

## Round 21 measurement

`QW_K2_CHECK=1` timing at a fixed position, twenty passes each:

```
k=1 39.85 ms per pass  (25.1 tok/s)
k=2 50.03 ms per pass  (25.01 ms/token, 40.0 tok/s)
```

The two-row body costs 1.255x the one-row body, so one weight sweep now buys 1.59x
the tokens.  `forward2` is bit-exact on both rows.  What remains is a draft model
for row 1 (the MTP head), and the half of the dispatch count that is still issued
once per row (step 3 of this plan).
