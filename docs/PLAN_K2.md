# Implementation plan: the k=2 engine path

Status: **the only remaining lever** for the 40 tok/s objective. Every other
hypothesis has been measured and eliminated (see `docs/STATUS.md` §3).

## Why it is the only lever

| | k=1 | k=2 |
|---|---|---|
| 497-linea weight sweep | 26.87 ms | **28.68 ms for two tokens** |
| per-token linear cost | 26.87 ms | **14.34 ms** |
| end-to-end decode | 36.04 ms = 27.75 tok/s | ~47 ms / 2 tokens = **42 tok/s** at alpha 1.0 |

The 1-token-per-pass ceiling is 33 tok/s (14.41 GB / 519 GB/s). Round 6 showed the
second token is nearly free in the linear part; the remaining ~9 ms/token of
non-linear work is what the first cut does not yet amortise.

## The change, in order of value

### Step 1 — linear layers, k=2 (this is where the win is)

Every activation buffer becomes a two-row tile: `[2][rows]`, row `r` at byte
offset `r * rows * 2`.

* `QLinear::encode` → `QLinear::encode_k(..., k = 2)`.
* Kernel: **`msl::K_Q4_GEMV_K2`** (the compile-time specialisation). Do *not* use
  the runtime-`k` `q4_gemv_k`: rounds 3-5 lost three rounds to the resulting
  thread-local-memory spill (46.6 ms vs 28.6 ms).
* Grid stays `(out_f * 32, 1, 1)`, threadgroup `(32, 1, 1)`; `y[(t * out_f) + row]`
  is already the layout the kernel writes.

### Step 2 — non-linear layers, looped twice (no new kernels)

For the first cut, keep every existing single-row kernel and issue it twice with
`buf_offset(..., r * rows * 2)` instead of `buf(...)`. That doubles the non-linear
dispatch count but requires no new MSL and no layout reasoning.

Ordering constraint: the gated-delta-net recurrence and the conv window are
**sequential** — step for position `p` must complete before step `p+1`. Attention
is not: both `kv_append`s happen first, then the scores kernel runs twice with
`t = p+1` and `t = p+2` (the kernel already takes `t` per query, which is exactly
the causal mask).

### Step 3 — verify

* `bash loop/verify.sh` must stay **6/6** (this is non-negotiable; it has been
  green for every round since round 2).
* New test: after `forward(p)` and `forward2(p+1)`, the row-1 logits must match a
  plain `forward(p+1)` run: identical argmax, `max|delta logit| < 0.5`. Bit
  equality is not expected — the k=2 kernel accumulates in a different order —
  but the *token* must not change.
* Keep `forward` and `forward2` side by side so parity can always be re-checked
  against the single-token path.

### Step 4 — then the MTP draft

With k=2 working, the draft is one MTP step: `mtp.fc` (`2*5120 -> 5120`) plus the
single full-attention decoder layer, on fp16 weights (~0.42 B params, ~2.5 ms).
Quantise `models/Qwen3.8-27B-bf16-mtp/` with MLX affine (group 64) and reuse
`q4_gemv`. Note the norm convention recorded in `docs/M1_NOTES.md`: the bf16
repo's `mtp.layers.0.input_layernorm / post_attention_layernorm / q_norm` need
`+1`, `mtp.norm` does not.

## Expected result

```
k=2 pass = 28.68 ms (linear) + ~18.3 ms (non-linear, 2 x 9.17, first cut)
         + ~2.5 ms  (MTP draft)
         = ~49 ms per (1 + alpha) accepted tokens
   alpha = 1.0 -> 40.8 tok/s
   alpha = 0.8 -> 36.7 tok/s
```

So step 2 must eventually be fused (one 2-row norm/gate kernel instead of two
dispatches) to pull the non-linear part back to ~9 ms, which gives ~40 ms per
`1+alpha` tokens and 45 tok/s at alpha 0.8. Both steps are ordinary work; step 2's
fusion is where the remaining few milliseconds live.
