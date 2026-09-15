# Round 019 — `forward2` exists, compiles, and is wrong in exactly one place

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6, `forward` untouched)** G3=no change G4=pass

## What was built

`forward2(pos)` — two consecutive positions in one weight pass — generated
mechanically from the verified single-token `forward` rather than retyped, so the
dispatch order is preserved by construction:

* `TILE`-wide projections via `encode_k(..., TILE)` with the compile-time
  `q4_gemv_k2` (added to `Kernels` next to `q4_gemv`);
* every per-row operation re-issued once per row through
  `buf_offset(i, buf, row * ROW_BYTES)`, with `pos as i32` -> `(pos + row) as i32`
  and `.scalar(3, t)` -> `.scalar(3, t + row as i32)` for the two attention kernels;
* the gated-delta-net arm kept as **one** per-row loop (its `in_proj_qkv` writes the
  shared four-row conv window), with only `in_z/in_b/in_a` hoisted to k=2 before it
  and `out_proj` hoisted after;
* final norm per row + `lm_head.encode_k(..., TILE)`, so both rows get logits;
* `set_tokens(&[u32])` (embeds one token per row; `set_token` now delegates to a new
  private `embed_row`), `logits_row(row)`, `peek_row1(n)`;
* a diagnostic behind `QW_K2_CHECK=1` that compares `forward2` against two ordinary
  single-token forwards, including a self-pair run.

## The measurement

```
k2check: row0 argmax ref=2614 got=2614  max|d|=2.6724
k2check: row1 argmax ref=1898 got=513   max|d|=8.8164
k2check: self-pair row0 argmax ref=2614 got=10211 max|d|=17.6094
k2check: layer 0 ref absmax=19.89062 got absmax=19.89062
k2check: layer 1 ref absmax=29.23438 got absmax=29.23438
k2check: layer 2 ref absmax=43.00000 got absmax=43.00000
k2check: layer 3 ref absmax=49.90625 got absmax=48.62500
```

Read: `set_tokens` is exact (row 0 equals `set_token`'s `x` bit for bit, so the
embedding path is cleared). **Layers 0-2 - all gated-delta-net - match exactly**, so
the tile layout, the k=2 kernels, the hoisted GDN projections, the MLP and the
residual adds are all correct. **Layer 3 is the first full-attention layer, and it
diverges there.** The fault is therefore localised to the Full arm: `q_norm`,
`k_norm`, `rope`, `kv_append`, `attn_scores`, `attn_out`, `gate_mul`, `o_proj`.

The self-pair case (both rows the same token, where row 0 must reproduce the
single-token forward *exactly*) failing harder than the real case is further
evidence of the same thing: row 0's own path is wrong, not cross-row interference.

## Where the next round starts

The Full arm was transformed into *many* small row loops because `b.barrier()`
flushes a run, so each of `q_norm`, `k_norm`, `rope`, `kv_append`, `scores`, `out`,
`gate_mul` became its own `for row` loop. Order is preserved for row 0, so the
suspect is a buffer whose row stride in `TILED` is wrong - `qg` (`nh * hd * 4`),
`scores` (`nh * max_t * 4`) and `attn_*` (`nh * hd * 2`) are the only strides the
GDN layers never exercise. Check those three first.

`forward` and the 6/6 parity gate are untouched; `forward2` is unused by the engine
and reachable only through `QW_K2_CHECK=1`.
