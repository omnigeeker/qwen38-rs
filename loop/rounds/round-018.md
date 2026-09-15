# Round 018 — the k=2 plan re-priced, and the number that changes it

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6, untouched) G3=no change G4=pass

This round read the **entire** layer body of `forward()` (runner.rs:467-755) to
price the refactor line by line. It changes the plan materially, so it is written
up in `docs/PLAN_K2.md` as revision 2 rather than attempted blind.

## The correction

v1 assumed that putting two tokens through one weight pass would amortise the
per-token non-linear cost. It does not, if the per-row operations are simply run
twice: the ~9.08 ms that is not the weight sweep is mostly **the GPU-side cost of
~1282 small dispatches** (rounds 7-12 eliminated every host-side explanation), so
two per-row passes cost two times that overhead.

| scheme | 2 tokens | equivalent tok/s |
|---|---|---|
| two k=1 passes (today) | 71.9 ms | 27.8 |
| forward2, linear k=2, non-linear per-row | ~51.8 ms | 38.6 |
| + lm_head k=2 | ~46.1 ms | 43.4 |
| + non-linear fused to one dispatch per op | ~40.9 ms | 48.9 |

So reaching 40 tok/s needs **both** the batching (steps 1-2) and the dispatch
fusion (step 3). A plan that stopped at forward2 would land at 38.6 and stall.

## The blocker the read exposed

The gated-delta-net branch cannot be wrapped row-by-row syntactically:
`in_proj_qkv` is encoded by hand to write straight into `window[3 * conv_dim]`, the
shared four-row convolution history, and `copy_dispatch` then rotates
window -> `conv_hist`. Running row 1's projection before row 0's convolution would
clobber it. The window needs a fifth row for two tokens, which is a layout change.

That branch is 48 of the 64 layers, so the first cut leaves it per-row (its
projections read their weights twice) and accelerates everything else: all 64 MLPs,
all 16 full-attention blocks, the final norm and lm_head.

## Next round

Step 1: write `forward2` with the row-wrapped body, `encode_k(..., TILE)` +
`K_Q4_GEMV_K2` for the projections, `buf_offset(i, buf, row * ROW_BYTES)` for
everything else, `forward` left untouched so parity can be re-checked at any time.
Gate: 6/6 plus a row-1 logits comparison against a plain `forward(pos+1)`.
