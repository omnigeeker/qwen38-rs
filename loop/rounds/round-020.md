# Round 020 — the `scores` stride; row 0 is now bit-exact

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6, `forward` untouched)** G3=no change G4=pass

## The bug, found by bisecting on the right signal

Round 19 localised the failure to the first full-attention layer. The stride table
used to re-issue per-row operations was **missing `scores`** (`nh * max_t * 4`), so
both rows wrote the same 24x2048 fp32 block and row 0's `attn_out` consumed *row 1's*
softmax. That is why the fault showed up in row 0 as well as row 1 - and only from
the first full-attention layer onward, since the delta-net layers never touch that
buffer. Adding the stride fixes row 0 completely:

```
k2check: layer 3 ref absmax=49.90625 got absmax=49.90625     (round 19: 48.62500)
k2check: row0 argmax ref=2614 got=2614 max|d|=0.0000         (round 19: max|d|=2.6724)
k2check: row1 argmax ref=1898 got=513  max|d|=8.7119
k2check: self-pair row0 argmax ref=2614 got=760 max|d|=17.3066
```

**Row 0 is bit-exact** (`max|d| = 0.0000`). That is a strong result: it independently
confirms the tile layout, all four `q4_gemv_k2` projections, the hoisted delta-net
projections, the MLP, the residuals, the layer-0..N sequencing and `set_tokens`.

## Row 1, and a contradiction worth chasing

Row 1 is still wrong, and so is row 0 of the self-pair run - but here is the
puzzling part, which is the most useful thing this round produced:

* `[t0, t1]`, the **first** `forward2` call after a fresh `reset()`, gives a
  bit-exact row 0;
* `[t0, t0]`, the **second** `forward2` call in the same process, does not.

Row 0's arithmetic cannot depend on row 1's token in any correct implementation, so
the self-pair result cannot be explained by cross-row interference. The two run
orders differ only in what the *previous* call left behind. That points at one of:

1. `reset()` does not clear something `forward2` writes - the delta-net state or the
   convolution history. The k=1 reference runs before the first `forward2` would have
   to be cleaned the same way, so this is testable by reordering the checks.
2. `forward2` writes out of bounds for row 1 and corrupts a neighbouring buffer, so
   the *next* forward pass is the one that reads garbage. The row-1 offsets that
   would do that are exactly the ones with a non-zero legacy offset added on top:
   `conv_out + 2 * key_dim * 2`, the two `rmsnorm_nw` reads and the delta-net's
   `conv_out` value slice.

Test 1 costs nothing: run the self-pair case **first** and the `[t0, t1]` case second.
If the failures swap, it is state; if the same case fails, it is layout.
