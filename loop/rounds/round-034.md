# Round 034 — the tiled forward is exonerated, and the server was missing its snapshots

- date: 2026-09-15
- gates: G1=pass G2=oracle 6/6 (see the open issue) G4=pass

## A fixed bug: the server never enabled the snapshots

`spec_snap` was wired to the `QW_SPEC` environment variable at model load:

```rust
spec_snap: std::env::var("QW_SPEC").is_ok(),
```

The CLI sets that variable when it takes the speculative path, so it was on there. **The
server never sets it**, so `commit_row` - the call that rewinds the recurrent state after a
rejected draft - was a silent no-op in the endpoint, and every rejected draft stayed in the
GDN state.  It is now an explicit capability, `model.enable_spec_snap()`, called by both
front ends that use `spec_step`.  Oracle parity 6/6 and the 64-token byte-for-byte check
still pass.

## The tiled forward is bit-exact

`QW_TILE_CHECK=<depth>` walks `<depth>` tokens greedily with the sequential path, then
replays the same prefix through the tiled path from a fresh prefill and compares logits row
by row:

```
tilecheck: prefix 237 rows 3 tokens [35978, 13, 561]
tilecheck: row 0 max|dlogit|=0.000000e0 argmax 13 vs 13
tilecheck: row 1 max|dlogit|=0.000000e0 argmax 561 vs 561
tilecheck: row 2 max|dlogit|=0.000000e0 argmax 3712 vs 3712
```

**Zero difference, not "small", at prefixes 37, 101 and 237.**  The k=3 GEMV, the per-row
GDN and attention loops all reproduce the sequential evaluation exactly.  The old
`QW_K2_CHECK` only ever tested a two-token prefix, which is why it never caught anything.

## Comparing against mlx-lm over 200 tokens

A fresh stepwise mlx-lm reference (the same method `tools/oracle.py --stepwise` uses) shows
that **both** Rust paths leave the reference at index 103 in the same place and re-converge
two tokens later - a near-tie flip, and the same for both, so it says nothing about
speculation.  The 16-token oracle gate is too short to see it.

## The open issue, now much sharper

| comparison | diffs (200 tokens) | first |
|---|---|---|
| plain vs spec | 14, re-converging | 162 |
| plain vs spec **with acceptance forced to zero** (`QW_NO_ACCEPT=1`) | **190, permanent** | **4** |

Forcing `k = 0` disables every accepted-draft branch and leaves only the shared forward plus
`commit_row(0)`.  That the sequence then diverges at token 4 and never recovers means the
rejection path does **not** rewind something that a later forward reads.  The ring window,
the k/v caches and the GDN recurrence were each checked by hand and each appears
position-indexed and self-correcting, so the next step is to stop reasoning and dump the
actual buffers: snapshot the GDN `state`, the `window` rings and a layer's k/v cache after a
rejected pass in both paths and diff them.  Every emitted token is still the argmax of its
own forward pass, and `QW_NO_SPEC=1` restores the plain path exactly.

Gates: fmt clean, clippy `-D warnings` clean, tests pass, oracle parity 6/6.
