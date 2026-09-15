# Round 029 — the conv window becomes a per-layer ring: -192 dispatches, output unchanged

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6 + byte-identical) G3=pass G4=pass

## Why this and not the draft

Instrumenting the spec loop split a pass into draft vs verify:

```
spec: draft 9.44 ms/pass, verify 121.89 ms/pass (7% of a pass)
```

Even throttled, the ratio is the point: **the draft is a few percent of a pass**, so
optimising it cannot carry the target.  The verify pass is everything.

## The change

Each linear-attention layer maintained its convolution history by copying it around the
shared four-row `window` on **every row**: `conv_hist -> window` before the convolution
and `window -> conv_hist` after it.  That is two dispatches per row per GDN layer whose
only job is to move 3 x conv_dim halves a few kilobytes.

`Gdn` now owns a persistent `[conv_k][conv_dim]` ring.  The qkv projection writes
straight into slot `pos % conv_k`, and `conv1d_silu_ring` reads the four slots in ring
order (`r = (slot + 1 + j) & 3`, so `j == 3` is the current row).  Both copies
disappear.

The ring is indexed by absolute position, which has two useful consequences:

* a rejected speculative row leaves a stale slot only until that position is rewritten,
  which happens before anything reads it again - so the window needs **no snapshot**;
  `csnap` and its per-row copies are gone too, and `commit_row` now restores only the
  recurrent state;
* `reset()` zeroes the window, which reproduces the old "zero history at sequence start"
  behaviour exactly.

## Verified

| | before | after |
|---|---|---|
| dispatches, k=1 | 1282 | **1186** (-96) |
| dispatches, k=2 | 1794 | **1602** (-192) |
| oracle parity | 6/6 | **6/6** |
| spec vs plain greedy | identical | **identical** |

The count fell by exactly the predicted amount, and the convolution is bit-for-bit the
same computation (the same four rows multiplied by the same weights, reordered into ring
order).  Parity 6/6 against the mlx-lm oracle and byte-identical speculative output
confirm it.

At the measured ~16 us per non-linear dispatch this is worth ~3 ms of a 49 ms pass
(~+6%).  The box is throttling hard right now (74.75 ms/token in this session against
41.56 unthrottled), so the wall-clock gain could not be measured cleanly this round; the
dispatch count is thermal-independent evidence and the timing check is queued for the
next clean window.

## Where the remaining milliseconds are

The k=2 chain is 1602 dispatches: 497 GEMVs (~28.6 ms of weight streaming, already 97% of
peak) plus ~1100 small ops.  The next targets, in order of dispatch count:

| op | count (k=1) | how |
|---|---|---|
| rmsnorm + rmsnorm_s | 257 | fold the residual add in, fold the q/k norms together |
| ewise_add | 128 | fold into the preceding norm |
| silu_mul | 64 | let the `up` gemv do the activation (gate is already barrier-ordered) |
| conv1d_silu / gdn_step / rmsnorm_gated | 144 | batch the tile rows in one dispatch |

Everything left is a variant of "fewer, fatter dispatches", because the chain is
latency-bound and the work per dispatch is tiny.

Gates: fmt clean, clippy `-D warnings` clean, tests pass, parity 6/6, spec output
byte-identical to greedy.
