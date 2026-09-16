# Round 047 - batching from 1634 dispatches to 1378, and why all of it had to go back

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=no speedup claimed G4=pass
- outcome: 256 dispatches were removed, a latent **correctness** trap was found and avoided, and the
  change was then measured to be **1.04% slower** and reverted.  The round-045 plan was built on
  the assumption that the ~24.6 ms non-sweep rest is dispatch-bound; that assumption is now dead.

## What was tried

Round 045 concluded that the power-insensitive ~24.6 ms rest is dispatch, sync and latency bound,
and proposed removing ~416 of the 1137 non-GEMV dispatches by tiling fusions.  Four families were
batched over TILE, on the observation that these kernels already take the row from the threadgroup
index, so a flattened `(row, head)` index times `in_stride` should land on exactly the address the
per-row `buf_offset` computed:

| family | per-row | batched |
|---|---|---|
| `rmsnorm_gated` (phase 3) | 3 / layer | 1 / layer |
| `rmsnorm_s` q_norm | 3 / layer | 1 / layer |
| `rmsnorm_s` k_norm | 3 / layer | 1 / layer |
| `rope_partial` q and k | 6 / layer | 2 / layer |

Result: **1634 -> 1378 dispatches**, parity 6/6 - and `tools/accept.sh` failing on
`spec==plain: 291 of 300 tokens differ, first at 4`.

## The trap: `key_dim` is not `nkv * hd`

`crates/qw-model/src/runner.rs:982` defines `key_dim = hk * dk` - **48 * 128 = 6144** - while the
attention's k row is `nkv * hd` = **4 * 256 = 1024**.  The per-row k_norm/k_rope pass
`row * (key_dim * 2)` as a byte offset, but the kernel addresses `row * in_stride` from whatever
base it is given, and `in_stride` is `hd`.  Two different row strides therefore live in the same
call, and **no flattened index `(r * nkv + h)` can satisfy both**: `(r*nkv+h)*hd` is right for
x and wrong for y.  The q path has no such mismatch (`nh * hd * 2` bytes is exactly
`in_stride * nh`), which is why only half the batching compiled into the wrong addresses.

That the gate caught this is the point of the gate: parity 6/6 passed - six short greedy prompts
never exercise a mis-strided row - and only the 300-token `spec == plain` comparison failed.

Fixing it properly needs the kernel to separate the row from the head (pass the head count and a
`y_stride`, then `row = idx / n_heads`, `head = idx % n_heads`), which is a real kernel change, not
a grid change.

## The part that was verifiably correct, and how it performed

`rmsnorm_gated` has no stride mismatch: `value_dim = hv * dv` and the kernel's row stride is
`D = dv`, so `(r * hv + h) * dv` is exactly the address the per-row offset produced.  Kept alone,
it passed everything - parity 6/6, `accept.sh` 12/12 ACCEPTED, dispatch count 1634 -> **1538** -
and measured, over three clean pairs with orders alternating:

```
pair1  old 18.43  new 18.19  ratio 1.0132
pair2  old 18.43  new 18.24  ratio 1.0104
pair3  old 18.37  new 18.25  ratio 1.0066
median paired ratio 1.0104  ->  1.04% SLOWER
```

**Removing 96 dispatches made the pass slower.**  Reverted; the tree is back to round 046.

## What this kills

The dispatch count is not the lever.  Round 045's arithmetic - 1137 dispatches at 7.6 us each,
halve the count for +12.5 ms - does not survive contact: 96 dispatches is 6% of a pass's dispatch
count and removing them costs 1%.  Most likely the per-row dispatches overlap with the neighbouring
kernels in a way the batched form cannot, or the smaller grid schedules better; either way, a pass
is not paying 7.6 us per dispatch in a way that merging recovers.

So the non-sweep rest needs a different explanation before it can be attacked.  The one thing this
round does establish is negative and firm: **the ~25 ms rest is not reducible by dispatch
batching**, and the next attempt on it has to start from measurement rather than from the count.
