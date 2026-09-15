# Round 027 — the pass is linear-optimal already; speculation needs a prefix state commit

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6, byte-identical output) G3=infrastructure G4=pass

## Read the state first, and it had the answer

Round 006's table: the k=2 *linear sweep alone* runs in **28.64 ms = 503 GB/s, 97% of
peak**, with k=3 at 32.34 ms and k=4 at 38.41 ms.  So the 497 GEMVs were never the
problem, and rounds 25-26 spent their effort on the part that was already optimal.  The
difference between that 28.64 ms and a full k=2 pass (~49 ms) is the 20 ms spent
*outside* the linears.

## Where the 20 ms goes: a 1282-dispatch serial chain

`QW_DISPATCH_HIST=1` now prints the per-entry-point dispatch count for any pass.  A k=1
pass is **1282 dispatches, only 497 of them GEMVs**:

| kernel | count | share |
|---|---|---|
| q4_gemv | 497 | 39% |
| rmsnorm + rmsnorm_s | 257 | 20% |
| ewise_add | 128 | 10% |
| copy_off | 96 | 7% |
| silu_mul | 64 | 5% |
| conv1d_silu / gdn_step / rmsnorm_gated | 48 each | 12% |
| rest (rope, kv_append, attn, gate) | 96 | 7% |

The non-linear 40% of the pass is a chain of small launches, each serialized behind the
previous one.  Fusing the residual add into the preceding norm (-64 to -128 dispatches)
is worth ~5-10%.

## Barriers are not the cost (third independent confirmation)

`QW_BARRIER=none` (every dispatch in one encoder, no fences) is **correct** - parity 6/6,
byte-identical generation - but not faster, and in the one cold pair clearly slower.
Reverted.  Round 25 found the same thing comparing encoder switches against resource
fences: the fences are cheap and the chain length is what costs.

## The finding that decides the whole design

A verify pass runs the recurrent state **through every drafted token**.  If the second
draft is rejected, the state is wrong for the accepted prefix, and the only correct
recovery without extra machinery is to restore the pre-pass snapshot and re-run the
accepted row - a full k=1 pass.  At alpha = 0.56 that is a **net loss**:

```
accept both (p=0.31): 2 tokens in 49 ms
accept one  (p=0.44): 1 token  in 49 + 40 ms
accept none (p=0.19): 1 token  in 49 + 40 ms
E[tokens]/E[time] = 1.06 / 76.6 ms = 13.8 tok/s   (plain decoding is 24.6)
```

So speculative decoding here **requires the verify pass to commit the state of its
longest accepted prefix**, not to roll back.

## Implemented: per-row state snapshots and `commit_row`

* `Gdn` gains `snap`/`csnap`, holding `TILE` snapshots of `state` and `conv_hist`.
* Inside forward2's per-row GDN loop, each row's recurrent update is followed by a copy
  into its snapshot slot (only when `QW_SPEC=1`).
* `commit_row(row)` copies slot `row` back into the live state, so after accepting the
  first row of a two-row pass the recurrence matches exactly that token.
* `copy_off` moves 2-byte units, so the fp32 `state` is counted and offset in halves;
  the copy is still byte-exact.

Behaviour-neutral by construction and verified: parity 6/6 and **byte-identical
generation** with `QW_SPEC` on and off.

## Next

Wire the draft -> verify -> accept loop (one MTP draft, k=2 verify, up to two tokens per
pass), then fuse the residual add into the preceding norm.  The objective still needs
both: the target is 40 tok/s end to end and the goal stays active.
