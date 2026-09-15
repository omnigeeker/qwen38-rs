# Round 007 — the remaining 9 ms/token is encoder teardown (and the binding is missing)

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6 after revert)** G3=no change G4=pass

## Where the non-linear time goes

One decode step is 35.9 ms, of which ~27.0 ms is the 497-linea weight sweep
(measured directly by `bench`). The other ~8.9 ms is attention, delta net, norms,
the logit readback — and **1282 encoder teardowns per token**.

`CommandBatch::barrier()` closes the current compute encoder
(`enc.end_encoding()`) and lazily opens a new one for the next dispatch, because
the code assumes Metal only guarantees cross-dispatch memory ordering across
encoder boundaries. At ~7 us per teardown x 1282 dependencies that is ~9 ms per
token — the whole unexplained gap, and it is pure driver overhead, not work.

## Attempt and outcome

Rewrote `barrier()` to insert `memoryBarrierWithScope(MTLBarrierScopeBuffers)`
inside the *same* encoder (Metal executes the dispatches of one encoder in
submission order; a barrier makes prior writes visible). This is the standard
fix, and `finish()` already ends the encoder, so the structure was sound.

It does not build: the `metal` crate binding in use exposes neither
`MTLBarrierScope` nor `memory_barrier_with_scope`. Reverted — the engine is back
to 27.1 tok/s with 6/6 parity intact.

## Next round: add the missing binding, then measure

Three routes, in order of preference:

1. `memory_barrier_with_resources(&[&MTLResourceRef])` if this binding has it —
   `CommandBatch` would track the buffers bound since the last barrier and pass
   them along. Contained change in `kernel.rs` plus a mechanical rewrite of the
   `b.barrier()` call sites.
2. A tiny raw `msg_send` shim for `memoryBarrierWithScope:` on the encoder.
3. Migrate this crate to `objc2-metal`, which exposes both.

Expected: ~9 ms off a 35.9 ms step at k=1, i.e. ~28 ms -> **36 tok/s single
stream**, at which point k=2 (round 6: two tokens for the price of one) is what
carries the objective past 40. The k=2 engine path is therefore the other
half of the next rounds: attention over two positions, the delta-net recurrence
for two steps, and norms for a two-row tile.
