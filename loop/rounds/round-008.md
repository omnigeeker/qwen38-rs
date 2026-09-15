# Round 008 — in-encoder memory barriers land, and prove the 9 ms is real work

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6) G3=**no change** G4=pass

## What was done

Round 7 identified ~1282 encoder teardowns per token as the prime suspect for the
~8.9 ms/token that is not the weight sweep, but the `metal` 0.31 binding appeared
not to expose the barrier selector. Three routes were tried:

1. `objc2-metal` (`MTLBarrierScope`, `memoryBarrierWithScope`) — the crate is
   present, but these encoder types come from the *old* `objc` crate, not objc2
   (`ComputeCommandEncoderRef implements metal::objc::Message, but not
   objc2::Message`).
2. `objc2::msg_send!` — same trait mismatch.
3. `objc::msg_send!` (the binding's own objc re-export) with
   `use objc::{sel, sel_impl}` — **this works**. `barrier()` now inserts
   `memoryBarrierWithScope:MTLBarrierScopeBuffers` inside the running encoder
   instead of `end_encoding()` + a fresh encoder, which is what llama.cpp's Metal
   backend does.

It builds and it is *correct*: 6/6 prefill-oracle parity held, and so did the
8-test GPU suite.

## But it buys nothing

```
before: 35.83 / 35.92 ms per token
after:  35.80 / 35.69 ms per token   (four runs, all within run-to-run noise)
```

So encoder teardown is **not** the missing 9 ms — the round-7 comment in
`kernel.rs` ("cheap: no command buffer round-trip") was right, and my suspicion
was wrong. The ~8.9 ms is genuine GPU work: 16 attention layers, 48 delta-net
layers, the norms and the 248k-logit readback.

The change was reverted anyway: it is performance-neutral, it tripped
`clippy -D warnings` (the objc macro imports), and a neutral change that alters
memory-ordering semantics is pure risk to correctness with no upside. The
mechanism is now known and can be re-applied if a future kernel ever needs it.

## What this means

Two rounds of driver-overhead hypotheses are now dead. The remaining levers are
the two real ones, and both are on the critical path anyway:

1. **The k=2 engine path.** Round 6 made a second token nearly free in the linear
   part; amortising the per-token fixed costs (norms, the logits readback, the
   1282 encoder *submissions*) over two tokens is where the rest of the 40 tok/s
   budget is.
2. **Cut real non-linear work**, starting by attributing the 8.9 ms between
   attention and delta net with a skip-measurement, so the next optimisation is
   aimed at the part that actually costs.

## Next round

Attribute the 8.9 ms: run a token with the full-attention layers skipped and with
the delta-net layers skipped (timing only, output discarded), then start the k=2
engine path with the result in hand.
