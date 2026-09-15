# Round 010 — the binding cache is correct, and still buys nothing

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6) G3=**no change** G4=pass

## Hypothesis retried, and confirmed as a diagnosis

Round 9's binding cache broke correctness (0/6, 3.7 ms/token) and round 9's report
named the cause: the cache was not invalidated on a pipeline switch, and the
argument table belongs to the pipeline state. Round 10 re-landed it with exactly
that one change.

**The diagnosis was right**: parity came straight back to 6/6, and the 8-test GPU
suite passed. The encoder now lives for the whole batch with an in-encoder
`memoryBarrierWithScope`, and redundant `set_buffer` calls are skipped.

**It is still neutral in wall clock**: 35.95 / 36.00 ms per token against a 35.85
baseline. The reason is visible in the dispatch order: consecutive dispatches in
this engine almost always switch kernel (gemv -> rmsnorm -> conv1d -> gdn -> gemv
...), so the pipeline changes and the cache is dropped at nearly every dispatch.
It only saves anything where the same kernel repeats back to back (the three
q/k/v projections, ~2 binds each): a rounding error against ~8000 calls.

## Three driver-side hypotheses are now dead

| hypothesis | measured |
|---|---|
| encoder teardown per dependency (round 7) | neutral (round 8) |
| encoder churn, in-encoder barrier (round 8) | neutral |
| ~8000 objc binding calls per token (rounds 9-10) | neutral |

So the ~8.9 ms that is not the 27 ms weight sweep is **GPU-side**: ~1282 kernel
dispatches, each with its own scheduling latency and its own real work (48 conv1d,
48 delta-net steps, ~200 norms, attention, the logits readback). At a few
microseconds of GPU-side dispatch cost each, that alone is several milliseconds.

The lever is therefore **fewer dispatches**, not cheaper ones — or, better, the
k=2 path, which amortises the entire per-token fixed cost over two tokens
(round 6 made the linear part of the second token free).

The change was reverted again: neutral in wall clock, and the `objc::msg_send!`
macro trips `clippy -D warnings` (`unexpected cfg cargo-clippy`) from inside the
macro expansion. It is recorded here, working, for whenever a batched path makes
binding count matter.

## Next round

Start the k=2 engine path: it is the only lever that amortises all of this at
once, and round 6 already proved the weight pass is nearly free for a second
token.
