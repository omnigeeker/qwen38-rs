# Round 009 — binding cache: a precise failure that names its own fix

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6, after revert) G3=no change G4=pass

## Hypothesis

One token issues ~1282 dispatches and each rebinds its buffers
(`set_buffer` per buffer, an Objective-C call each), i.e. ~8000 objective-C calls
per token *before* the command buffer is committed. Unlike GPU time, that cost
does not overlap anything: the CPU encodes the whole token, then commits. If it
is ~1 us per call it is ~8 ms — the same order as the ~8.9 ms that is not the
weight sweep, and it is independent of the barrier question round 8 settled.

Fix attempted: cache the last pipeline and the last binding per buffer index in
`CommandBatch`, skip redundant `set_buffer` calls, and keep the encoder alive
across dependencies with an in-encoder `memoryBarrierWithScope` (round 8's
mechanism) so the bindings stay valid — buffer bindings are encoder state.

## Result: it built, it ran, and it was wrong

```
parity: 0/6          (was 6/6)
decode: 3.7 ms/token (was 35.9) — the GPU was skipping most of the work
```

The timing is the useful part: 3.7 ms/token is far below the 27 ms weight sweep,
so a large number of dispatches were reading stale bindings rather than doing
arithmetic. Reverted; 6/6 restored, 27.89 tok/s restored.

## The failure names its own cause

The cache was **not** invalidated when the pipeline changed. The first draft did
clear it on a pipeline switch and I removed that to "simplify" — which is very
likely exactly the bug: on Apple GPUs the argument table is part of the
*render/compute pipeline state*, so changing `set_compute_pipeline_state`
invalidates previously bound buffers, and skipping the rebind leaves the shader
reading whatever the new pipeline's default bindings are.

So the retry is precise: keep the cache, but clear it whenever `set_pipeline`
fires (and only then). That is a one-line difference from what was measured here,
and it is worth one more round because the payoff — removing thousands of
Objective-C calls per token from the critical path — is measured in milliseconds
per token, not microseconds.

## Next round

Re-land the binding cache with invalidation on every pipeline switch, verify 6/6,
and measure. If it is still wrong, drop the binding cache entirely and keep the
in-encoder barrier only if it shows a gain on its own.
