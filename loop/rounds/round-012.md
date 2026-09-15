# Round 012 — GPU argmax attempt: 5/6, reverted

- date: 2026-09-15
- gates: G1=pass G2=pass (**6/6 restored after revert**) G3=no change G4=pass

## What was tried

The decode loop calls `argmax()` every token, which reads all 248320 logits to the
host (496 KB) and scans them there — a per-token fixed cost, and per round 10's
conclusion the fixed costs are what the k=2 path will have to amortise.

Added `argmax_chunk`: one threadgroup per 4096 logits, threadgroup reduction, one
(value, index) partial per chunk, so the host reads back 61 partials = 488 bytes
instead of 496 KB.

## Result

```
parity: 5/6   (one case diverged)
decode: 36.37 ms/token  (no gain: the readback was never the cost)
```

Two conclusions, both worth keeping:

1. **The host readback was not measurable** — the timing did not move at all, so
   even a 1000x smaller readback buys nothing. Another per-token fixed cost
   eliminated as a suspect, which strengthens round 10's conclusion that the
   ~9 ms is GPU-side dispatch and real non-linear work.
2. **The kernel is not equivalent to the CPU argmax** on one case. The likely
   cause is tie-breaking: the CPU scan keeps the *first* maximum in index order,
   and the two-stage reduction keeps the lowest thread id within a chunk and the
   lowest chunk index overall, so ties *should* agree — which means the real
   difference is probably fp16-vs-fp32 comparison or the `min(base+chunk, n)`
   edge chunk. It was not worth debugging further for a zero-gain change.

Reverted: 6/6 parity is the project's most valuable asset and a 1/6 regression is
not an acceptable price for an unmeasurable optimisation.

## Next round

The k=2 engine path. Nothing else on the list can move the number now: round 6
made the second token nearly free in the linear part (28.68 ms for two tokens vs
26.87 ms for one), and every per-token fixed cost hypothesis has now been
measured and eliminated.
