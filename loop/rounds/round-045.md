# Round 045 - the instrument was lying by 3.5%, and the model is always measuring a hot GPU

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=pass G4=accept.sh 12 entered below
- outcome: round 044's 7% win is really ~1.7%; the paired sweep is now calibrated and agrees
  with end-to-end; the reason the model can never be measured cold is identified; and the real
  target for 50 tok/s is narrowed to a power-insensitive ~25 ms.

## The instrument had a 3.5% bias

The paired sweep compared candidates against a baseline arm measured in the same round, on the
assumption that order and drift cancel.  They do not.  A duplicate of the baseline - identical
kernel, identical grid - measured in the same rounds reads:

```
run 1:  k3 + u4  0.9535   |  k3 (baseline dup)  0.9649   -> calibrated u4 = 0.9882
run 2:  k3 + u4  0.9521   |  k3 (baseline dup)  0.9559   -> calibrated u4 = 0.9960
```

So the raw column flatters every candidate by ~3.5%, and round 044's "median ratio 0.9304, 76/80
wins" was that bias plus a ~1% effect.  The end-to-end interleaved A/B settles it: **4 pairs,
orders alternating, median paired ratio 0.9831 - u4 is +1.7% tok/s.**  The bench is now honest
by construction: the last variant is the calibration arm and every ratio is printed both raw and
`calibrated`, and the calibrated column is the one that matches the model.

## Why the model can never be measured cold

Round 044's `QW_BENCH_BURN` showed a cold sweep at 33-38 ms and a burned sweep at 113 ms.  The
model always reads ~119 ms, even as the first thing after a 150 s cooldown:

```
after 150 s idle, model steady-state: 53.7 ms/token   (plateau measured 53.4)
forward2 113.82 ms | logits 2.46 ms | tail 2.76 ms
```

The reason is the **prefill**: it runs the prompt one pass at a time (12 tokens = 12 passes,
0.5-2.8 s of sustained GPU load) before a single decode step is timed, and that is enough to
trigger the throttle.  So the bench is cold unless explicitly burned, while the model is always
hot - which is the whole of the 3.5x gap that round 042 could not explain, and not anything about
the kernels.  Consequence: **`gen`'s steady-state window is the reproducible metric and it
measures the throttled rate; the cold rate is only ever visible in the first second of a run.**

## Where that leaves the target

```
plateau pass 143.6 ms = sweep 119 (83%) + draft 9.7 (7%) + non-GEMV 8.6 (6%) + tail 3 + logits 2.5
mains   pass  60   ms = sweep ~33 (55%) + everything else ~27 (45%)
```

The sweep is bandwidth-throttled at the plateau and u4-style kernel work is worth ~1%.  The
"rest" is **~24.6 ms at the plateau against ~27 ms on mains - it barely responds to the power
state at all**, so it is dispatch, sync and latency bound rather than bandwidth bound, and at
mains it is 45% of the pass.  Halving it is +12.5 ms, which is 42.4 -> ~51 tok/s.  That, not the
sweep, is where 50 tok/s has to come from, and it is measurable at the plateau (8.7% of a pass,
against a 0.5% noise floor).

Its composition, measured: draft 9.7 (two MTP steps, each with a host round trip for the argmax),
non-GEMV dispatches 8.6, tail 3.0, logits pulls 2.5.
