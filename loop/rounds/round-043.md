# Round 043 - the machine throttles 3x under sustained load; that was the whole "in situ" mystery

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=pass G4=accept.sh 12/12 ACCEPTED
- outcome: round 042's open question is answered, the answer kills the remaining optimisation
  lever for this power state, and a reproducible acceptance metric replaces the average-rate
  number that was never comparable across runs.

## The answer: it is the clock, not the kernel

Round 042 ended unable to explain why the identical 497-dispatch GEMV sweep costs 113.8 ms inside
a pass but 32.6 ms in the isolated bench, with CPU encoding, encoder splitting, dependencies and
x locality all refuted.  The remaining candidate was GPU clock state, and `QW_BENCH_BURN=<secs>`
settles it - it hammers the same sweep until a deadline, then times:

```
cold            38.42 ms
after 3 s burn 110.60 ms    (burn sweeps averaged 100.0 ms each)
after 10 s burn 160.57 ms   (burn sweeps averaged 111.1 ms each)
new process, cold again  32.49 ms
```

The same dispatches, same kernels, same weights: 3-5x slower once the GPU has been under load,
and back to full speed in a fresh process.  **The in-situ/isolated gap was a power-management
artifact.**  A corollary worth stating plainly: the isolated bench can never be compared against
a pass, in either direction, and rounds 039-042 all leaned on that comparison.

## The throttle plateaus, which is what makes measurement possible again

```
burn  5 s -> 156.2 ms/sweep, then measured 206.76 ms   (still degrading)
burn 15 s -> 144.2 ms/sweep, then measured 110.63 ms
burn 30 s -> 112.8 ms/sweep, then measured 113.12 ms
burn 45 s -> 113.6 ms/sweep, then measured 113.18 ms
after 120 s idle -> 36.41 ms
```

After ~20-30 s the state is stable to **0.05%** across separate processes, and a 2 minute idle
fully restores it.  Every A/B in rounds 039-042 was taken during this transient, which is exactly
what produced ratios that flipped sign between runs.

## A reproducible acceptance number

The average rate over a whole run mixes two clock states, so `spec:` now also reports a
steady-state window measured after the plateau is reached (the last third of the tokens):

```
18.48 tok/s (132 tokens in 7.14 s, 2.69 tokens/pass, 54.1 ms/token)
18.43 tok/s (132 tokens in 7.16 s, 2.69 tokens/pass, 54.3 ms/token)
18.38 tok/s (132 tokens in 7.18 s, 2.69 tokens/pass, 54.4 ms/token)
```

**0.5% spread over three identical runs.**  That is the number to quote and to compare.  18.4
tok/s here against 42.4 tok/s on mains is the same binary, so the mains/battery factor is ~2.3x.

## Where a pass's time goes, now measured rather than inferred

`verify` split (throttled plateau):

```
set_tokens              0.03 ms
forward2              126-128 ms
3x(logits pull+argmax)   2.73 ms
```

So the GPU-side argmax idea on the standing list is worth 2% and is not worth doing - another
hypothesis killed cheaply.  Inside forward2, the round-042 ablation gives sweep 113.8 / non-GEMV
8.6, i.e. **the sweep is 77% of a pass and everything else together is 23%.**  The 1137 non-GEMV
dispatches, which rounds 040-041 were preparing to fuse at length, are worth 6%.

## The sweep at the plateau, and why nothing is left to win there

```
k=1  84.10 ms  171 GB/s
k=2  97.88 ms  147 GB/s
k=3 113.15 ms  127 GB/s
```

Fitting `a + b*k` gives a fixed 69.6 ms for the weight stream plus a flat **14.5 ms per row**.
The fixed term is 14.41 GB at only 207 GB/s - against ~440 GB/s measured cold - so the plateau
has throttled memory bandwidth as well as clocks, and 14.41 GB of 4-bit weights is irreducible.

The per-row term is the only reducible part (29 ms of 146 ms) and row blocking is precisely the
kernel for it.  Re-tested at the plateau, 60 tightly paired rounds, orders alternating:

```
k3 + 2 rows/tg  median ratio 1.0690  wins  3/60
k3 + 3 rows/tg  median ratio 1.0387  wins  7/60
k3 + 4 rows/tg  median ratio 1.0266  wins 15/60
```

Slower at every blocking factor, now with a trustworthy measurement behind it.  **Row blocking is
closed for good** - the per-row cost is the FMA issue rate at the throttled clock, not x loads, so
removing x loads cannot pay.

## Consequence for the objective

On this power state the sweep saturates two resources that are both throttled by the platform:
memory bandwidth (207 GB/s) and ALU issue (14.5 ms per row).  Neither is reachable from the code.
The 50 tok/s target has to be worked and verified on mains with Low Power Mode off; on battery the
honest statement is 18.4 tok/s steady-state, reproducibly measured.  What this round does buy is
that any future mains measurement is now comparable, and that four more candidate optimisations
were killed rather than pursued.
