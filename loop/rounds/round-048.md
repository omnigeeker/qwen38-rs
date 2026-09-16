# Round 048 - the non-sweep work is 33% of a pass, not 7%, and it is spread over every dispatch

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=no speedup claimed (measurement round) G4=pass
- outcome: no code change.  The cost model inherited from rounds 043/045 is wrong by 4x, the
  dispatch-count lever is closed by measurement from both sides, and the mains budget is now
  written down with a number for every term.

## Within-run timing, which is the only kind that survives the throttle

`QW_ENCODE_TIME=1` prints one line per `CommandBatch`, so a single run gives the cost of every
phase against the *same* thermal state.  A 12-token prefill plus one spec pass, in one process:

```
forward p=0   commit+wait 256.59 ms  1186 dispatches   <- first GPU work in the process
forward p=1                  34.38 ms  1186 dispatches  <- coolest sustainable full forward
forward p=2                  35.30 ms
forward p=3                  37.43 ms
forward p=4                  45.01 ms
forward p=5                  53.33 ms
forward p=6                  77.44 ms
forward p=7                 130.44 ms   <- plateau reached after ~7 tokens
forward p=8                 128.82 ms
forward p=11                 89.36 ms
mtp_step(want_logits=false) 0.76 - 9.78 ms  (11 of them)
verify (TILE=3)            156.99 ms  1634 dispatches
```

Two things fall out.

**The throttle ramp is visible token by token.**  The same kernel drops from 34.38 ms to 130 ms
over seven tokens and then settles at 89-130 ms.  Any A/B that interleaves runs is comparing
thermal states, which is why everything measured across processes in rounds 043-046 wobbled.

**The coolest full forward is 34.38 ms** = 14.41 GB at 419 GB/s, against a DRAM peak of 546 GB/s.
That is the closest thing to a mains proxy the battery can produce, and it is worth 29 tok/s for a
hypothetical single-row decode that never throttles.

**`mtp_step` is cheap.**  With `want_logits=false` it skips the lm_head and the f16 readback and
costs 0.8-9.8 ms (the MTP layer's ~226 MB of weights).  So the MTP warm-up is *not* what makes the
spec prefill cost 1.1-2.6 s for 12 tokens - that is entirely the thermal ramp, and the prefill time
is therefore a thermometer, not a constant.

## The cost model is wrong by 4x

Within one run, plateau: a k=1 full forward is ~93 ms and the TILE=3 verify is ~157 ms.  The bench's
pure-GEMV sweep is 85.74 ms at k=1 and 113.15 ms at k=3.  Per row the full model spends ~32 ms
where the GEMV alone spends ~14.5 ms:

| component | k=1 | k=3 | share of the 157 ms pass |
|---|---|---|---|
| GEMV sweep (weights + x) | 85.7 | 113.2 | 72% |
| everything else | ~7 | **~44-52** | **28-33%** |

Round 043's ablation put "all non-GEMV dispatches together" at 8.6 ms, and round 045 rebuilt the
40 -> 50 plan on that number.  A full pass says 44-52 ms.  The ablation measured the *bench sweep's
own batch*, which contains only GEMV dispatches, so what it actually demonstrated was that the
sweep has no non-GEMV work in it - not that the model's non-GEMV work is small.

## And it is not attributable to any one family

Eleven ablations, each run on the full spec pass with `QW_NO_ACCEPT=1` so the row count is fixed,
reading the verify batch's own `commit+wait`:

```
baseline                    198.99 ms (1634 disp)
skip rmsnorm                199.43 ms (1169 disp)
skip gdn_step               192.16 ms (1138 disp)
skip conv1d_silu_ring_tile  198.27 ms (1586 disp)
skip attn_scores_softmax    195.01 ms (1170 disp)
skip attn_out               194.95 ms (1170 disp)
skip ewise_add              195.42 ms (1058 disp)
skip silu_mul               195.41 ms (1122 disp)
skip rope_partial           195.88 ms (1154 disp)
skip kv_append              192.29 ms (1170 disp)
skip gate_mul               196.34 ms (1170 disp)
```

Every skip lands inside +-3.5% and the baseline is the *slowest* of the twelve.  Removing 465 of the
1137 non-GEMV dispatches (the whole rmsnorm family) buys nothing measurable.  ~52 ms over 1137
dispatches is ~46 us each: the cost is the dispatch itself, spread evenly, not any kernel's work.
That is consistent with round 046's result from the other side - merging 96 dispatches made the pass
1.04% *slower* - and it closes the dispatch-count lever both ways: the count is what costs, and
merging it back does not recover the cost.

## The mains budget, with every term

At mains the same run structure gives a pass of ~60 ms.  With the weight stream at DRAM peak
(14.41 GB / 546 GB/s = 26 ms, and round 001's 42.4 tok/s is 60 ms/pass, so 26 + 34 = 60):

| term | mains | plateau | hard floor? |
|---|---|---|---|
| weight stream, 14.41 GB | 26 ms | 69.6 ms | **yes, DRAM peak** |
| per-row ALU (3 x 14.5) | ~19 ms | 43.5 ms | ~28% of fp32 peak |
| non-GEMV, 1137 dispatches | ~15 ms | ~52 ms | dispatch latency |

TILE=4 and TILE=5 were re-costed against this: the weight stream is paid once per pass but the ALU
rows and the non-GEMV rows grow linearly, so 2.83 tokens at ~73 ms and 3.03 tokens at ~89 ms are
both worse than 2.54 at 60 ms.  TILE=3 is the optimum, as measured in round 002.

**So the ceiling for this design is ~42-45 tok/s and it is already there.**  50 tok/s needs one of
the three floors moved: a matrix-unit (simdgroup) path to replace the ALU-bound per-row dots, fewer
bytes per weight, or a per-row kernel count that is actually lower - which needs the padded-stride
fix (row = idx/n_heads, head = idx%n_heads, explicit y_stride) that round 047 identified, and which
this round shows must then be justified against a 1% regression, not against a 3x win.
