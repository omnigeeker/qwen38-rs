# Round 042 - the sweep is 93% of the forward, and in situ it runs at half the isolated bandwidth

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G4=pass
- outcome: four hypotheses measured and KILLED; three permanent diagnostics added.  No speedup
  claimed.  The target moved: the non-GEMV dispatch work I spent two rounds trying to fuse is
  worth 7% of a pass, not the 37% I assumed.

## First: a reading error in the engine's own report, which invalidated round 041's arithmetic

`gen.rs:387` divides `t_draft`/`t_pass` by `drafts` (= passes * (TILE-1)), not by `passes`.  So
the printed "ms/pass" is really **ms per draft**, and the true split is draft 9.9 + verify 133.1
+ tail ~3 = ~146 ms/pass, which matches the wall clock (56.55 ms/token * 2.54 = 143.6).  Round
041's "half the pass is unaccounted for" was an artifact of that divisor.  `verify` is 91% of a
pass.  The tail (`commit_row` + `promote_hidden` + the post-accept redraft), which nothing timed,
turned out to be 1.7-4.8 ms.

## The ablation that reorders the whole plan

`QW_ONLY_KERNEL`/`QW_SKIP_KERNEL` filter dispatches by entry-point name inside `encode`, so a
pass can be attributed to kernel families instead of guessed from counts.  k=2 forward:

```
full                        122.40 ms
only q4_gemv                113.78 ms     <- 497 dispatches
skip gdn_step               117.17        (-5.2)
skip rmsnorm family         120.43        (-2.0)
skip conv1d / skip attn     122.9 / 123.4 ( 0  )
```

**The 1137 non-GEMV dispatches are worth 8.6 ms - 7% of the forward.**  So the fusion work
(ewise_add into rmsnorm, batching rmsnorm_gated over TILE) that rounds 040-041 were aimed at
could not have paid more than a few percent.  The sweep is the whole game.

## But the same 497 dispatches cost 3.5x more in a pass than in the sweep

```
                        in situ        isolated bench
k=1 (497 disp)     55.08 - 64.48 ms      27.39 - 36.24 ms
k=3 (497 disp)    113.78 ms              32.62 ms
```

Same entry points, same grid (`out_f*32`, threadgroup 32), same kernels, same weights, same
x/y shape.  Four explanations tested and all refuted:

| hypothesis | test | result |
|---|---|---|
| CPU command encoding is serial before commit | timed batch creation to `finish` | **1.14-1.86 ms** for 1634 dispatches - not the cost |
| `barrier()` splitting into 1090 encoders | real encoder counter | 1090 encoders vs **1 encoder: no change** (61.64 vs 55.08 on the filtered run) |
| dependencies / x producers stall the GEMVs | filtered run has no producers at all | still slow |
| x locality (shared always-hot buffer in the bench) | `QW_X_PER_LINEAR=1`, one x per linear | **33.48 ms vs 36.24 - no slowdown** |

That is the open question, and it is now the highest-value one in the project: if the in-situ
sweep ran at the bench's rate the forward would fall from ~122 ms to ~40 ms.

`barrier()` turning out to be `end_encoding()` rather than a memory barrier is worth recording on
its own - it means the batch never uses a within-encoder barrier, and that this costs nothing
measurable.

## Diagnostics added, all env-gated

- `QW_ENCODE_TIME=1` - per batch: CPU encode ms, commit+wait ms, dispatch count, encoder count.
- `QW_ONLY_KERNEL=<substr>` / `QW_SKIP_KERNEL=<substr>` - dispatch filters for attribution.
- `QW_X_PER_LINEAR=1` - bench uses a distinct x buffer per linear.
- `QW_NO_ENC_SPLIT=1` - **unsafe** timing bound: keeps every dispatch in one encoder.

## Environment

Still battery + Low Power Mode (`powermode 1`).  The same binary read 55.18-55.52 ms/token at the
start of the round (stable to 0.5%, which is what made these ablations readable at all) and
drifted to 83-87 ms/token by the end.  Absolute numbers remain incomparable across power states;
every conclusion above is a within-state comparison.
