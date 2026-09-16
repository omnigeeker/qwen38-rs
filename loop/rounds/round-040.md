# Round 040 - pass census, two copy folds, and a measurement-environment trap

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte G4=pass
- measured: 2.0% faster (4/4 interleaved pairs), 1922 -> 1634 dispatches per pass

## First: the pass census, which is where the 50 has to come from

`QW_DISPATCH_HIST` on a real decode pass: **1922 dispatches**, of which only 497 are
`q4_gemv_k3` (26%).  The remaining 1425 are elementwise and reduction work:
`copy_off` 288, `gdn_step` 144, `rmsnorm_gated` 144, `rmsnorm` 129, `rmsnorm_nw_tile` 96,
`rmsnorm_s` 96, `rope_partial` 96, `ewise_add` 128, `silu_mul` 64, `conv1d_silu_ring_tile` 48,
and 48 each of `kv_append`, `attn_scores_softmax`, `attn_out`, `gate_mul`.

The engine also reports its own split: **draft 5.9 ms/pass, verify 89.6 ms/pass** - the MTP
chain is about 6% of a pass, so drafting is not where the time is.

## A measurement trap I fell into last round and have now cleared

`bench --tokens 3` reports 78.70 ms per sweep at `rows=0` but 38.19 ms at `rows=1`, which looks
like a 2x win from row blocking.  It is not.  `bench` with `rows=0` calls `QLinear::kernel_k()`,
which is `q4_gemv_k` - the **runtime-k** kernel whose token loop is not unrolled.  `rows=1` picks
`q4_gemv_k3`, the unrolled specialisation, with an identical grid.  The engine's real path
already uses the unrolled one (`Kernels::q4_gemv_tile` = `K_Q4_GEMV_K3`), so the 78.70 ms figure
was measuring a kernel the model never runs, and round 039's note that the tiled kernel reaches
"~62% of the single-row bandwidth" was based on it.

The corrected numbers, same harness, unrolled kernels both sides:

```
k=1  31.33 ms   460 GB/s
k=3  38.19 ms   377 GB/s      marginal row = 3.43 ms
```

3.43 ms per extra row is 57.6 GFLOP, i.e. ~16.8 TFLOP/s of fp32 - essentially the chip's ALU
peak.  **The three-row GEMV is therefore already close to optimal**, and the 6.86 ms it spends
on rows 2-3 cannot be removed without changing numerics (packed fp16/tensor-core math), which
would put the oracle parity at risk.  So the 50 tok/s target has to come out of the 1425
non-GEMV dispatches, not out of the GEMV.

## The change: fold two copies into the kernels that already touch the data

Both were `copy_off` launches whose data the surrounding kernel had already loaded or produced.

1. **State snapshot into `gdn_step`.**  The speculative rewind needs the delta-net state as of
   the end of each row; the kernel has those values in registers as it finishes, so it now
   writes them itself (buffers 13/14, the write guarded by `snap_on`).  This removes one
   whole-state copy per row *and the read that copy did* - 3.1 MB per layer per row, ~450 MB
   per pass.
2. **Ring update into `conv1d_silu_ring_tile`.**  The row a thread just convolved is exactly the
   raw row the next pass needs in the ring, so it writes its own ring slot.  No thread can read
   the slot it writes: every window read in that dispatch is for `pos < pos0`, at least three
   slots behind.  The separate copy loop in `forward2` is gone.

Result: 1922 -> **1634** dispatches (exactly -288 as predicted), parity 6/6, spec==plain
byte-identical over 300 tokens, and the plain path byte-identical to the previous build.

```
iter 1: baseline 59.35  folded 57.90  ratio 0.9756
iter 2: baseline 58.89  folded 57.98  ratio 0.9845
iter 3: baseline 60.30  folded 58.41  ratio 0.9687
iter 4: baseline 58.96  folded 58.47  ratio 0.9917
median paired ratio 0.9801  =>  2.0% faster, 4/4 pairs
```

## The environment trap that makes absolute numbers meaningless right now

Mid-round the absolute throughput collapsed from 23.6 to 58.5 ms/token and stayed there.  It is
not the engine: `pmset -g batt` reports **"Now drawing from 'Battery Power' ... discharging"**
and the power mode is `1` (Low Power).  Apple Silicon caps GPU and memory clocks hard in that
state.  Three consecutive runs then read 58.53 / 58.54 / 58.25 ms - stable, and 2.5x slower than
the same binary on mains power.

Consequence: **absolute tok/s measured in this session is only comparable within one power
state.**  All the interleaved A/B work here is still valid (both arms share the state), but any
"we hit 50" claim has to be made with the machine plugged in and Low Power Mode off.  This is
also why the 4 pairs above are quoted as a ratio rather than as tok/s.

Separately: a stray `qwen38 serve --addr 127.0.0.1:8088` process from an earlier round (it uses
an old CLI flag, so it predates the current interface) was found holding 18.7 GB.  Killed.  Worth
remembering that a leftover server can silently sit in the way.
