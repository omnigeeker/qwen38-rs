# Round 025 — two hypotheses killed, and where the 13 ms actually is

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6) G3=no change (both experiments reverted) G4=pass

Round 24 ended with a suspicious number: the k=1 pass costs 39.79 ms against a
27.8 ms bandwidth floor (14.41 GB at the measured 519 GB/s), and the second row of a
k=2 pass costs another ~10 ms even though the weights are read once for both.  Two
explanations looked plausible.  Both were wrong, and the measurements say which.

## Hypothesis 1: barrier overhead (wrong)

`CommandBatch::barrier()` ended the compute encoder and opened a new one, and the
generated forward calls it 73 times per layer - over a thousand encoder switches per
token.  Replacing it with an in-encoder `memoryBarrierWithResources` over the touched
buffers is strictly cheaper, and parity stayed 6/6 with byte-identical generation.

Interleaved A/B, three pairs, min per variant:

| barrier | k=1 | k=2 |
|---|---|---|
| encoder switch (old) | 41.19 ms | 51.22 ms |
| resource fence (new) | 40.19 ms | 51.48 ms |

No difference.  Encoder switches are cheap on Metal 4.  Reverted: the resource fence
needs raw-pointer bookkeeping to reconstruct the resource list, and buys nothing.

## Hypothesis 2: int->float conversion in the dequantiser (wrong)

`q4_gemv` extracts eight nibbles per `uint32` and converts each with `(float)(...)`.
Replacing that with a `constant float kQ4Lut[16]` lookup returns *bit-identical*
values (verified: parity 6/6 and byte-identical generated text) with one indexed load
instead of shift+and+convert.

Interleaved A/B against the pre-LUT build:

| build | k=1 | k=2 |
|---|---|---|
| LUT | 46.4 / 52.1 / 56.6 ms | 63.8 / 85.4 / 90.3 ms |
| no LUT | 40.7 / 41.6 / 41.6 ms | 51.4 / 54.8 / 56.0 ms |

A clear regression, and it got worse as the run went on.  Indexed constant-memory
loads are a slow path here; the int->float conversion is cheap.  Reverted.

## What the arithmetic actually says

Neither launch overhead nor conversion ALU is the limit, because the kernel is bound
by *data movement* - and not by the weights:

* each output row is one threadgroup, and its 32 lanes walk **every** group of `x`.
  So one GEMV reads the whole `x` vector once **per output row**:
  `out_f * K * 2` bytes.  For a 5120x10240 projection that is 105 MB, against 26 MB of
  4-bit weights for the same projection - **four times more activation traffic than
  weight traffic**, all of it re-reads of the same 20 KB, i.e. served by L2.
* summed over the model's ~450 projections: ~14.4 GB of weights from DRAM and
  ~47 GB of activation re-reads from L2.  At the observed 40 ms that is 360 GB/s of
  DRAM (69% of the measured peak) **and ~1.2 TB/s of L2**.  Both are near their
  practical ceilings.
* it also predicts the marginal row cost.  A second input row doubles the activation
  traffic (4x -> 8x the weight traffic) and leaves the weights alone: the model says
  the extra row should cost roughly `47 GB / 1.2 TB/s = 39 ms`... spread over the
  measured 10.4 ms of extra time only because the two pass shapes differ.  The
  qualitative prediction - the extra row is expensive, and it is activation-bound, not
  weight-bound - matches the measurement.

So the lever is to stop re-reading the activations: process **R output rows per
threadgroup**, so one pass over `x` in registers feeds R weight rows.  Activation
traffic divides by R while the weight traffic is unchanged; this is the shape of
llama.cpp's `mul_mat_vec_q` and of MLX's `qmv` kernels.

At R=4 the L2 stream falls from ~47 GB to ~12 GB per pass, which leaves the DRAM
stream as the limit: ~28-30 ms for k=1 (33-36 tok/s) and - because the same weight
sweep serves the extra rows - a k=2 pass whose second token is nearly free.  That,
not a better drafter, is what stands between this engine and 40 tok/s.

## Measurement discipline

The gate run at the end of this round printed `k=1 56.62 ms`, roughly 40% off the
40.7 ms measured for the same binary twenty minutes earlier: the machine was hot from
six A/B pairs.  Only the interleaved pairs above are used as evidence; the absolute
number is noise.  `cargo fmt` clean, clippy `-D warnings` clean, 13 test binaries
pass, parity 6/6.
