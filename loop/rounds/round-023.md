# Round 023 — the MTP head: quantised, wired, drafting, and measured

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6, decoder untouched, `forward2` still bit-exact)** G3=no change G4=pass

## What exists now

`tools/mtp_quantize.py` turns the official bf16 MTP shard into the same MLX affine
4-bit layout the decoder already reads (group 64, 4 bits, bf16 scales/biases) and
writes `models/Qwen3.8-27B-mtp-4bit/mtp.safetensors`: 8 matrices (`fc`, `q/k/v/o`,
`gate/up/down`) plus 7 norms, 31 tensors.

`Mtp` in `runner.rs` is that head: `fc` fusing the token embedding with the
decoder's hidden state, one full-attention decoder layer with its own KV cache, and
its own residual stream.  `Qwen38::mtp_step(next_token, pos, want_logits)` advances
it one token and returns the draft logits for `pos + 1`; `reset()` clears its caches.
It is loaded from `$QW_MTP_DIR` or from a sibling `Qwen3.8-27B-mtp-4bit` directory.

`QW_MTP_CHECK=1` measures the thing that matters: the head rides along the prompt
(so its cache covers the prompt, which is what a real decode loop must do), then each
draft is compared against the token the decoder actually picks.

```
mtp check: prompt 12 tokens, head cache warm in 530 ms
mtp check: acceptance 11/32 = 34.4% | first draft 7.0 ms | head+head sweep 47.6 ms/token
```

## Two conversion bugs, both found by measurement

**fp32 scales read as bf16.**  The first version quantised `w.astype(float32)`, and
MLX gives the scales/biases the dtype of the input - so the file had fp32 scales
while the engine reads bf16 (which is what the 4-bit decoder export stores).  Every
draft came out NaN.  Quantising the bf16 tensor directly fixed it.

**The `pre_fc_norm_*` norms need the `+1` shift.**  Their raw values are *negative*,
which no RMSNorm weight is; the bf16 release stores every norm weight as `w - 1` and
mlx-lm applies the shift whenever a checkpoint carries `mtp.*` tensors.  The MTP
module's own norms are covered by that rule too - my first version shifted only the
layer norms.  With the two `pre_fc_norm` weights shifted, acceptance went 12.5% ->
34.4%.  (`mtp.norm` is the exception: its raw values already sit at ~1.)

The concatenation order is confirmed by experiment rather than by assumption:
`[embed | hidden]` is 34.4%, `[hidden | embed]` is 0% (`QW_MTP_SWAP=1`).

## The decoder did not move

Parity 6/6, `forward2` bit-exact on both rows, and the two-token pass still measures
`k=1 40.30 ms | k=2 49.17 ms = 40.7 tok/s`.  MTP is opt-in: without the weights the
engine behaves exactly as before.

## Where this leaves the objective

34.4% is not yet enough to pay for itself.  A k=2 speculative pass would yield
`0.34 + 1 = 1.34` tokens for `49.2 + 7 = 56 ms`, i.e. 24 tok/s - *slower* than the
25 tok/s of plain decoding.  Two numbers have to move before the draft/verify loop is
worth wiring:

* **acceptance** (34% -> the 70-80% band a trained MTP head should reach).  The
  remaining candidates are the hidden state the head consumes (pre- vs
  post-final-norm), whether the head's own attention should read the decoder's cache
  rather than its own, and a per-norm shift audit.
* **the head's cost**, ~7 ms/token against a ~1.6 ms bandwidth floor for its 205 MB of
  weights plus the 636 MB head sweep.  The gap is dispatch and launch overhead, the
  same lever as the decoder's per-row ops.

A process note that cost real time this round: a `println!` whose argument list had
drifted out of step with its format string reported an "inf" norm that was not there,
and the diagnosis went down a blind alley.  Diagnostics now return one structured
value (`mtp_dump`) so a print cannot mis-align, and the numbers in this document all
come from such single-value dumps.
