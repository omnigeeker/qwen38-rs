# Round 002 — M1: full forward pass, exact mlx-lm parity

- date: 2026-09-15
- milestone: M1 (single-token forward) — **complete**
- gates: G1=pass (fmt + clippy -D warnings + 37 tests) G2=**pass (token-for-token)** G3=28 tok/s G4=pass
- tok/s: 27.9 decode (35.9 ms/token), 6/6 oracle cases exact

## Hypothesis of the round

> A correct decode path can be assembled from the kernels already validated in
> round 1 (GEMV, RMSNorm, RoPE, SiLU) plus new attention / gated-delta-net
> kernels, and validated **against mlx-lm token for token** rather than against a
> re-derivation of the same formula.

Result: confirmed, after three real bugs were found and fixed (below).

## What was built

New kernels (`crates/qw-metal/src/msl_ops.rs`, all unit-tested against CPU
references in `crates/qw-metal/tests/ops_smoke.rs`):

| kernel | role |
|---|---|
| `rmsnorm_s` | RMSNorm with input row stride, output scale, optional weight (attention q/k norms **and** delta-net q/k norms) |
| `rmsnorm_gated` | delta-net output norm: `rms_norm(y, w) * silu(z)` |
| `conv1d_silu` | depth-wise causal conv (k=4, 10240 channels) + SiLU |
| `gdn_step` | the gated-delta-net recurrence with a persistent fp32 state `[48][128][128]` |
| `attn_scores_softmax`, `attn_out` | GQA attention over the head-major KV cache |
| `kv_append`, `gate_mul` | cache append; `out * sigmoid(q_gate)` |
| `copy_off`, `round_bf16` | window shift; optional round-through-bf16 |

`crates/qw-model/src/runner.rs` wires all 64 layers into **one command buffer per
token** (1282 dispatches, encoder boundary per dependency) and exposes
`set_token` / `forward(pos)` / `argmax` / `reset`.

## Bugs found (all silent — none crashed)

### 1. `scales`/`biases` are BF16, not FP16

The MLX 4-bit export stores the per-group side tensors in the *source* dtype
(bf16). `q4_gemv` read them as `half`, and `QLinear::cpu_reference` did the same,
so **the GPU/CPU comparison in round 1 agreed with itself while both were
wrong** — the kernel check gave "0.15% error" on garbage. Fixed by reading
`ushort` and widening with `as_type<float>(bits << 16)`.
*Lesson: a self-consistent test cannot validate an interpretation. Only an
external oracle can.*

### 2. `rmsnorm_s` buffer ABI drift

When the kernel gained `w` / `scale` / `has_weight`, the Rust dispatches kept the
old indices, leaving `y` unbound (and reading an uninitialised `w`). Symptom:
all-zero logits in one run, NaN in the next — **nondeterministic** because it read
whatever the driver left in the slot.

### 3. An extra 1/128 query scale in the attention path

The gated-delta-net scales its queries by `inv_scale²` (`qwen3_5.py`), and that
line was copied into the *attention* q-norm, where the reference has no such
scale (attention only applies `scale = head_dim^-0.5` inside SDPA). Effect: the
softmax became nearly uniform, attention degenerated to an average of V.

Measured per-layer divergence from mlx-lm (relative L2 of the residual stream,
17-token prompt) before → after:

| layer | type | before | after |
|---|---|---|---|
| 2 | linear | 0.0035 | 0.0035 |
| 3 | **full** | **0.0372** | **0.0027** |
| 15 | full | 0.1056 | 0.0072 |
| 31 | full | 0.2674 | 0.0139 |
| 47 | full | 0.3864 | 0.0144 |
| 63 | full | 0.5388 | 0.0179 |

The jump at the first full-attention layer (0.35% → 3.7%) is what pointed at the
attention path; the remaining 1.8% is fp16-vs-bf16 arithmetic.

## Correctness evidence

```
$ qwen38 verify --oracle loop/artifacts/oracle.json      # mlx-lm greedy, batched prefill
short_greeting        tokens=ok greedy 16/16 PASS
capital_qa            tokens=ok greedy 16/16 PASS
code_snippet          tokens=ok greedy 24/24 PASS
math_steps            tokens=ok greedy 24/24 PASS
multilingual          tokens=ok greedy 16/16 PASS
long_context_marker   tokens=ok greedy 12/12 PASS
parity: 6/6 cases
```

`verify` compares the tokenizer output and then runs greedy decoding **through
our own engine**, token for token, against mlx-lm's greedy ids.

Against the *stepwise* oracle (`tools/oracle.py --stepwise`, prompt fed one token
at a time) the score is 5/6: `math_steps` diverges at token 16. That is not our
bug — mlx-lm's own prefill and stepwise paths disagree with each other on that
case (their kernels differ), so no implementation can match both.

Tooling added for this: `tools/hidden_dump.py` (per-layer, stepwise, mlx), and
`qwen38 gen --dump-hidden` / `--dump-vectors` on our side for layer-by-layer
numerical comparison.

## Performance

```
one full weight sweep (497 linears, 14.41 GB): 29.9 ms  = 33.5 tok/s, 482 GB/s (93% of peak)
end-to-end decode (13-token prompt, 64 tokens): 35.9 ms/token = 27.9 tok/s
mlx-lm reference on the same machine:           33 ms/token  = 30 tok/s
```

We are at 83% of the single-token weight-pass ceiling; the remaining 6 ms/token
is the non-linear work (attention, delta net, norms, 1282 encoder boundaries, and
the host-side 248k logit readback for argmax). Beating 40 tok/s therefore requires
**more than one token per weight pass** — i.e. MTP, which is next.

## Next round

1. MTP: quantise the local bf16 `mtp.*` shard, add `mtp.fc` + the MTP decoder
   layer, and run draft/verify speculative decoding.
2. Prefill path (`q4_gemm_t`) so prompts do not pay T decode steps.
3. Then M3: cut the 6 ms/token non-linear overhead (batch the projections, stop
   re-reading the fp32 delta-net state, GPU-side argmax).
