# Round 001 — M0: engine skeleton, Metal JIT runtime, zero-copy loader, loop harness

- date: 2026-09-15
- milestone: M0
- gates: **G1 = pass**, **G2 = pass (kernel level)**, G3 = baseline measured, **G4 = pushed**

## Hypothesis / goal

A from-scratch Rust engine on this exact machine must first prove three
environmental facts before any model work is worth doing:

1. Metal kernels can be compiled **without** the Xcode Metal toolchain
   (this box has only Command Line Tools).
2. The 15 GB 4-bit checkpoint can be brought into GPU address space **without a
   copy** (mmap + `newBufferWithBytesNoCopy`).
3. Our own 4-bit affine GEMV can match a CPU reference on **real** weights —
   which also validates every byte offset produced by the loader.

## What was built

| crate | content |
|---|---|
| `qw-metal` | device/queue, `mtl` buffers, **runtime MSL JIT** with content-hash caching, `CommandBatch` (one command buffer per step, encoder boundaries as barriers), `q4_gemv` / `rmsnorm` / `ewise_add` kernels |
| `qw-weights` | safetensors header parse (`__metadata__` aware), mmap, `bytesNoCopy` alias with page-aligned length, tensor handle with offset, `QuantSpec` (affine 4-bit, g64) + CPU dequant |
| `qw-model` | config parsing (64 layers, 48 linear + 16 full, rope 1e7/0.25, MTP=1), full weight-name layout, `QLinear` bound to the store |
| `qw-engine` | tokenizer (`tokenizer.json`, ChatML template), sampler (greedy/top-k/top-p, seeded, reproducible) |
| `qw-server` | OpenAI + Anthropic wire types, router, per-protocol error shapes |
| `qw-cli` | `qwen38 info / check / bench / verify / serve / gen` |
| `tools/` | `oracle.py` (mlx-lm reference + `--dump-hidden`), `download_model.sh` (ModelScope, incl. `--mtp`), `prompts.json` |
| `loop/` | `LOOP.md` constitution, `state.json`, `run_round.sh`, `verify.sh`, `env.sh` |

## Evidence

**Environment facts established**

- `xcrun metal` absent; `mx.fast.metal_kernel` JIT works → runtime compilation is
  the only viable path, and our own JIT works too (`gpu_is_available_and_jit_compiles`).
- Toolchain installed *inside the repo* (`.toolchain/`) because writes to `$HOME`
  are denied in this sandbox: `rustc 1.98.1`, `cargo 1.98.1`, clippy + rustfmt.
- Weights fetched with the ModelScope CLI (`mlx-community/Qwen3.8-27B-4bit`,
  15 GB, 3 shards, `tokenizer.json` present). ModelScope cache redirected into the
  workspace with `MODELSCOPE_CACHE`.

**Loader validated against the real checkpoint** (`qwen38 info`):

```
shards             : 3 (3 zero-copy mmap aliased)
tensors            : 2180
expected-tensor check: 0 missing
MTP weights in shard: false      <- MLX 4-bit export strips mtp.*; bf16 shard needed
```

**Correctness (G1/G2, kernel level)** — `cargo test --workspace`:

```
gpu_smoke::gpu_is_available_and_jit_compiles            ok
gpu_smoke::unified_memory_budget_check                  ok (115.4 GB recommended working set)
gpu_smoke::ewise_add_matches_cpu                        ok
gpu_smoke::q4_gemv_matches_dequantized_cpu_reference    ok
qw_model/qw_engine/qw_weights/qw_server unit tests      18 ok, 0 failed
```

Two real bugs were found and fixed by these tests (both worth recording):

1. `dispatch_threads` takes a grid in **threads**, not threadgroups — the first
   GEMV run silently computed only the first 2 of 64 rows.
2. An `ewise_add` assertion at magnitude ≈512 used an absolute 1e-3 tolerance;
   at that magnitude the fp16 ULP is 0.5, so the *test* was wrong, not the kernel.
   Fixed by comparing against the correctly-rounded fp16 result.

**Reference oracle generated** (`loop/artifacts/oracle.json`, mlx-lm on the same
4-bit weights, greedy, 6 prompts):

```
short_greeting     16 tok  10.9 tok/s (includes warm-up)
capital_qa         16 tok  30.2 tok/s
math_steps         24 tok  27.7 tok/s
multilingual       16 tok  29.8 tok/s
```

This is the bar: **the fastest existing Mac engine reaches ~28–30 tok/s** on this
model, consistent with the measured 494–519 GB/s read bandwidth and 15.13 GB of
weights. The project's 40 tok/s target therefore **requires MTP speculation**;
there is no kernel cleverness that can exceed the bandwidth ceiling.

## Decision

**ACCEPTED** for M0. Next: M1 — single-token forward (RMSNorm/MLP/GQA/gated-delta-net)
aligned layer-by-layer against `oracle.py --dump-hidden`.

## Next actions

1. ~~`qwen38 check`~~ — done, 8/8 tensors pass.
2. ~~`qwen38 bench`~~ — done, 33.5 tok/s linear-path upper bound at 482 GB/s.
3. Implement remaining kernels: `silu_mul`, `q4_gemm_t` (prefill), `attn_decode`
   (partial RoPE + GQA + KV cache), `gdn_step` (conv1d + fp32 recurrent state).
4. Implement the 64-layer forward and the MTP block.
5. Fetch the bf16 `mtp.*` shard via ModelScope and quantise it locally.

## Late-round additions (same round, after the first push)

### Real-weight kernel validation — `qwen38 check`

The zero-copy loader's offsets were validated by running our `q4_gemv` on **real
checkpoint tensors** and comparing against a CPU dequantised reference:

```
tensor                                                   out     in     max_abs    rms_ref   norm_err
...layers.0.mlp.gate_proj                              17408   5120      0.4913     320.91     0.153% ok
...layers.0.mlp.down_proj                               5120  17408      0.9590     566.37     0.169% ok
...layers.0.linear_attn.in_proj_qkv                    10240   5120      0.4888     334.15     0.146% ok
...layers.0.linear_attn.out_proj                        5120   6144      0.4999     362.59     0.138% ok
...layers.3.self_attn.q_proj                           12288   5120      0.4966     331.77     0.150% ok
...layers.3.self_attn.o_proj                            5120   6144      0.4023     357.88     0.112% ok
...layers.63.mlp.up_proj                               17408   5120      0.4762     329.29     0.145% ok
...lm_head                                            248320   5120      0.4954     326.76     0.152% ok
```

The ~0.15% normalised error is exactly the fp16 output quantisation floor
(ULP/2 relative to the tensor RMS), i.e. the arithmetic agrees with the reference.

### Finding: safetensors alignment is header-relative

Every tensor is 16-byte aligned **relative to `data_start = 8 + header_len`**, but
`data_start % 16` is 6 / 10 / 5 for the three shards. Since the file is mapped at a
page boundary, the *absolute* tensor addresses are misaligned for `half4`/`uint4`
loads, so a pure whole-file alias cannot feed vectorised kernels.

Resolution: `Shard` now tries the alias path only when every absolute address is
16-byte aligned, and otherwise **materialises** the shard into one GPU buffer with
every tensor at a 256-byte boundary. Cost measured: **3.1 s for the whole 15 GB**
(one memcpy), and the kernels then see fully aligned operands. The alias path
stays in the code for checkpoints that permit it.

### Throughput baseline — `qwen38 bench`

One full pass over all 497 quantised linears with the real weights (the
bandwidth-bound part of a decode step):

```
quantised linears: 497 (14.41 GB of weights, 0 unaligned)
one full weight sweep: 29.9 ms -> 33.5 tok/s linear path, 482 GB/s effective
```

**482 GB/s effective = 93 % of the 519 GB/s measured device peak** — the kernel is
already essentially at the hardware limit for the linear path. Two consequences:

1. The remaining headroom in decode is in the non-linear parts (attention, GDN,
   norms, sampling) and, decisively, in **emitting more than one token per weight
   pass (MTP)**, because 33.5 tok/s is the ceiling for a single token per sweep.
2. Any further kernel tuning on the GEMV path can win at most ~7 %.

