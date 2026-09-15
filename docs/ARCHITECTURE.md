# Architecture

## 1. Why the design looks like this

The machine is an **M5 Max (18-core CPU, 40-core GPU, 128 GB unified) on macOS 26.6**
with only the Command Line Tools installed:

* `xcrun metal` is **absent** → offline kernel compilation is impossible.
  `MTLDevice::newLibraryWithSource` still works (verified), so every kernel is
  JIT-compiled on first use and cached by content hash. This is exactly what
  `mlx.fast.metal_kernel` does.
* Unified memory means a 15 GB weight file can be **aliased** into a Metal buffer
  instead of copied: `mmap` the shard, `newBufferWithBytesNoCopy`, keep the
  mapping alive for the process lifetime. Loading the whole model measures
  ~15 ms and 0 extra RSS.

Decode is bandwidth-bound, not FLOP-bound: every generated token reads all
15.13 GB of language-model weights once. Measured peak read bandwidth is
494–519 GB/s, so the hard ceiling is ~30–33 tok/s for one stream. Anything above
that requires emitting more than one token per weight pass — i.e. **MTP
speculative decoding** (`mtp_num_hidden_layers = 1` in this checkpoint).

## 2. Model plan (verified against the real checkpoint)

```
vocab 248320 · hidden 5120 · ffn 17408 · 64 layers · ctx 262144
full_attention_interval = 4  ->  48 linear (gated delta net) + 16 full attention
full attention: 24 q-heads / 4 kv-heads, head_dim 256, partial RoPE (rotary_dim 64, theta 1e7)
linear attention: 16 k-heads x 128, 48 v-heads x 128, conv kernel 4
MTP: 1 layer (mtp.fc, mtp.pre_fc_norm_{embedding,hidden}, mtp.layers.0, mtp.norm)
```

Numerical contract copied from `mlx_lm/models/qwen3_5.py` (the oracle):

* RMSNorm weights are stored **zero-centred** and must be `+1.0` at load.
* gated delta net: `q = inv²·rmsnorm(q)`, `k = inv·rmsnorm(k)`, `inv = head_k_dim^-0.5`;
  `beta = sigmoid(b)`; `g = exp(-exp(A_log)·softplus(a + dt_bias))`; state in fp32.
* MLP is SwiGLU (`silu(gate) * up`), pre-norm residual on both sub-layers.

## 3. Quantisation and kernels

Weights are MLX affine 4-bit: `uint32` words holding 8 little-endian nibbles,
one fp16 `scale` and `bias` per 64 values per row, `w = q*scale + bias`.

Kernel plan (in `qw-metal/src/msl.rs`):

| kernel | purpose | state |
|---|---|---|
| `q4_gemv` | decode-shape quantised GEMV (one threadgroup per row, 32 lanes) | done, verified vs CPU |
| `rmsnorm` | fused RMSNorm (+ `+1.0` shifted weight) | done |
| `q4_gemm_t` | prefill-shape quantised GEMM (dequant into `simdgroup_matrix`) | next |
| `attn_decode` | GQA attention with KV cache, partial RoPE | next |
| `gdn_step` | gated delta net recurrence (fp32 state, conv1d window) | next |
| `mtp_step` | MTP block + verification | next |

Correctness is enforced in layers: kernel unit tests compare against CPU
references, then hidden states are compared layer-by-layer against the oracle
(`tools/oracle.py --dump-hidden`), and finally greedy token ids must match.

## 4. Concurrency and scheduling

* One `MTLCommandQueue`; one command buffer per forward step with as few encoders
  as possible. Encoder boundaries are the barrier mechanism between dependent
  kernels.
* No CPU↔GPU round-trip inside a step beyond reading the sampled token id.
* Request scheduler (M5/M6): continuous batching over a shared KV/state pool;
  the gated delta net's recurrent state is O(1) per sequence in context length,
  and only 16 layers keep a KV cache, so long-context batching is unusually cheap
  for a 27B model.

## 5. Protocol layer

`qw-server` exposes one router with two front-ends over a single internal request
type:

* `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`
* `POST /v1/messages` (Anthropic), including `system` as string or text blocks
* `GET /health`

Streaming (SSE) uses each protocol's own chunk shape; errors are shaped per
protocol (`{"error": {...}}` vs `{"type":"error","error":{...}}`).
