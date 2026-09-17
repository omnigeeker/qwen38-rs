# qwen38-rs

A from-scratch **Rust** inference engine for **Qwen3.8-27B FP4** on Apple Silicon
(Metal), built and tuned for a single-box **M5 Max / 128 GB / macOS 26** setup.

* Engine host code: 100% Rust. GPU kernels: MSL, JIT-compiled **at runtime**
  (`MTLDevice::newLibraryWithSource`) — this machine has only the Command Line
  Tools, so there is no `xcrun metal` and no offline `metallib` step.
* Weights: 4-bit affine quantisation (`group_size = 64`), loaded **zero-copy**
  straight from the ModelScope-downloaded safetensors via `mmap` +
  `newBufferWithBytesNoCopy`.
* Target: **≥ 40 tok/s single-stream decode** with **MTP speculative decoding**
  (the model ships one MTP head), while staying **token-for-token identical** to
  the `mlx-lm` reference on the correctness suite.
* Deliverable: a local endpoint speaking **both OpenAI and Anthropic** protocols.

Measured hardware limits on this box (see `loop/state.json`):

| quantity | value |
|---|---|
| peak streaming read bandwidth | 494–519 GB/s |
| 4-bit LM weight bytes | 15.13 GB |
| decode ceiling without speculation | ≈ 30–33 tok/s |
| `mlx-lm` (Python, same weights) | 28–30 tok/s steady state |

Because the ceiling without speculation is below the goal, **MTP is not optional**
— it is the mechanism that makes ≥ 40 tok/s reachable.

## Layout

```
crates/qw-metal     Metal runtime: device, buffers, runtime MSL JIT, dispatch
crates/qw-weights   safetensors parsing, zero-copy upload, quant metadata
crates/qw-model     Qwen3.8 config, weight naming, (next) the forward pass
crates/qw-engine    tokenizer, sampler, KV/state cache, MTP, scheduler
crates/qw-server    OpenAI + Anthropic wire types and HTTP routes
crates/qw-cli       `qwen38` binary: info / serve / bench / gen / verify
tools/              mlx-lm oracle, ModelScope download, prompt suite
loop/               the iteration protocol, state machine, per-round records
```

## Documentation

* **[docs/TUTORIAL_zh.md](docs/TUTORIAL_zh.md)** — 保姆级教程: build it, download the
  weights, start the endpoint, verify all five routes, and the full API capability
  table with the known limits (`max_tokens`, context, concurrency, `<think>` blocks).
* **[docs/AGENT_FRAMEWORKS_zh.md](docs/AGENT_FRAMEWORKS_zh.md)** — 把端点接进各个 Agent
  Framework: OpenAI/Anthropic SDKs, LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI,
  Aider, Continue, Cline/Roo, OpenHands, Dify, n8n, Open WebUI, Cherry Studio, Zed,
  Claude Code — each with a copy-paste config and a way to verify it.
* **[docs/PLAN_BATCH16.md](docs/PLAN_BATCH16.md)** — 并行推理 / batch-16: why the
  parallel axis has to be the batch dimension, what the first 16-row kernel measured,
  and the L2-sharing design that replaces it.
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the forward pass, the
  speculative loop and the Metal kernels fit together.
* **[loop/LOOP.md](loop/LOOP.md)** — the per-round protocol and the performance record.

One thing to know before wiring anything up: the engine serves a **text** model, so
`tools` / `tool_choice` are ignored.  Frameworks that drive a loop with function calls
must go through the shim, which renders the tools into the prompt and parses
`<tool_call>` back into OpenAI `tool_calls`:

```bash
./target/release/qwen38 serve --port 8080 --max-ctx 8192   # the engine
python3 tools/qwen_tool_shim.py                            # :8081, adds tool calling
```

## Quick start

```bash
# 1. toolchain (installed inside the repo; nothing is written to $HOME)
. loop/env.sh

# 2. weights (~16 GB, ModelScope CLI)
./tools/download_model.sh          # add --mtp to also fetch the bf16 MTP shards

# 3. build + tests
cargo build --release --workspace
cargo test --workspace

# 4. inspect the real checkpoint (all 2180 tensors resolved, zero-copy)
cargo run --release -p qw-cli --bin qwen38 -- info

# 5. reference oracle (mlx-lm) for the correctness gate
python3 tools/oracle.py --out loop/artifacts/oracle.json

# 6. serve it (OpenAI + Anthropic, ~12 s to load)
./target/release/qwen38 serve --port 8080 --max-ctx 8192
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

## Working in rounds

`loop/LOOP.md` is the protocol: four gates (build, correctness, performance,
publish), one hypothesis per round, evidence recorded in `loop/rounds/`.
Run one iteration with:

```bash
loop/run_round.sh "short title of this round's hypothesis"
```

Milestones (M0 → M6) and their exit criteria are listed in `loop/LOOP.md`.

## Status

| milestone | state |
|---|---|
| M0 scaffold, Metal JIT, zero-copy loader, loop harness | **done** |
| M1 single-token forward parity with mlx-lm | **done** |
| M2 prefill + KV cache + greedy parity | **done** |
| M3 4-bit GEMV/GEMM perf ≥ 450 GB/s, ≥ 30 tok/s | **done** |
| M4 MTP speculative decoding ≥ 40 tok/s | **done** (42.4 tok/s measured on mains) |
| M5 OpenAI + Anthropic endpoints (streaming) | **done**, `accept.sh` 12/12 |
| M6 tool-calling shim for agent frameworks | **done**, `tools/qwen_tool_shim.py` |

Current iteration: raising the single-decoder target to 50 tok/s. Absolute throughput on
this box swings by 4× with thermal state (battery + Low Power Mode throttles the memory
path, not the cores), so every number in `loop/state.json` is recorded together with the
state it was measured in — read the caveats there before quoting one.

License: Apache-2.0.
