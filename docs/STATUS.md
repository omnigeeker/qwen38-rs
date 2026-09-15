# qwen38-rs — status after round 11

A from-scratch Rust + Metal inference engine for **Qwen3.8-27B FP4** on Apple
Silicon (M5 Max, 128 GB, macOS 26.6.2). All shaders are MSL compiled at runtime;
there is no Xcode, no Metal toolchain and no Python in the engine.

Repo: https://github.com/omnigeeker/qwen38-rs

## 1. Correctness — proven against mlx-lm, token for token

```
$ ./target/release/qwen38 verify --oracle loop/artifacts/oracle.json
short_greeting       16/16 PASS      code_snippet        24/24 PASS
capital_qa           16/16 PASS      math_steps          24/24 PASS
multilingual         16/16 PASS      long_context_marker 12/12 PASS
parity: 6/6 cases
```

The check runs greedy decoding **through this engine** and compares the emitted
token ids with mlx-lm's, step by step, on six prompts. Against the *stepwise*
oracle (`tools/oracle.py --stepwise`) it is 5/6 — mlx-lm's own batched-prefill and
single-token paths disagree with each other on `math_steps`, so no implementation
can match both.

Layer-by-layer agreement with mlx-lm on a 17-token prompt (relative L2 of the
residual stream): **0.3%** at layer 3, **1.8%** at layer 63 — i.e. fp16-vs-bf16
arithmetic noise, nothing structural (it was 3.7% / 54% before the attention
scale bug in round 2 was found).

Gates (`bash loop/verify.sh`): fmt, `clippy -D warnings`, 39 tests (8 of them GPU
kernels checked against CPU references), then the oracle parity above.

## 2. Performance — measured, single stream, 14.41 GB of weights

| what | number |
|---|---|
| weight sweep, k=1 (497 linears) | 26.87 ms → **536 GB/s** = 103% of the 519 GB/s measured device streaming peak |
| weight sweep, k=2 (specialised kernel) | 28.68 ms for **two** tokens → 14.34 ms/token, **69.7 tok/s** equivalent, 502 GB/s |
| end-to-end decode, k=1 | 36.04 ms/token = **27.75 tok/s** (mlx-lm on the same machine: 30 tok/s) |
| theoretical ceiling, 1 token per pass | 33 tok/s |

The single-pass limit is physics: 14.41 GB ÷ 519 GB/s = 27.8 ms. **40 tok/s is
therefore impossible with one token per weight pass** — it needs the second token,
and round 6 showed how to get it nearly free.

## 3. What is refuted (measured, not guessed)

Four rounds attacked the ~9 ms per token that is not the weight sweep
(36.04 total − 26.87 weights). All four are dead:

| hypothesis | measurement |
|---|---|
| R output rows per threadgroup (x reuse) | k=2: 60.9-62.8 ms vs 46.5 ms — worse |
| x slice hoisted into registers | k=2: 63.1 ms — worse |
| 16-byte `uint4` weight loads | k=2: 46.3 ms — wash |
| **runtime-`k` accumulators spilling to thread-local memory** | k=2: 46.6 → **28.6 ms** — the real bug (round 6) |
| encoder teardown per dependency | neutral (rounds 7-8) |
| ~8000 Objective-C binding calls per token | neutral (rounds 9-10) |

The remaining ~9 ms is GPU-side: ~1282 dispatches per token, each with GPU
scheduling latency plus real work (48 conv1d, 48 gated-delta-net steps, ~200
norms, GQA attention, the 248k-logit readback). The lever is **fewer dispatches**
(fusion) or **amortising the whole fixed cost over two tokens**.

## 4. How 40 tok/s is reached from here

Speculative decoding at k=2, using the specialised kernels:

```
pass = 28.68 ms (two tokens of linear work)
     + ~9 ms    (the fixed non-linear cost, unchanged: same 1282 dispatches
                 now serving both positions)
     + ~2.5 ms  (one MTP draft step: 0.42 B params of fp16 weights)
     = ~40 ms per (1 + alpha) accepted tokens

  alpha = 0.8  ->  1.8 tokens / 40 ms  =  45 tok/s
  alpha = 0.6  ->  1.6 tokens / 40 ms  =  40 tok/s
```

So the target is reachable with a *mediocre* acceptance rate, and the two pieces
left are both ordinary work:

1. **The k=2 engine path** — attention over two positions, the delta-net
   recurrence for two consecutive steps, norms and the residual stream for a
   two-row tile. This is also M2's prefill path.
2. **The MTP head** — quantise the local bf16 `mtp.*` shard (MLX affine, group 64),
   add `mtp.fc` plus the single decoder layer, and run draft/verify.

Then M5/M6 for the objective: OpenAI- and Anthropic-compatible endpoints (scaffold
already in `crates/qw-server`, currently returning 503) and continuous batching.

## 5. Running it

```bash
. loop/env.sh                                   # workspace-local rustc 1.98.1
cargo test --workspace                          # 39 tests, 8 on the GPU
./target/release/qwen38 verify --oracle loop/artifacts/oracle.json
./target/release/qwen38 gen --prompt "Hello!" --max-tokens 16 --dump-top 6
./target/release/qwen38 bench --tokens 1        # k=1 sweep
./target/release/qwen38 bench --tokens 2 --rows 1   # specialised k=2 sweep
bash loop/verify.sh                             # the full gate set
python3 tools/oracle.py --stepwise              # regenerate the mlx reference
```

Models: `models/Qwen3.8-27B-4bit/` (15 GB, ModelScope) and
`models/Qwen3.8-27B-bf16-mtp/` (the bf16 MTP shard, staged for round 12).
