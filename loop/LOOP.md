# qwen38-rs — Loop Engineering Protocol

> 目标（不变式，跨轮次保持不变）：
> 用 **Rust** 在 **Apple M5 Max / 128 GB / macOS 26** 上实现一个全新的本地推理引擎，
> 部署 **Qwen3.8-27B FP4**（MLX affine 4-bit, group_size=64），**支持 MTP 投机解码**，
> 在不牺牲正确性的前提下，把 **单流 decode 吞吐优化到 ≥ 40 tok/s**，
> 最终提供 **OpenAI 兼容 + Anthropic 兼容** 的本地 Endpoint。

本文件是整个迭代循环的“宪法”。每一轮都必须按此执行，并留下可复现的证据。

---

## 1. 里程碑（Milestones）

| ID | 目标 | 出口条件（Evidence） |
|----|------|----------------------|
| M0 | 工程骨架 + Metal 运行时 JIT + 权重零拷贝加载 + Loop 机制 | `cargo test` 全绿；`qwen38 info` 打印真实权重统计；GPU kernel `q4_gemv` 与 CPU 反量化参考一致 |
| M1 | 单 token 前向：embedding / RMSNorm / MLP / GQA 全注意力 / Gated DeltaNet | 第 0 层、前 4 层、全 64 层 hidden state 与 mlx-lm 参考逐步对齐（max-abs < 2e-2 @fp16） |
| M2 | prefill + KV/递归状态缓存 + 采样循环 | 固定 prompt 下 greedy 输出与 `mlx-lm` **逐 token 相同** |
| M3 | 4-bit GEMV/GEMM 性能工程 | 有效带宽 ≥ 450 GB/s；单流 decode ≥ 30 tok/s（无投机） |
| M4 | MTP 投机解码（模型自带 1 层 MTP head） | 单流 decode **≥ 40 tok/s**，且 M2 正确性不回归 |
| M5 | HTTP Endpoint（OpenAI + Anthropic，含 SSE 流式） | curl 实测两种协议均可流式返回；不回归 M2/M4 |
| M6 | 批处理 / 连续批处理（可选增益） | 8 并发总吞吐 ≥ 120 tok/s |

**Definition of Done**：M0–M5 全部达成，且 `loop/rounds/` 中存在对应的通过证据。

---

## 2. 四道闸门（Gates）

每一轮**必须**依次通过；任何一道失败，本轮结论为 `REJECTED`，不得声明进展。

* **G1 — 构建与静态检查**
  `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
* **G2 — 正确性（硬闸门，性能再好也不能破）**
  1. 全部 kernel 单测通过（GPU vs CPU 参考）；
  2. 端到端 greedy 一致性：`loop/artifacts/oracle.json` 中每条 prompt 的输出 token 序列与引擎**完全一致**；
  3. 首 token top-8 logits 与参考的偏差在容差内（fp16 累加路径：`max_abs_diff < 5e-2`，相对排序一致）。
* **G3 — 性能**
  `qwen38 bench` 输出 `tok/s`。相对 `loop/state.json` 中的 `best_tok_s` 允许波动 ≤ 3%；
  若下降更多，必须给出解释并在同一轮内回退或修复。
* **G4 — 发布**
  提交并推送到 GitHub（`omnigeeker/qwen38-rs`），commit message 内含本轮指标。

---

## 3. 单轮流程（`loop/run_round.sh`）

1. 读取 `loop/state.json`，确定 `round = N` 与 `next_actions`；
2. 选定**单一假设**（一轮只改一件事，便于归因）；
3. 实现改动；
4. G1 → G2 → G3；
5. 把结果写进 `loop/rounds/round-NNN.md`（假设 / 改动 / 证据 / 结论）；
6. 更新 `loop/state.json`（`best_tok_s`、`gates`、`next_actions`）；
7. G4：`git add -A && git commit && git push`；
8. 若 DoD 未达成 → 保持目标 active，进入下一轮。

---

## 4. 正确性证伪优先（Falsification-first）

* 任何性能优化（kernel 融合、量化重排、KV 量化、MTP）**先**跑 G2，**再**看 G3。
* 分层定位：先用 `tools/oracle.py --dump-hidden` 比对逐层 hidden state，
  把误差定位到具体模块（embedding / attn / gdn / mlp / norm / lm_head），再修。
* 数值约定（来自 mlx-lm `qwen3_5.py`，必须严格遵守）：
  * RMSNorm 权重在 checkpoint 中为**零中心**存储，加载时 `+1.0`；
  * `gated_delta`：`q = inv_scale² · rmsnorm(q)`，`k = inv_scale · rmsnorm(k)`，`inv_scale = head_k_dim^-0.5`；
  * `beta = sigmoid(b)`，`g = exp(-exp(A_log) · softplus(a + dt_bias))`，state dtype = fp32；
  * 位置编码：partial RoPE（`partial_rotary_factor = 0.25`，rotary_dim = 64），theta = 1e7。
* 容差必须写进测试，禁止“看起来对”。

---

## 5. 性能分析方法（每一轮都要有量化证据）

* 层级计时：attention / gdn / mlp / lm_head 各自耗时占比（`--profile`）。
* 有效带宽 = 读入权重字节 / 耗时，必须与硬件上限对比：
  M5 Max 实测峰值读带宽 **494–519 GB/s**，4-bit 权重 15.13 GB
  ⇒ 无投机解码的理论天花板 **≈ 30–33 tok/s**。
* **因此 ≥ 40 tok/s 只能靠 MTP 投机解码达成**（2 token/step，接受率 ≥ 70%）。
  这一点是设计前提，不是可选项。

---

## 6. 目录约定

```
crates/qw-metal      Metal 运行时（运行时 MSL JIT、kernel、buffer）
crates/qw-weights    safetensors 解析 + 零拷贝上传 + 量化元数据
crates/qw-model      架构定义、权重命名、config
crates/qw-engine     tokenizer / sampler / KV cache / MTP / 调度
crates/qw-server     OpenAI + Anthropic 协议层与 HTTP
crates/qw-cli        qwen38 命令行
tools/               oracle、下载脚本、prompt 套件
loop/                本协议、状态机、每轮记录、产物
models/              权重（不入库）
```

## 7. 硬约束

* 只允许 **Rust** 作为引擎实现语言；GPU kernel 用 MSL（无 Xcode Metal 工具链可用，
  一律**运行时 JIT**，见 `qw-metal`）。
* 不得为了跑通而裁剪模型结构（必须 64 层、混合注意力、真实 4-bit 权重）。
* 每轮必须推送 GitHub；失败轮次也要记录（诚实的负结果优于虚假的成功）。
