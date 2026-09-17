# qwen38-rs 保姆级教程：把本地 Qwen3.8-27B 跑成一个 OpenAI/Anthropic 端点

> 面向 macOS + Apple Silicon。全程复制粘贴即可，每一步都给出**预期输出**和**失败时怎么办**。
> 读完你能做到：本机起一个 `http://127.0.0.1:8080` 的推理服务，用任何 OpenAI/Anthropic 客户端
> 调用它，并把它接进各种 Agent Framework（见 `docs/AGENT_FRAMEWORKS_zh.md`）。

---

## 0. 先搞清楚能力边界（省你几小时）

| 你想要的 | 现在能不能 | 说明 |
|---|---|---|
| OpenAI `/v1/chat/completions`（含 SSE 流式） | ✅ 能 | 见 §5 |
| OpenAI `/v1/completions`（原始补全） | ✅ 能 | 见 §5 |
| Anthropic `/v1/messages`（含流式） | ✅ 能 | 见 §5 |
| `GET /v1/models`、`GET /health` | ✅ 能 | 见 §5 |
| 任意 api_key | ✅ | 服务端**不校验** key，填个非空字符串即可 |
| **原生 function calling（`tools`）** | ⚠️ 需代理 | 引擎没有实现 `tools`；用 §7 的代理即可让工具型 Agent 工作 |
| Embeddings（`/v1/embeddings`） | ❌ 没有 | 向量检索请另配 embedding 模型 |
| 多并发请求 | ❌ 串行 | 单引擎单队列，见 §6.4 |
| 图片/多模态 | ❌ 没有 | 纯文本 |
| 长输出 | ⚠️ 模型会自己收尾 | 实测 250–400 token 就自发结束，见 §6.3 |

**一句话**：它就是一个**文本 LLM 端点**，标准协议都在；缺的是 `tools` 和 embeddings。

---

## 1. 硬件与前置要求

* Apple Silicon Mac，**统一内存 ≥ 64 GB**（权重 15 GB + KV/状态 + 系统余量；本教程在 M5 Max 128 GB 上验证）
* macOS 26（更早版本大概率也行，未验证）
* 已安装 **Xcode Command Line Tools**（只需要 `clang`/`git`，不需要完整 Xcode）
* Python 3.11（只有做参考校验和跑工具代理时才需要）

确认一下：

```bash
sw_vers -productVersion          # 期望：26.x
uname -m                         # 期望：arm64
xcode-select -p                  # 期望：/Library/Developer/CommandLineTools
```

> **重要提醒**：本机若处于**电池供电 + 低电量模式**，解码速度会掉到市电的约 1/3。
> 想让性能好看，请插电并关闭低电量模式：
> ```bash
> pmset -g batt | head -1          # 看 "AC Power" 还是 "Battery Power"
> pmset -g | grep powermode        # 1 = 低电量模式开启，0 = 关闭
> ```
> 具体影响见 §8。

---

## 2. 准备：工具链 → 权重 → 构建

### 2.1 工具链（装在本仓库内，不污染 `$HOME`）

```bash
cd /path/to/QWen3.8-27B
. loop/env.sh
```

### 2.2 下载权重（约 16 GB）

```bash
./tools/download_model.sh --mtp
```

`--mtp` **不是可选项**：MTP 投机解码头是达到目标吞吐的关键（没有它，解码上限约 30 tok/s）。

预期在 `models/` 下出现：

```
models/Qwen3.8-27B-4bit/         15 GB   主权重（MLX affine 4-bit，group_size=64）
models/Qwen3.8-27B-mtp-4bit/    228 MB   MTP 头（4-bit，服务启动时自动加载）
models/Qwen3.8-27B-bf16-mtp/    3.2 GB   MTP 头的 bf16 原始分片（量化它的原料）
```

> 下载需要 ModelScope CLI：`pip install modelscope`。
> 若网络不稳，脚本可重复执行，已下载的分片会跳过。

### 2.3 构建

```bash
cargo build --release --workspace
```

产物是 **`./target/release/qwen38`**（约 1–2 分钟，取决于机器）。

M5 等机器只有 Command Line Tools 时**不需要** `xcrun metal`：GPU 内核是 MSL 源码，
**运行时 JIT 编译**（`MTLDevice::newLibraryWithSource`）。

> 这也意味着 **MSL 语法错误只在运行时暴露，不在 `cargo build` 时**。改内核后一定要真跑一次。

---

## 3. 上电前自检（强烈建议）

```bash
# 3.1 看 checkpoint 是否完整：应有 0 missing
./target/release/qwen38 info

# 3.2 与 mlx-lm 参考实现逐 token 对齐：应输出 parity: 6/6 cases
./target/release/qwen38 verify --oracle loop/artifacts/oracle.json

# 3.3 完整验收（正确性 + 端点，约 4 分钟）
bash tools/accept.sh
```

`accept.sh` 的最后一行应看到：

```
12 passed, 0 failed
ACCEPTED
```

其中包含两条关键门禁：
* `spec==plain: byte-identical over 300 tokens` —— 投机解码**不改变**输出内容；
* `4-bit GEMV matches the CPU reference on real weights` —— 量化矩阵乘没有算错。

> ### ⚠️ 一条已知的未决问题：`spec==plain` 门禁偶发失败
>
> **实测记录**：该门禁在绝大多数运行中通过（一次 12/12 ACCEPTED；随后连续 7 次单独的
> plain/spec 对比全部逐字节一致，其中 3 次 plain、4 次 spec，两条路径各自也都可复现），
> 但**在 6 次门禁执行中有 1 次报 `230 of 300 tokens differ, first at 68`**
> （前 68 个 token 一致，之后整体级联偏移 —— 典型的"某一步 argmax 翻转后被放大"形态）。
>
> **目前无法确定性复现**：按同样的顺序（先 `verify` + `check` 再跑这一对）重跑仍然一致。
> 最可能的解释是投机路径在高负载/热机下存在**竞态或近 tie 的舍入差异**——
> 批量 verify（一次算 4 行）与单行 kernel 的累加顺序不同，理论上在 top-2 logit 极接近时
> 可以选择不同 token，而这不属于"字节一致"能保证的范畴。
>
> **因此：把"投机解码不改变输出"当作已验证但未证明的性质**。做严格对比时请连跑多次，
> 出现分歧时记录当时的机器状态。这是一个**待查的开放问题**（见 `loop/state.json` 的
> `open_issues`）。

---

## 4. 启动服务

```bash
./target/release/qwen38 serve --port 8080 --max-ctx 8192
```

预期输出：

```
loading engine from models/Qwen3.8-27B-4bit ...
mtp head loaded from models/Qwen3.8-27B-4bit/../Qwen3.8-27B-mtp-4bit
engine ready in 12.1s (context 8192 tokens)
qwen38 listening on http://127.0.0.1:8080
```

**约 12 秒**加载完（权重 `mmap` 零拷贝，主要耗时是 MTP 头与元数据）。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--port` | `8080` | 监听端口（只绑 `127.0.0.1`，不对外暴露） |
| `--model-dir` | `models/Qwen3.8-27B-4bit` | 权重目录 |
| `--model-id` | `qwen3.8-27b-fp4` | 客户端看到的模型名 |
| `--max-ctx` | `8192` | **上下文行数**；`prompt + 输出` 必须放得下 |

### 建议：另开一个终端看日志

服务把日志写到 **stderr**，前台运行。想看请求日志就前台跑；想后台常驻见 §10。

---

## 5. 验证端点（5 条路径，逐条复制即可）

### 5.1 健康检查

```bash
curl -s http://127.0.0.1:8080/health
```

```json
{"engine":"qwen38-rs","model":"qwen3.8-27b-fp4","model_dir":"models/Qwen3.8-27B-4bit","status":"ok"}
```

### 5.2 模型列表（OpenAI 格式）

```bash
curl -s http://127.0.0.1:8080/v1/models
```

```json
{"object":"list","data":[{"id":"qwen3.8-27b-fp4","object":"model","created":...,"owned_by":"local"}]}
```

### 5.3 OpenAI 对话（非流式）

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-fp4",
    "messages": [{"role":"user","content":"用一句话介绍你自己"}],
    "max_tokens": 200,
    "temperature": 0
  }'
```

返回标准 OpenAI 结构。注意两点：

* `finish_reason` **恒为 `"stop"`**（服务端不区分 `length`）；
* `content` 里通常带一段 `<think>…</think>`，这是模型的思考块（见 §6.2）。

### 5.4 OpenAI 对话（SSE 流式）

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp4","stream":true,"max_tokens":100,
       "messages":[{"role":"user","content":"数到五"}]}'
```

以 `data: {...}` 逐块返回，最后 `data: [DONE]`。

### 5.5 Anthropic 对话（非流式 + 流式）

```bash
# 非流式
curl -s http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp4","max_tokens":150,
       "messages":[{"role":"user","content":"说 hello"}]}'

# 流式
curl -N http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp4","max_tokens":100,"stream":true,
       "messages":[{"role":"user","content":"说 hello"}]}'
```

Anthropic 路径支持 `system`、`stop_sequences`、`top_k`、`temperature`、`top_p`。

### 5.6 用官方 SDK 验证（最接近真实框架）

```bash
pip install openai anthropic
```

```python
# OpenAI SDK
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")
r = c.chat.completions.create(
    model="qwen3.8-27b-fp4",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=128,
)
print(r.choices[0].message.content)
```

```python
# Anthropic SDK
from anthropic import Anthropic
c = Anthropic(base_url="http://127.0.0.1:8080", api_key="sk-local")
r = c.messages.create(
    model="qwen3.8-27b-fp4",
    max_tokens=128,
    messages=[{"role": "user", "content": "你好"}],
)
print(r.content[0].text)
```

---

## 6. API 能力对照表与已知行为

### 6.1 字段支持情况

**`/v1/chat/completions` 真正生效的字段**

| 字段 | 行为 |
|---|---|
| `model` | 忽略（只有一个模型） |
| `messages` | 支持 `system` / `user` / `assistant`；`content` 可以是字符串或 parts 数组 |
| `max_tokens` | 生效，默认 **256**，上限为 `--max-ctx` |
| `temperature` / `top_p` / `top_k` | 生效 |
| `stream` | 生效 |
| `stop` | 生效 |
| `seed` | 生效 |

**被静默忽略的字段**（发了不报错，但不起作用）：
`tools`、`tool_choice`、`response_format`、`stream_options`、`n`、`presence_penalty`、
`frequency_penalty`、`logprobs`、`user`。
`/v1/messages` 同理，忽略 `tools` 与 `tool_choice`。

> **给 Agent Framework 的提醒**：`tools` 被忽略时，模型不会崩，它只会**用自然语言描述**它想调用什么，
> 而框架在等结构化的 `tool_calls`，于是表现为"卡住/空转"。**必须走 §7 的代理**。

### 6.2 `<think>` 思考块

这是 Qwen3 系的标准行为：模型先输出 `<think>…</think>`，再输出正式回答。原样透传，不剥离。

* 人看的场景（ChatGPT 式前端）会看到思考过程，属正常；
* 程序消费的场景建议自己剥掉，或直接用 §7 的代理——它会把思考块移到 `reasoning_content`，
  `content` 只留正式回答。

### 6.3 模型会自己收尾

实测：要求"数到 400"只数到 290，要求"重复 120 行"只给 103 行 —— 模型主动输出了
`<|im_end|>`（回合结束符，服务端正确识别为 EOS）。

所以 `max_tokens` 是**上限不是目标**；指望一次生成上千 token 的长文或大段代码替换会失望。
Agent 场景下这反而是好事（每步短、迭代快），但**单次大文件改写不要指望**。

### 6.4 上下文与并发

* `prompt + max_tokens ≤ --max-ctx`。放不下时服务端**明确报错**（不会越界、不会崩）：
  `prompt is 9000 tokens but the context is 8192; restart with a larger --max-ctx`
* **并发 = 1**：模型跑在一个专用线程上，请求经 `mpsc` 排队串行处理。
  多客户端同时打不会错，但会**互相排队**。Agent 框架的并行子任务请自行串行化。

### 6.5 没有 CORS 头

引擎的 HTTP 响应**不带** `Access-Control-Allow-Origin`。因此：

* 服务端调用（Open WebUI、Dify、n8n、Python/Node SDK）**不受影响**；
* 浏览器页面里直接 `fetch` 会被 CORS 拦。此类前端请改指向 §7 的代理（它带 CORS）。

---

## 7. 让工具型 Agent 工作：OpenAI 工具代理

引擎没有 `tools`，但 Qwen3 本身**训练过函数调用格式**（`<tool_call>…</tool_call>`）。
`tools/qwen_tool_shim.py` 做的就是这件事：

```
Agent Framework ──OpenAI(tools)──► :8081 代理 ──渲染进 prompt──► :8080 引擎
        ▲                                  │
        └──── 解析 <tool_call> 还原成 tool_calls ─┘
```

只用标准库，无需安装任何依赖。

### 7.1 启动

```bash
# 另开一个终端（引擎要保持在 8080 运行）
python3 tools/qwen_tool_shim.py
```

```
[shim] listening on http://127.0.0.1:8081/v1  ->  http://127.0.0.1:8080
```

可用环境变量调整：

```bash
QWAN_PORT=9000 QWAN_UPSTREAM=http://127.0.0.1:8080 python3 tools/qwen_tool_shim.py
```

### 7.2 验证代理（一次完整的工具往返）

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-27b-fp4",
    "messages":[{"role":"user","content":"北京现在天气怎么样？用工具查。"}],
    "tools":[{"type":"function","function":{
      "name":"get_weather","description":"查询某城市当前天气",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "max_tokens":400,"temperature":0
  }'
```

预期（已实测）：

```json
{"choices":[{"index":0,
  "message":{"role":"assistant","content":null,
    "reasoning_content":"The user wants to know the current weather in Beijing...",
    "tool_calls":[{"id":"call_0_...","type":"function",
      "function":{"name":"get_weather","arguments":"{\"city\": \"Beijing\"}"}}]},
  "finish_reason":"tool_calls"}],"usage":{...}}
```

要点：`finish_reason` 是 `tool_calls`、`arguments` 是 JSON 字符串、`content` 为 `null`、
思考块进了 `reasoning_content`。

把工具执行结果按标准 OpenAI 格式回灌（`role:"tool"` + `tool_call_id`），即可拿到最终自然语言回答
—— 这条也已实测通过。

### 7.3 接通方式

**把框架的 base_url 从 `http://127.0.0.1:8080/v1` 改成 `http://127.0.0.1:8081/v1`，其余不变。**

### 7.4 代理的取舍（诚实说明）

* 流式是**伪流式**：因为要收完整段回复才能判断有没有工具调用，代理会先缓冲再分块发出。
  首字节延迟 ≈ 整段生成时间。客户端超时请设大一些（`QWAN_TIMEOUT` 默认 600 秒）。
* 工具调用支持**并行调用**（多个 `<tool_call>` 会全部解析出来）。
* 模型偶尔会输出不合法的 JSON，代理会丢弃该次调用并在 stderr 打日志，不会 500。

---

## 8. 性能预期与正确测法

### 8.1 先看这张表

| 场景 | 实测/预期 |
|---|---|
| 电池 + 低电量模式，持续解码 | **≈ 19–21 tok/s** |
| 电池，刚冷却（前几个 token） | 瞬时可达 30+ tok/s，随后按 §8.2 衰减 |
| 市电、关闭低电量模式 | 目标 50 tok/s；当前实现实测标定约 **46–49**（见 `loop/state.json`） |
| 首次请求（含首次 GPU 工作） | 明显更慢（一次性的管线/JIT 预热，约 200 ms 量级） |
| 加载模型 | ≈ 12 s（一次性） |

### 8.2 为什么"测出来的数"经常对不上

这台机器上**热节流极强**，而且**同一进程内、逐 token 就能看到**：

```
冷却态 k=1 前向：30.9 ms  →  热态：~130 ms      （4.2×）
```

原因不是核心降频，而是**内存路径被压**：冷却态权重流 554 GB/s（已达 DRAM 峰值），
热态掉到 ~130 GB/s。

因此：

* **绝对数字几乎不可比**：跑第一趟和第五趟能差 2–4 倍；
* **只有同一次运行内、或严格交替的 A/B 才可信**；
* 报数时**必须说明供电与散热状态**。

### 8.3 怎么正确地拿一个数

```bash
# 让机器冷却后再测（ACCEPT_COOL 秒数，默认 240）
ACCEPT_COOL=240 ./tools/accept.sh --perf

# 或手动：冷却后用固定长度生成读 steady-state 行
QW_SPEC=1 ./target/release/qwen38 gen \
  --prompt "The history of the Roman Empire is a long one that begins" \
  --max-tokens 400 --no-stop 2>&1 | grep steady-state
```

`gen` 的 `steady-state X tok/s (N tokens in T s, P tokens/pass, M ms/token)` 取自**后 1/3 token**，
已经是相对可信的口径。`P tokens/pass` 是投机解码每次前向平均产出的 token 数。

> 注意 `bench` 子命令输出的是**纯线性层扫掠速度**，`end_to_end=false`，**不是** otps，别引用它。

---

## 9. 排错手册

| 现象 | 原因 | 解决 |
|---|---|---|
| `curl: (7) Failed to connect` | 服务没起或端口不对 | 看前台日志；`--port` 是否一致 |
| `/health` 返回 503 / `engine is still loading` | 权重还在加载（约 12 s） | 等；或看日志里的 `engine ready in` |
| `engine failed to load: ...` | 权重目录不对/不完整 | 跑 `qwen38 info`，确认 `0 missing` |
| 输出被截断在中途 | 模型自发 `<|im_end|>` | 正常行为（§6.3）；换更"可续写"的原始补全提示 |
| 输出里出现 `<think>` | Qwen3 思考块 | 正常；程序消费请剥离或用 §7 代理 |
| Agent 框架"卡住"、"一直转圈" | 框架在等 `tool_calls`，引擎没实现 `tools` | 走 §7 代理 |
| 报错 `prompt is N tokens but the context is 8192` | 上下文不够 | 加大 `--max-ctx 16384` 重启（内存会相应增加） |
| 浏览器前端报 CORS | 引擎不带 CORS 头 | 走 §7 代理 |
| 并发请求很慢 | 引擎串行（§6.4） | 正常；减少并行度 |
| 速度只有十几 tok/s | 电池 + 低电量模式 | 插电、`pmset -g \| grep powermode` 确认关闭 |

---

## 10. 日常运维

### 10.1 后台常驻

```bash
mkdir -p logs
nohup ./target/release/qwen38 serve --port 8080 --max-ctx 8192 \
  > logs/serve.log 2>&1 &
echo $! > logs/serve.pid
tail -f logs/serve.log
```

停止：`kill "$(cat logs/serve.pid)"`

### 10.2 开机自启（launchd）

`~/Library/LaunchAgents/local.qwen38.serve.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.qwen38.serve</string>
  <key>ProgramArguments</key><array>
    <string>/ABS/PATH/QWen3.8-27B/target/release/qwen38</string>
    <string>serve</string><string>--port</string><string>8080</string>
    <string>--max-ctx</string><string>8192</string>
  </array>
  <key>WorkingDirectory</key><string>/ABS/PATH/QWen3.8-27B</string>
  <key>StandardOutPath</key><string>/ABS/PATH/QWen3.8-27B/logs/serve.log</string>
  <key>StandardErrorPath</key><string>/ABS/PATH/QWen3.8-27B/logs/serve.log</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
```

```bash
# 把 /ABS/PATH 换成真实绝对路径，然后：
launchctl load  ~/Library/LaunchAgents/local.qwen38.serve.plist   # 启动
launchctl list | grep qwen38                                      # 看状态
launchctl unload ~/Library/LaunchAgents/local.qwen38.serve.plist  # 关闭
```

### 10.3 一次性生成（不进服务）

```bash
./target/release/qwen38 gen --prompt "你好，介绍一下你自己" --max-tokens 200
```

---

## 11. 下一步

* **接进 Agent Framework** → `docs/AGENT_FRAMEWORKS_zh.md`（LangChain、LlamaIndex、Cline、Continue、
  Aider、OpenHands、AutoGen、CrewAI、Dify、n8n、Open WebUI、Cherry Studio、Claude Code … 逐一配置）
* **引擎内部架构** → `docs/ARCHITECTURE.md`
* **迭代记录与性能标定** → `loop/state.json`、`loop/rounds/`
* **手动验收** → `bash tools/accept.sh`

---

## 附：环境变量速查（只影响 CLI，不影响端点行为）

| 变量 | 作用 |
|---|---|
| `QW_SPEC=1` | `gen` 开启 MTP 投机解码 |
| `QW_NO_STOP` / `--no-stop` | 忽略 EOS，用于吞吐测量 |
| `QW_TAIL=1` | 打印尾部各阶段耗时（set_tokens / forward2 / logits / commit） |
| `QW_ENCODE_TIME=1` | 打印每个 `CommandBatch` 的编码与 `commit+wait` 耗时 |
| `QW_SKIP_KERNEL=<name>` | 跳过名字含该子串的内核派发（诊断用；**空串会跳掉全部**） |
| `QW_NO_ACCEPT=1` | 强制不接受任何 draft（诊断用） |
| `QW_COOL_SLEEP=<秒>` | 在 prefill 后空转，用来测"冷却态"性能 |
| `QW_DEBUG=1` | 逐 token 打印残差流统计 |
