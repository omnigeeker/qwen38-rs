# 把本地 qwen38 端点接进各个 Agent Framework

配套阅读：`docs/TUTORIAL_zh.md`（先把端点跑起来并验证通过，再来看本文）。

本文每一条都是**可直接复制的配置**，并标明三件事：**填哪个 URL**、**要不要用 §A 的代理**、**怎么验证**。

---

## 0. 三个通用要素

所有框架都只问三件事，答案永远一样：

| 要素 | 值 |
|---|---|
| **API Base / Base URL** | OpenAI 协议：`http://127.0.0.1:8080/v1`<br>Anthropic 协议：`http://127.0.0.1:8080` |
| **API Key** | 任意非空字符串，例如 `sk-local`（服务端不校验） |
| **Model / Model ID** | `qwen3.8-27b-fp4`（或任意字符串，服务端忽略 `model` 字段） |

### 0.1 该填 8080 还是 8081？

| 你的框架要做什么 | 端口 | 原因 |
|---|---|---|
| 聊天、写作、RAG、总结、代码补全建议 | **8080**（引擎） | 纯文本，引擎直接支持 |
| **自己决定调用哪个工具**（Cline、Roo、OpenHands、AutoGen tools…） | **8081**（代理） | 引擎没有 `tools`，代理负责把工具调用翻译成模型能懂的 prompt 并解析回来 |
| 用 Anthropic 协议（Claude Code、Anthropic SDK） | **8080** | 见 §E，注意 Claude Code 的限制 |

代理的启动方式与验证见 `docs/TUTORIAL_zh.md` §7。它和引擎**同时运行**，互不影响。

### 0.2 接入前先自检（30 秒）

```bash
curl -s http://127.0.0.1:8080/health           # status: ok
curl -s http://127.0.0.1:8080/v1/models        # id: qwen3.8-27b-fp4
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp4","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

三条都出东西，再往下配。**不要跳过这一步**——90% 的"框架连不上"最后都是端点没起或端口填错。

---

## A. 需要工具调用的框架（走 8081 代理）

```bash
# 先确认引擎在 8080，然后：
python3 tools/qwen_tool_shim.py        # 监听 8081
```

### A.1 Cline / Roo Code（VS Code 插件）

设置 → Provider 选 **OpenAI Compatible**：

| 字段 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8081/v1` |
| API Key | `sk-local` |
| Model ID | `qwen3.8-27b-fp4` |
| Context Window | `8192`（与 `--max-ctx` 一致） |
| Max Output Tokens | `1024` 左右即可（模型通常 300 token 内自己收尾） |
| **Supports Images** | ❌ 关闭 |
| **Supports Computer Use** | ❌ 关闭 |

> Cline 依赖模型稳定输出工具调用。本模型能出（已实测 `get_weather` → `{"city":"Beijing"}`），
> 但**单步思考短、会主动收尾**，所以复杂任务请把 `system prompt` 里"一次只做一件事"写死，
> 并接受"需要多轮"的节奏。

### A.2 OpenHands

`~/.openhands/config.toml`（或启动时的 LLM 配置界面）：

```toml
[llm]
model = "openai/qwen3.8-27b-fp4"
base_url = "http://127.0.0.1:8081/v1"
api_key = "sk-local"
max_output_tokens = 2048
temperature = 0.2
```

验证：`openhands` 启动后让它 `ls /tmp`，看它是否发出 `execute_bash` 工具调用。

### A.3 AutoGen（带 function calling）

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = [{
    "model": "qwen3.8-27b-fp4",
    "base_url": "http://127.0.0.1:8081/v1",   # 用代理
    "api_key": "sk-local",
}]

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user = UserProxyAgent("user", human_input_mode="NEVER",
                      code_execution_config={"work_dir": "coding", "use_docker": False})
user.initiate_chat(assistant, message="把 /tmp 下的文件列出来")
```

> 不带工具时用 8080 即可（见 §B.7）。

### A.4 Continue（agent 模式 / tools）

`~/.continue/config.yaml`：

```yaml
models:
  - name: qwen38 agent
    provider: openai
    model: qwen3.8-27b-fp4
    apiBase: http://127.0.0.1:8081/v1   # agent 模式走代理
    apiKey: sk-local
    contextLength: 8192
    roles: [chat, edit, apply]
```

> 只用 `chat` 角色（不调用工具）时把 `apiBase` 换成 `http://127.0.0.1:8080/v1` 即可。

### A.5 其他工具型框架

任何"填 OpenAI Compatible + 支持 function calling"的框架，做法都一样：
**Base URL 换成 `http://127.0.0.1:8081/v1`**。例如 Goose、Kilo Code、Windsurf 的 OpenAI-compatible 模式、
自研的 ReAct Agent。判断标准只有一条：**它是否要求模型返回 `tool_calls`**。

---

## B. 不需要工具的框架（直接用 8080）

### B.1 OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")

resp = client.chat.completions.create(
    model="qwen3.8-27b-fp4",
    messages=[
        {"role": "system", "content": "你是简洁的中文助手。"},
        {"role": "user", "content": "用三句话解释什么是 KV cache"},
    ],
    max_tokens=512,
    temperature=0.6,
)
print(resp.choices[0].message.content)
```

流式只需 `stream=True`，与官方 API 用法完全一致。

### B.2 OpenAI Node / TypeScript SDK

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "sk-local",
});

const r = await client.chat.completions.create({
  model: "qwen3.8-27b-fp4",
  messages: [{ role: "user", content: "你好" }],
  max_tokens: 256,
});
console.log(r.choices[0].message.content);
```

### B.3 Anthropic Python SDK

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8080", api_key="sk-local")

msg = client.messages.create(
    model="qwen3.8-27b-fp4",
    max_tokens=512,
    system="你是简洁的中文助手。",
    messages=[{"role": "user", "content": "你好"}],
)
print(msg.content[0].text)
```

### B.4 LangChain / LangGraph

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3.8-27b-fp4",
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-local",
    temperature=0.6,
    max_tokens=1024,
    timeout=600,        # 本地推理慢，务必放大超时
)
print(llm.invoke("用一句话说明什么是投机解码").content)
```

LangGraph 把 `llm` 当普通节点用即可。**链式调用请把 `timeout` 调大**（默认超时会误判为故障）。

若要用 `create_react_agent` / `bind_tools`：**把 `base_url` 改成 8081 代理**，
并注意模型的工具调用能力有限，建议 `create_react_agent(llm, tools, prompt="一次只调用一个工具")`。

### B.5 LlamaIndex

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="qwen3.8-27b-fp4",
    api_base="http://127.0.0.1:8080/v1",
    api_key="sk-local",
    is_chat_model=True,
    context_window=8192,
    timeout=600,
)
print(llm.complete("什么是 RAG？"))
```

RAG 时 `Settings.embed_model` 需要**另配**一个 embedding 模型——本引擎没有 `/v1/embeddings`。

### B.6 CrewAI

```python
from crewai import LLM
llm = LLM(
    model="openai/qwen3.8-27b-fp4",
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-local",
    temperature=0.3,
)
```

### B.7 AutoGen（纯对话）

```python
config_list = [{
    "model": "qwen3.8-27b-fp4",
    "base_url": "http://127.0.0.1:8080/v1",   # 不用工具 => 8080
    "api_key": "sk-local",
}]
```

### B.8 Aider

Aider 用 diff 编辑而非 function calling，因此**可以直接连引擎**：

```bash
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-local

aider --model openai/qwen3.8-27b-fp4 \
      --no-show-model-warnings \
      --map-tokens 1024 \
      --no-auto-commits
```

> 建议：`--map-tokens` 调小（模型上下文只有 8192）；大文件重写不要交给它一次做完
> （模型会提前收尾，见 `TUTORIAL_zh.md` §6.3）。

### B.9 Continue（只用 chat/edit）

```yaml
# ~/.continue/config.yaml
models:
  - name: qwen38
    provider: openai
    model: qwen3.8-27b-fp4
    apiBase: http://127.0.0.1:8080/v1
    apiKey: sk-local
    contextLength: 8192
    roles: [chat, edit]
```

### B.10 Open WebUI

用 Docker 起（推荐，避免它自带的 Python 环境冲突）：

```bash
docker run -d --name open-webui -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

浏览器打开 `http://localhost:3000` → 管理员设置 → **连接** → OpenAI API：

| 字段 | 值 |
|---|---|
| API URL | `http://host.docker.internal:8080/v1` |
| API Key | `sk-local` |

> 容器里 **不能** 用 `127.0.0.1`（那是容器自己）；用 `host.docker.internal`。
> 非 Docker 安装则填 `http://127.0.0.1:8080/v1`。
> 保存后点模型下拉的刷新按钮，选 `qwen3.8-27b-fp4`。

### B.11 Cherry Studio / Chatbox / NextChat 等桌面客户端

图形界面，通用填法：

| 字段 | 值 |
|---|---|
| 模型服务商 | OpenAI（或"自定义 OpenAI 兼容"） |
| API 地址 / Base URL | `http://127.0.0.1:8080/v1` |
| API 密钥 | `sk-local` |
| 模型名称 | `qwen3.8-27b-fp4`（手动添加即可） |

> 这类 Electron 客户端是本地进程发起请求，**不需要**代理解决 CORS。
> 纯网页版（浏览器里跑）会被 CORS 拦，改指向 `http://127.0.0.1:8081/v1`。

### B.12 Dify

设置 → 模型供应商 → **OpenAI-API-compatible** → 添加模型：

| 字段 | 值 |
|---|---|
| 模型名称 | `qwen3.8-27b-fp4` |
| API Base URL | `http://127.0.0.1:8080/v1`（Dify 在容器里则用 `http://host.docker.internal:8080/v1`） |
| API Key | `sk-local` |
| **Function calling** | **不支持**（重要：选"支持"会让 Dify 的 Agent 节点永远等不到工具调用） |
| Vision | 不支持 |

Dify 的 **Agent 节点**需要 function calling，请改指向 8081 代理并把 Function calling 设为"支持"；
**Chatflow / 工作流 / 知识库**用 8080 即可。

### B.13 n8n

OpenAI 节点 → Credential → **Base URL** 填 `http://host.docker.internal:8080/v1`（容器内）
或 `http://127.0.0.1:8080/v1`（本机安装），API Key 填 `sk-local`。

n8n 的 **AI Agent 节点**（需要工具）请指向 8081。

### B.14 Flowise / Langflow

两者都有"OpenAI Compatible"节点：

* Base Path / Base URL：`http://127.0.0.1:8080/v1`
* Model Name：`qwen3.8-27b-fp4`
* API Key：`sk-local`

Flowise 若在 Docker 内，同样用 `host.docker.internal`。
带 **Tool Agent** 的流程改 8081。

### B.15 Zed 编辑器

`~/.config/zed/settings.json`：

```json
{
  "language_models": {
    "openai_compatible": {
      "qwen38": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          { "name": "qwen3.8-27b-fp4", "max_tokens": 8192 }
        ]
      }
    }
  }
}
```

Zed 会用它做内联助手与"编辑预测"，属于文本场景，8080 足够。

---

## C. 一份跨框架的最小验证脚本

配完任何一个框架，先用这段确认"端点侧没问题"，再看框架：

```python
#!/usr/bin/env python3
"""最小连通性 + 能力自检。用法: python3 check_endpoint.py [base_url]"""
import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1").rstrip("/")


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


print("models :", [m["id"] for m in get("/models")["data"]])

out = post("/chat/completions", {
    "model": "qwen3.8-27b-fp4",
    "messages": [{"role": "user", "content": "只回复两个字：正常"}],
    "max_tokens": 32, "temperature": 0,
})
msg = out["choices"][0]["message"]
print("reply  :", repr((msg.get("content") or "")[-40:]))
print("usage  :", out["usage"])

# 工具调用（只有在 8081 代理上才会返回 tool_calls）
if ":8081" in BASE:
    out = post("/chat/completions", {
        "model": "qwen3.8-27b-fp4",
        "messages": [{"role": "user", "content": "北京天气如何？用工具。"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "查天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "max_tokens": 256, "temperature": 0,
    })
    tc = out["choices"][0]["message"].get("tool_calls")
    print("tool   :", tc[0]["function"] if tc else "（没有返回 tool_calls）")
```

```bash
python3 check_endpoint.py http://127.0.0.1:8080/v1    # 引擎
python3 check_endpoint.py http://127.0.0.1:8081/v1    # 代理（会额外测工具调用）
```

---

## D. 按模型特性调参（重要，别忽略）

这个 4-bit 量化模型有三个实测特性，直接影响 Agent 表现：

| 特性 | 表现 | 对策 |
|---|---|---|
| **会输出 `<think>` 思考块** | `content` 里混着思考过程 | 用 8081 代理（自动移到 `reasoning_content`）；自己写客户端就正则剥掉 `</think>` 之前的内容 |
| **会主动收尾** | 要求 400 项只给 290 项 | `system` 里写"必须完整输出，不要提前结束"；**单步任务拆小**；不要指望一次生成上千 token |
| **单步输出短（250–400 token）** | 大段代码/长文会被截断 | 让 Agent 分多轮、小步走；`max_tokens` 设 1024 以内就够 |
| **prefill 逐 token 扫权重，且不缓存** | 每轮开始前要等 `历史长度 × 40~85 ms`；3000 token 历史 = **4.2 分钟**，且轮轮重算（平方级） | 历史压到最短；Agent 轮数控制在个位数；优先单轮 RAG 而非长循环；超时设 600 s 以上（见主教程 §6.6） |

推荐的 system prompt 起手式：

```
你是一个执行任务的助手。规则：
1. 需要调用工具时，一次只调用一个，不要在同一轮里同时输出解释和调用。
2. 回答直接给结果，不要复述任务。
3. 输出必须完整，不要在句子中间停止。
```

> Qwen3 支持在用户消息里加 `/no_think` 来抑制思考块。本引擎的模板是固定 ChatML，
> 这个软开关**不一定生效**，可以试，不要依赖。

---

## E. Anthropic 协议与 Claude Code

引擎实现了 `/v1/messages`（非流式 + 流式），支持 `system`、`stop_sequences`、`top_k`。

### E.1 Anthropic SDK

见 §B.3。**可用**。

### E.2 Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=sk-local
export ANTHROPIC_MODEL=qwen3.8-27b-fp4

claude
```

**必须知道的限制**：Claude Code 的 agent loop **强依赖 `tool_use` / `tool_result` 内容块**
（读写文件、跑命令全靠它）。本引擎返回的是纯文本，没有 `tool_use` 块，因此：

* **能**：当普通对话/问答用（问代码问题、解释、写片段）；
* **不能**：让它真正改文件、跑命令——它会一直等不到工具结果。

想在这类 Anthropic 协议的工具型客户端里用，需要一个**Anthropic 侧的工具代理**
（把 `tools` 渲染进 prompt 并返回 `tool_use` 块）。`tools/qwen_tool_shim.py` 目前只做 OpenAI 协议；
Anthropic 侧的转换可以照它的 §7 逻辑改写（把 `parse_tool_calls` 的输出包成
`{"type":"tool_use","id":...,"name":...,"input":...}` 即可）。

---

## F. 常见坑速查

| 现象 | 原因 | 解决 |
|---|---|---|
| 连接被拒 | 端点没起 / 端口不符 | `curl /health`；确认 `serve --port` |
| 框架超时 | 默认超时太短（本地推理慢） | 把框架的 `timeout` 设到 300–600 秒 |
| 输出为空 / 只有思考 | 全被 `<think>` 占了，或 `max_tokens` 太小 | 提高 `max_tokens`；剥离思考块 |
| Agent 一直转圈不干活 | 框架在等 `tool_calls`，却连的是 8080 | 改连 8081 代理 |
| 报上下文超限 | `prompt + max_tokens > --max-ctx` | 重启时 `--max-ctx 16384`；或减小系统提示 |
| Docker 里连不上 | 容器里的 `127.0.0.1` 是容器自身 | 用 `host.docker.internal` |
| 浏览器前端 CORS 报错 | 引擎不带 CORS 头 | 走 8081 代理 |
| 并发很慢 | 引擎串行处理请求，且多开进程无收益（实测 0.99×，受内存带宽限制） | 正常；Agent 并行子任务请串行化 |
| 第一句话要等很久 | prefill 逐 token 扫权重、无前缀缓存 | 缩短历史；见主教程 §6.6 |
| 速度只有十几 tok/s | 电池 + 低电量模式 | 插电并关低电量模式（见主教程 §8） |

---

## G. 一页速查

```
引擎      : ./target/release/qwen38 serve --port 8080 --max-ctx 8192
工具代理  : python3 tools/qwen_tool_shim.py          # -> 127.0.0.1:8081
模型名    : qwen3.8-27b-fp4
密钥      : sk-local（任意非空）

纯文本场景  -> http://127.0.0.1:8080/v1   (OpenAI)
            -> http://127.0.0.1:8080      (Anthropic)
工具型 Agent -> http://127.0.0.1:8081/v1   (OpenAI，经代理)
Docker 内   -> http://host.docker.internal:8080/v1

不支持的: tools(直连)、embeddings、多模态、并发>1、response_format
```
