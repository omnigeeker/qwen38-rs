#!/usr/bin/env python3
"""OpenAI tool-calling shim for the local qwen38 endpoint.

The engine serves a text model: it understands `/v1/chat/completions` and
`/v1/messages`, but it does not implement the OpenAI `tools` field.  Agent
frameworks that drive a loop with function calls (Cline, Roo Code, Continue in
agent mode, OpenHands, AutoGen with tools, ...) therefore cannot talk to it
directly.

This proxy sits in front of it and speaks enough of the OpenAI protocol for them
to work:

  * `tools` / `tool_choice` in the request are rendered into the prompt using the
    Qwen3 function-calling template, which is the format the model was trained on;
  * `<tool_call>{"name": ..., "arguments": ...}</tool_call>` in the reply is
    parsed back into OpenAI `tool_calls`;
  * the model's `<think>...</think>` preamble is moved out of `content` (and
    surfaced as `reasoning_content`, which most frameworks ignore or display).

Only the standard library is used, so there is nothing to install.

    python3 tools/qwen_tool_shim.py                      # listens on :8081
    QWAN_PORT=9000 python3 tools/qwen_tool_shim.py       # somewhere else
    QWAN_UPSTREAM=http://127.0.0.1:8080 python3 tools/qwen_tool_shim.py

Then point the framework at `http://127.0.0.1:8081/v1` instead of at the engine.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("QWAN_UPSTREAM", "http://127.0.0.1:8080").rstrip("/")
PORT = int(os.environ.get("QWAN_PORT", "8081"))
HOST = os.environ.get("QWAN_HOST", "127.0.0.1")
TIMEOUT = float(os.environ.get("QWAN_TIMEOUT", "600"))

# The Qwen3 function-calling template, verbatim from the model card.  The model
# was trained with exactly this wording, so it is worth not paraphrasing.
TOOL_PREAMBLE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
DANGLING_THINK_RE = re.compile(r"^.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def log(*args: object) -> None:
    print("[shim]", *args, file=sys.stderr, flush=True)


def upstream(path: str, payload: dict | None = None, stream: bool = False):
    """Call the engine.  Returns a file-like object, or parsed JSON."""
    url = f"{UPSTREAM}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"upstream {e.code}: {body}") from None
    return resp if stream else json.loads(resp.read().decode("utf-8", "replace"))


def render_tools(tools: list[dict]) -> str:
    """One JSON schema per line, as the template expects."""
    lines = []
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if not fn:
            continue
        lines.append(json.dumps({"type": "function", "function": fn}, ensure_ascii=False))
    return "\n".join(lines)


def inject_tools(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Prepend the tool block to the system message (or create one)."""
    block = TOOL_PREAMBLE.format(tools=render_tools(tools))
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = f"{block}\n\n{m.get('content') or ''}"
            return out
    return [{"role": "system", "content": block}] + out


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Flatten the tool-call turns the framework sends back to us.

    `assistant` messages carrying `tool_calls`, and the `tool` role results that
    follow them, are turned into plain text: the model only ever sees a
    conversation, never a structured tool protocol.
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            name = m.get("name") or m.get("tool_call_id") or "tool"
            out.append({"role": "user", "content": f"<tool_response>\n{content_text(m)}\n</tool_response>"})
            continue
        if role == "assistant" and m.get("tool_calls"):
            parts = [content_text(m)] if content_text(m) else []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(
                    "<tool_call>\n"
                    + json.dumps(
                        {"name": fn.get("name"), "arguments": json.loads(fn.get("arguments") or "{}")},
                        ensure_ascii=False,
                    )
                    + "\n</tool_call>"
                )
            out.append({"role": "assistant", "content": "\n".join(parts)})
            continue
        out.append({"role": role or "user", "content": content_text(m)})
    return out


def content_text(m: dict) -> str:
    """OpenAI content may be a string or a list of parts."""
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c)


def split_think(text: str) -> tuple[str, str]:
    """Return (visible, reasoning)."""
    reasoning = "\n".join(m.group(1).strip() for m in THINK_RE.finditer(text))
    visible = THINK_RE.sub("", text)
    # A truncated generation can leave an unclosed <think>; if the opening tag is
    # there with no closing one, everything from it on is reasoning.
    if "<think" in visible.lower():
        idx = visible.lower().index("<think")
        reasoning = (reasoning + "\n" + visible[idx:]).strip()
        visible = visible[:idx]
    elif not reasoning and "</think" in visible.lower():
        m = DANGLING_THINK_RE.match(visible)
        if m:
            reasoning = visible[: m.end()].strip()
            visible = visible[m.end() :]
    return visible.strip(), reasoning.strip()


def parse_tool_calls(text: str) -> list[dict]:
    """Pull `<tool_call>` blocks out of a reply.  Tolerates fenced JSON."""
    calls = []
    for i, raw in enumerate(TOOL_CALL_RE.findall(text)):
        body = FENCE_RE.sub("", raw.strip()).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            log(f"tool_call #{i} is not JSON, dropping: {body[:120]!r}")
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", {})
            calls.append(
                {
                    "id": f"call_{i}_{os.urandom(4).hex()}",
                    "type": "function",
                    "function": {
                        "name": obj["name"],
                        "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                    },
                }
            )
    return calls


def handled_text(text: str) -> tuple[str, str, list[dict]]:
    visible, reasoning = split_think(text)
    calls = parse_tool_calls(visible)
    if calls:
        # Anything that is not the call itself is the model thinking out loud
        # about the call; keep it as reasoning so the framework does not show a
        # half-sentence next to the tool call.
        prose = TOOL_CALL_RE.sub("", visible).strip()
        if prose:
            reasoning = (reasoning + "\n" + prose).strip()
        visible = ""
    return visible, reasoning, calls


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter access log
        log(fmt % args)

    # ---------------------------------------------------------------- helpers
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _error(self, code: int, msg: str) -> None:
        self._json(code, {"error": {"message": msg, "type": "shim_error", "code": code}})

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    # ------------------------------------------------------------------ routes
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/v1/models", "/models"):
            try:
                self._json(200, upstream("/v1/models"))
            except Exception as e:  # noqa: BLE001
                self._error(502, str(e))
        elif path in ("/health", "/healthz", "/"):
            self._json(200, {"status": "ok", "shim": True, "upstream": UPSTREAM})
        else:
            self._error(404, f"no route {path}")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        try:
            req = self._read()
        except Exception as e:  # noqa: BLE001
            self._error(400, f"bad json: {e}")
            return
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                self.chat(req)
            else:
                # /v1/messages, /v1/completions and anything else pass through.
                self._passthrough("/v1/completions" if path.endswith("completions") else "/v1/messages", req)
        except Exception as e:  # noqa: BLE001
            log("error:", repr(e))
            self._error(502, str(e))

    def _passthrough(self, target: str, req: dict) -> None:
        if req.get("stream"):
            resp = upstream(target, req, stream=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            for line in resp:
                self.wfile.write(line)
            return
        self._json(200, upstream(target, req))

    # ------------------------------------------------------------------- chat
    def chat(self, req: dict) -> None:
        tools = req.get("tools") or []
        msgs = req.get("messages") or []
        if isinstance(msgs, list):
            msgs = normalize_messages(msgs)
        if tools:
            msgs = inject_tools(msgs, tools)

        fwd = dict(req)
        fwd["messages"] = msgs
        fwd.pop("tools", None)
        fwd.pop("tool_choice", None)
        fwd.pop("parallel_tool_calls", None)
        fwd.pop("response_format", None)
        fwd.pop("stream_options", None)
        stream = bool(req.get("stream"))
        fwd["stream"] = False  # buffered here, re-streamed below if asked

        out = upstream("/v1/chat/completions", fwd)
        choice = (out.get("choices") or [{}])[0]
        raw = content_text(choice.get("message") or {})
        visible, reasoning, calls = handled_text(raw)

        message: dict = {"role": "assistant", "content": visible or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        finish = "tool_calls" if calls else (choice.get("finish_reason") or "stop")
        log(
            f"tools={len(tools)} -> {'tool_calls x' + str(len(calls)) if calls else 'text'} "
            f"({len(visible)} chars visible, {len(reasoning)} reasoning)"
        )

        if stream:
            self._stream(out, message, finish)
            return
        self._json(
            200,
            {
                "id": out.get("id", "chatcmpl-shim"),
                "object": "chat.completion",
                "created": out.get("created", 0),
                "model": out.get("model", "local"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": out.get("usage", {}),
            },
        )

    def _stream(self, out: dict, message: dict, finish: str) -> None:
        """Re-chunk a buffered completion as OpenAI SSE.

        The engine streams, but a tool call can only be recognised once the whole
        reply is in hand, so this buffers and then emits.  Frameworks that time
        out on a slow first byte should set a longer client timeout.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        ident = out.get("id", "chatcmpl-shim")
        created = out.get("created", 0)
        model = out.get("model", "local")

        def chunk(delta: dict, fin: str | None = None) -> dict:
            return {
                "id": ident,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
            }

        def emit(obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())

        emit(chunk({"role": "assistant", "content": ""}))
        if message.get("reasoning_content"):
            emit(chunk({"reasoning_content": message["reasoning_content"]}))
        text = message.get("content") or ""
        for i in range(0, len(text), 24):
            emit(chunk({"content": text[i : i + 24]}))
        for idx, tc in enumerate(message.get("tool_calls") or []):
            emit(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": idx,
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": ""},
                            }
                        ]
                    }
                )
            )
            emit(chunk({"tool_calls": [{"index": idx, "function": {"arguments": tc["function"]["arguments"]}}]}))
        emit(chunk({}, finish))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    log(f"listening on http://{HOST}:{PORT}/v1  ->  {UPSTREAM}")
    log("point your agent framework at that base URL; tools are rendered into the prompt")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
