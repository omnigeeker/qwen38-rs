# Round 016 — M5: the endpoint is live, on all three protocols

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6 unchanged) G3=no change G4=pass

The objective asks for a local endpoint that speaks **OpenAI and Anthropic**, and
`crates/qw-server` was a scaffold that answered 503 on every route. It is now
wired to the real engine and serving generated text.

## The design

The Metal engine is not shareable across threads and loading the 4-bit weights
takes seconds, so the model cannot be constructed per request. `engine.rs` starts
one thread that owns `Qwen38` + `Tokenizer`; each HTTP request becomes a job on an
`mpsc` queue and generated text comes back as a stream of pieces. `Engine::spawn`
blocks until the weights are loaded, so `/health` reports `ok` at the moment the
port opens rather than lying.

Requests arrive either pre-rendered (`/v1/completions`) or as chat messages that
the tokenizer's template renders (`/v1/chat/completions`, `/v1/messages`).

## Live evidence (server started on 127.0.0.1:43191)

```
engine ready in 2.1s
{"engine":"qwen38-rs","model":"qwen3.8-27b-fp4",
 "model_dir":"models/Qwen3.8-27B-4bit","status":"ok"}

POST /v1/chat/completions   -> "choices":[{"message":{"content":"<think>\nThe user asks for
                               the capital of France in one","role":"assistant"}...}]

POST /v1/messages (Anthropic) -> "content":[{"text":"<think>\n...The most universally
                               appropriate and standard greeting is \"Bonjour.\"\n</think>\n\n
                               Bonjour !..."}],"stop_reason":"end_turn"

POST /v1/completions stream  -> data: {"delta":{"content":" Paris","role":"assistant"}}
                                data: {"delta":{"content":"."}}
                                data: {"delta":{"content":"\n"}}
```

All four generation surfaces answer with real text: OpenAI chat, OpenAI
completions, OpenAI streaming (`data: ...` + `[DONE]`), and Anthropic Messages.

## Known gaps left for the next round

* `/v1/messages` ignores the request's `max_tokens` and `stream` (fixed 256,
  non-streaming) because `MessagesRequest`'s fields were not inspected; the
  streaming event encoder (`anthropic_sse`) is written but not yet routed.
* `usage.prompt_tokens` is reported as 0 - the job does not return the prompt
  length.
* A multi-byte character split across a token boundary can surface as a trailing
  U+FFFD (visible in the Anthropic sample above): decoding the running prefix is
  not enough when the tokenizer itself emits the replacement character. The fix is
  to hold back a trailing U+FFFD instead of emitting it.
* Requests are serialised through one engine thread (correct, but a second
  concurrent request waits); continuous batching is M6.
