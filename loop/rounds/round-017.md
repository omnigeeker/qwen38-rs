# Round 017 — M5 completed: real token accounting on every protocol

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6) G3=no change G4=pass

Round 16 left `usage.prompt_tokens` hard-coded to 0, which breaks any client that
counts tokens (and is simply wrong). The engine thread knows the prompt length the
moment it encodes it, so it now reports it.

## Change

`engine.rs` sends a typed event stream instead of bare strings:

```rust
pub enum EngineEvent { Prompt(usize), Piece(String) }
```

The prompt length arrives first (right after `tok.encode`), then one event per
generated piece. `collect()` returns `(prompt, text, pieces)` so the non-streaming
responses report both counts, and the SSE paths drop the metadata event through a
small `text_only()` adapter - which also keeps the streaming encoders on their
already-verified shape rather than threading the enum through the unfold state
machines.

A trailing U+FFFD is still held back (round 16b), so multi-byte characters split
across a token boundary no longer leak a replacement character.

## Live evidence

```
POST /v1/chat/completions  -> usage: {prompt_tokens: 12, completion_tokens: 24, total_tokens: 36}
POST /v1/messages          -> usage: {input_tokens: 9, output_tokens: 8}
POST /v1/completions stream-> data: {"delta":{"content":"4"}...}   (then [DONE])

gates: fmt clean, clippy -D warnings clean, 13 test binaries pass
parity: 6/6 cases
```

## The objective is not met yet

The endpoint deliverable is now complete; the **throughput** deliverable is not.
Single-stream decode is still ~27.8 tok/s against a 40 tok/s target, and the only
remaining lever is `docs/PLAN_K2.md` - two tokens per weight pass.

This round also read the layer body of `forward()` (line 467 onward) to price that
refactor precisely. The body is a flat sequence of dispatches per layer, and the
k=2 restructuring has to split it into segments:

```
rmsnorm(x -> h)                       per row
q / k / v projections                 k=2 over both rows        <- bandwidth
q_norm, k_norm, rope, kv_append       per row (pos-dependent)
attention scores + out                per row (causal)
sigmoid gate multiply                 per row
o_proj                                k=2 over both rows        <- bandwidth
residual add, post-attention norm     per row
gate / up projections                 k=2 over both rows        <- bandwidth
silu + gate multiply                  per row
down_proj                             k=2 over both rows        <- bandwidth
residual add                          per row
```

The three non-linear segments between the four projected groups are why this is a
~300 line change rather than a parameter swap: it is a reordering of the layer, not
a widening of it. `TILE` (round 15) and the specialised `K_Q4_GEMV_K2` kernel
(round 6) are already in place for it.
