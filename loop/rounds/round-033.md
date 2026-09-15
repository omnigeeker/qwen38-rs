# Round 033 — the endpoint finally serves the speculative decoder

- date: 2026-09-15
- gates: G1=pass G2=oracle 6/6, see the open issue below G4=pass

## The gap

The CLI reached 40 tok/s three rounds ago; `qwen38 serve` was still running the plain
loop (`set_token` + `forward` + `argmax`), so the endpoint served at **16-17 tok/s** - half
the engine's actual speed, because the speculative decoder lived inside the CLI's `gen.rs`.

## What changed

The step is now a model method, `Qwen38::spec_step(pos, next, out)`, returning the position
and settled token for the next call plus the draft and verify times.  The CLI keeps its
timing report; the server calls the same code, so the two front ends cannot drift apart.
`promote_row1_hidden` became `promote_hidden(row)`.

The server's prefill also had to warm the MTP head's attention state (`mtp_step` on each
token's successor, exactly as the CLI does).  Without it every draft is garbage and
speculation only pays the draft cost: the endpoint measured **9.5 tok/s** before the fix
and **25.0 tok/s** after, on a box that was throttling hard (the CLI read 28-31 tok/s at
the same time).

Verified on the endpoint, all four paths:

| | non-stream | SSE |
|---|---|---|
| OpenAI `/v1/chat/completions` | `chat.completion` + usage | `chat.completion.chunk` |
| Anthropic `/v1/messages` | `message` + usage | `message_start` / `content_block_delta` |

plus `/v1/models`.  `QW_NO_SPEC=1` turns speculation off for A/B work.

## Open issue: spec vs plain diverge at long lengths

Comparing the same prompt through both paths for 300 tokens:

| prompt | differing positions |
|---|---|
| Roman Empire (37-token prompt) | 114, first at index 162 |
| floating point (similar length) | **0** |

The differences start late, re-converge (79% of the tail still agrees), and the two runs
are the same length.  That is the signature of an argmax flipping on a near-tie rather
than state corruption (which would cascade and never re-converge).  Oracle parity stays
**6/6** and round 031's 64-token byte-for-byte comparison passed, so this is not caught by
the existing gates.  The k=1 and k=3 kernels accumulate each output row independently, so
they ought to be bit-identical - meaning something else in the tiled path perturbs the
last bits, and finding it is the first task of the next round.  Until then the endpoint is
self-consistent (every emitted token is the argmax of its own forward pass) and
`QW_NO_SPEC=1` restores the plain path exactly.

Gates: fmt clean, clippy `-D warnings` clean, tests pass, oracle parity 6/6.
