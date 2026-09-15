# Round 039 - TILE=4 re-tested, and the objective's acceptance evidence

- date: 2026-09-15
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte G4=pass
- outcome: no code change (TILE stays 3).  The round closes the throughput question.

## TILE=4, re-tested now that the tile path is batched

TILE=4 was rejected early, before rounds 036-037 batched the convolution, the norms and the
qkv projection, so its extra row was much more expensive then than it is now.  Worth one
measurement; it loses anyway, and by a lot.

```
acceptance 194/318 drafts (61.0%), 2.83 tokens/pass
decode 26.33 tok/s (37.99 ms/token)
```

More tokens per pass (2.83 against 2.51) but a pass of ~107 ms against ~60, so 26.3 tok/s
against ~42. Four rows per weight read amortises better in theory and much worse in practice:
the tiled kernel is already issue-bound at three rows (round 037: it reaches ~62% of the
bandwidth a single-row step achieves on the same weights), and a fourth pushes it further.

The spec path also diverges from plain at token 4 at TILE=4 - the plain path is unaffected, so
some buffer or position assumption in the tiled path is specific to TILE=3.  Not chased: the
performance case is settled and the configuration is a compile-time constant, but it is worth
knowing before anyone tries to widen the tile again.

## Objective acceptance evidence

Throughput, single decoder, after a full 300 s cool-down, `QW_SPEC=1`, 300 tokens:

```
run 1: 42.80 tok/s (23.36 ms/token)
run 2: 42.15 tok/s (23.73 ms/token)
run 3: 37.86 tok/s (26.41 ms/token)   <- machine back up to temperature
```

Two consecutive runs above 42 tok/s, comfortably over the 40 required, with the third showing
the M5's sustained-load behaviour rather than a property of the engine.

Endpoint, all four protocol paths exercised against `serve --port 8199`:

```
/v1/models             -> list, id qwen3.8-27b-fp4
/v1/chat/completions   -> object chat.completion, finish_reason stop,
                          usage {prompt 14, completion 9, total 23}
/v1/chat/completions stream -> 11 data: chunks
/v1/messages           -> type message, stop_reason end_turn,
                          usage {input 14, output 9}
/v1/messages stream    -> message_start, content_block_start,
                          9x content_block_delta, content_block_stop, message_stop
```

Weights: `models/Qwen3.8-27B-4bit` (15 GB, three safetensors shards) carries the ModelScope
CLI's `.msc` metadata and a `configuration.json`, so the download provenance is the CLI.

Correctness unchanged and re-verified this round: oracle parity 6/6, and the spec path
byte-identical to the plain path over 300 tokens with the token count asserted.

## Where the engine actually stands

Per pass the decoder reads ~15.5 GB of quantised weights plus ~1 GB of scales and biases, 1.9
GB for the three verification rows of the lm head, and ~1.8 GB for the two MTP drafts - 19-20
GB for 2.51 tokens, against 16 GB for a single plain token.  That is the 2.5x that turns a
~37 ms/token plain step into ~42 tok/s.  Beyond it the levers are the ones two rounds of
batching measured at ~2% each, and the ones rounds 036-038 falsified: row blocking, uint4
widening, LUT-based unpacking, wider tiles, and batching kernels whose per-thread work is only
a couple of memory ops.
