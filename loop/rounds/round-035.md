# Round 035 — the conv ring evicted a row the next pass still needed (found and fixed)

- date: 2026-09-15
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte G4=pass

## Root cause

Round 029 replaced the copied convolution history with a ring of `conv_k` (= 4) slots,
arguing that the window is position-indexed and therefore self-correcting.  That argument is
correct for a decoder that advances one position per pass - and wrong for a speculative one,
which writes `TILE` rows per pass and then throws most of them away.

Concretely, with a pass starting at position `p`:

* the pass writes ring slots for `p`, `p+1` and `p+2`;
* the slot for `p+2` is the same slot as `p-2` (`p+2 = p-2 + conv_k`), so **writing the
  second draft evicts the row for `p-2`**;
* if the pass is rejected (`k = 0`) the next pass starts at `p+1`, and *its* first row
  convolves over `{p-2, p-1, p, p+1}` - so it reads the discarded draft's row in place of
  `p-2`.

Both of the extra rows are real rows of the same sequence, so the corruption is a one-row-old
convolution input - small enough that it usually changes nothing, which is exactly why it
survived earlier rounds and why I twice wrote the resulting divergence off as floating-point
noise at a near-tie.  It was not noise.

## The fix

The ring is now `(conv_k + TILE).next_power_of_two()` = 8 slots, so the rows a pass writes
ahead cannot reach the rows the next pass still reads, and it is indexed with a mask rather
than a modulo so the kernel costs exactly what it did before.

## Evidence

| comparison | before | after |
|---|---|---|
| plain vs spec, 200 tokens | 14 diffs, first 162 | **identical** |
| plain vs spec with `QW_NO_ACCEPT=1` (k forced to 0), 200 tokens | **190 diffs, first 4** | **identical** |
| plain vs spec, three prompts x 300 tokens | - | **identical** |
| oracle parity | 6/6 | 6/6 |

The `QW_NO_ACCEPT=1` switch is what turned this from a rumour into a bug: forcing `k = 0`
disables every accepted-draft branch and makes the rejection path the *only* path, which is
how a once-in-a-hundred-passes corruption became a divergence at token 4.  Together with
`QW_TILE_CHECK` - which shows the tiled forward is bit-exact (max|dlogit| = 0.0 at prefixes
37, 101, 237) - the two switches separate "the arithmetic is wrong" from "the state is
wrong", and here it was unambiguously the state.

## Also this round

* `TILE = 4` was built and tested: correct (identical over 300 tokens) but strictly worse -
  acceptance collapses from 75.5% to 55.6% and tokens/pass only rises from 2.51 to 2.67,
  while the pass costs materially more.  `TILE = 3` stays.
* The box is heavily throttled from back-to-back model loads right now (plain decode reads
  87 ms/token against 24.6 ms in round 031's window), so absolute throughput is being
  re-measured in a cool window; the spec/plain ratio is 1.38x, unchanged.

## Cool-window re-measurement (after the fix)

The box was left to idle for five minutes, then three speculative runs and one plain run:

```
spec run 1: decode 40.65 tok/s (24.60 ms/token)
spec run 2: decode 38.03 tok/s (26.29 ms/token)
spec run 3: decode 40.49 tok/s (24.70 ms/token)
plain    : decode 26.94 tok/s (37.11 ms/token)
```

So the ring fix is performance-neutral - it reproduces round 031's 40.68 / 40.54 - while the
output is now provably the plain path's.  **But one run in three came in at 38.03, below the
40 target, so the bar is met on a median, not robustly.**  That is the reason the goal stays
open: the remaining throughput work is about margin, not about reaching 40 once.

## Endpoint re-verified on the corrected build

```
/v1/models              {"object":"list","data":[{"id":"qwen3.8-27b-fp4",...}]}
OpenAI non-stream       object=chat.completion finish=stop usage={completion_tokens:8, prompt_tokens:15, total_tokens:23}
Anthropic non-stream    type=message stop_reason=end_turn usage={input_tokens:15, output_tokens:8}
OpenAI SSE              11 `data:` chunks
Anthropic SSE           message_start, content_block_start, 9x content_block_delta, content_block_stop, message_stop
```

The server drives `spec_step` and now also calls `enable_spec_snap()`, so its output is the
same verified sequence the CLI produces rather than the unrecovered-state sequence it was
serving before this round.
