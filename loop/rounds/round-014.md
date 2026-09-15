# Round 014 — full self-verification before the k=2 refactor

- date: 2026-09-15
- gates: **G1=pass G2=pass (6/6) G3=baseline confirmed G4=pass**

Round 13's plan (`docs/PLAN_K2.md`) needs the repo in a known-good state before a
~300 line change to `forward()`, so this round re-ran the whole gate set end to
end and recorded the baseline it has to protect.

```
=== G1a: cargo fmt ===                     clean
=== G1b: clippy -D warnings ===            clean
=== G1c: unit tests ===                    13 test binaries, 39 tests, 0 failures
                                           (8 of them GPU kernels vs CPU references)
=== G2: end-to-end parity ===              parity: 6/6 cases
                                           short_greeting 16/16, capital_qa 16/16,
                                           code_snippet 24/24, math_steps 24/24,
                                           multilingual 16/16, long_context 12/12
=== gates G1/G2 OK ===
```

Sample generation still correct: `The capital of France is` -> ` Paris.`

Baseline to protect and to beat:

| metric | value |
|---|---|
| end-to-end decode | 35.73 ms/token = **27.99 tok/s** |
| k=1 weight sweep | 26.87 ms (536 GB/s) |
| k=2 weight sweep (specialised kernel) | 28.68 ms for two tokens |
| oracle parity | 6/6 |
| MTP bf16 shard staged | `models/Qwen3.8-27B-bf16-mtp/` present |
| working tree | clean, HEAD `0af6024` |

## The exact first edit of the refactor

`runner.rs::forward` (line 427) destructures the dimensions and then encodes all
64 layers. The first mechanical edit, per `PLAN_K2.md`:

1. `Scratch`: every activation buffer (`x`, `h`, `qg`, `q`, `k`, `v`, `attn_out`,
   `attn_gated`, `conv_out`, `gdn_gated`, ...) doubles in size; row `r` lives at
   byte offset `r * rows * 2`. No dispatch changes yet - the engine keeps using
   row 0, so parity must stay 6/6. That alone is a safe, verifiable commit.
2. Then the ~10 `QLinear::encode` call sites become
   `encode_k(&mut b, &kernel_k, x, y, 2)` with
   `kernel_k = b.kernel(msl::COMMON, msl::K_Q4_GEMV_K2)?` - and the three
   `q`/`k`/`v`/`o` projections of a full-attention layer each get their own
   k=2 encode.
3. Then the non-linear dispatches: issue twice with `buf_offset(r * rows * 2)`,
   delta-net and the conv window sequentially, attention after both `kv_append`s.

Step 1 is the next commit; it is small, additive and cannot change any number.
