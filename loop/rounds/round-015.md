# Round 015 — PLAN_K2 step 1: the two-token tile is reserved

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6 unchanged)** G3=no change (expected) G4=pass

## What changed

`Scratch` (the activation working set) now allocates `TILE = 2` rows for every
per-token activation: `x`, `h`, `qg`, `pk`, `pv`, `q`, `k`, `attn_out`,
`attn_gated`, `proj_out`, `z`, `a`, `b`, `conv_out`, `gdn_y`, `gdn_gated`,
`mlp_gate`, `mlp_up`, `mlp_act`, `logits`, `scores`.

`window` is deliberately **not** doubled: it is the four-row convolution history
that legitimately spans steps, not a per-token tile.

Nothing else changed. Every dispatch still addresses offset 0, so this cannot move
a number - which is exactly the point of doing it first:

```
parity: 6/6 cases                       (unchanged)
decode: 35.95 ms/token = 27.82 tok/s    (unchanged, within noise)
```

Cost: a few MB of extra scratch. Risk: none - no shader sees the new space.

## Why a whole round for this

`docs/PLAN_K2.md` calls for a ~300 line change to `forward()`, and the previous
three rounds established that this refactor must be landed in *independently
verifiable* pieces rather than in one go. Reserving the tile is the only piece
that changes an allocation without touching a single dispatch, so it is the only
one that can be proven safe by the gate set alone. The next pieces are:

1. the ~10 `QLinear::encode` sites become `encode_k(..., TILE)` with
   `msl::K_Q4_GEMV_K2` (the linear part is 26.87 of the 35.9 ms, so this is the
   piece that actually buys the second token);
2. the non-linear dispatches are issued per row with `buf_offset(r * rows * 2)`,
   delta-net and the conv window sequentially, attention after both `kv_append`s;
3. re-check 6/6 and compare a `forward2` logit row against a plain `forward`.
