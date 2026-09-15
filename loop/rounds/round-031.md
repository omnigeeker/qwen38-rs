# Round 031 — three rows per pass: speculative decoding crosses 40 tok/s

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6 + byte-identical) G3=**target reached** G4=pass

## Why this now

Round 30 made the draft cheap (6.35 -> 2.60 ms).  That changes the arithmetic of
speculation completely: with a draft at its bandwidth floor, **verify rows are the only
things worth buying**, and the pass should carry as many verified rows as it can.

At two rows the pass produced 1.85 tokens; at three it should produce ~2.55.  The row
machinery was already there - `TILE` sized every scratch buffer, the attention and GDN
loops iterate `0..TILE`, `commit_row` handles any row, and `Q4_GEMV_KS` already generated
a k=3 specialisation - so this was mostly control flow.

## The change

`TILE` 2 -> 3 and the tile-specialised GEMV now loads `K_Q4_GEMV_K3`.  `promote_row1_hidden`
became `promote_hidden(row)` so any row can be moved into row 0.

The spec loop drafts two tokens ahead (`d1` for `pos+1`, then `d2` for `pos+2` chained on
it), forwards three rows, and then:

| case | tokens emitted | state |
|---|---|---|
| `d1 == r0 && d2 == r1` | `d1`, `d2` | `r2` is settled by row 2 and becomes the next `next`; promote row 1, complete the head's cache for `d2`, promote row 2 |
| `d1 == r0` | `d1` | `r1` becomes `next`; `commit_row(1)` rewinds the recurrence to the end of row 1 |
| otherwise | - | `r0` becomes `next`; `commit_row(0)` |

Promoting row 1 into row 0 before `mtp_step(d2, pos+2, false)` is what lets the head
consume `d2` at its own position, and it leaves row 2 intact for the next promotion.

## Measured

```
spec: 59/76 drafts accepted (77.6%), 2.53 tokens/pass
timing: decode 40.68 tok/s (24.58 ms/token)
timing: decode 40.54 tok/s (24.67 ms/token)
```

Two consecutive runs above **40 tok/s**, up from 35.61 with two rows (+14%), on a total of
2.53 tokens/pass.  Oracle parity stays 6/6 and speculative output stays byte-identical to
plain greedy - which also validates the k=3 GEMV specialisation, since parity and the
64-token byte-for-byte comparison both exercise it.

The first run after a build reports 36.8 tok/s; the two that follow settle at 40.5-40.7.
The target is met in a normal thermal state but not yet with margin, which is the next
problem: the marginal verify row costs ~8 ms, and the box throttles hard under sustained
load.

## Next

- **TILE=4** (3.18 tokens/pass): ~45 tok/s if the marginal row stays near 8 ms, which buys
  the margin the target needs.  The spec loop wants generalising over `TILE` rows first.
- Then the chain: the marginal row is 5 GDN dispatches + 3 attention dispatches per layer.

Gates: fmt clean, clippy `-D warnings` clean, tests pass, parity 6/6, spec byte-identical.
