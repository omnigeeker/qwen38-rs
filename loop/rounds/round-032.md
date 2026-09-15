# Round 032 — the spec loop generalises over TILE, and the cache completion is not optional

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6 + byte-identical) G4=pass

## What changed

The three-case spec loop became one loop over `TILE` rows: draft `TILE - 1` tokens chained
on each other, forward `TILE` rows, take the longest prefix of drafts the target agrees
with (`k`), emit them, `commit_row(k)` unless row `TILE - 1` is already current, and
`promote_hidden(k)` so the next draft starts from the right hidden.  `TILE` is now the only
thing that decides how wide a pass is.

TILE=4 was built and verified on top of it: 2.91 tokens/pass, parity 6/6, byte-identical -
but its throughput could not be compared against TILE=3 in this session because the box
went into heavy throttling mid-comparison (the same binary reported 36.75 and then 24.74
tok/s).  The committed configuration stays TILE=3, the one with a verified 40.5 tok/s.

## The regression the generalisation introduced

Dropping `mtp_step(d[k-1], pos + k, false)` looked safe: the draft had already written that
token's k/v entry.  It had - but from row 0, which still held the hidden of `pos`, while
token `d[i]` sits at `pos + 1 + i` and needs the hidden before it.  Every chained draft
after the first was therefore cached one or two positions off.

| | with completion | without |
|---|---|---|
| acceptance | 77.6% | 71.2% |
| tokens/pass | 2.53 | 2.40 |

Restoring it (promote row `k - 1`, re-append `d[k-1]` at `pos + k`) returns both numbers to
the hand-written values.  This is the kind of "surely redundant" that costs acceptance
rather than correctness, so the oracle and byte-identical gates would never have caught it.

## Measured

`spec: 59/76 drafts accepted (77.6%), 2.53 tokens/pass`, parity 6/6, speculative output
byte-identical to plain greedy.  Throughput readings in this session (19-37 tok/s) are
thermal noise, not signal; the clean 40.54/40.68 tok/s of round 031 stands as the TILE=3
result.

Gates: fmt clean, clippy `-D warnings` clean, tests pass, parity 6/6, spec byte-identical.
