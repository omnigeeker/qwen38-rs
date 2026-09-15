# Round 028 — speculative decoding works end to end: +34% and byte-identical output

- date: 2026-09-15
- gates: G1=pass G2=pass (6/6 + byte-identical) G3=**win** G4=pass

## What was wired

On top of round 27's `commit_row`, `gen` now has a real speculative loop
(`QW_SPEC=1`).  `next` is a token the decoder has already settled at position `pos`
but not yet emitted; each pass drafts one token with the MTP head, verifies both rows
with `forward2`, and then:

* **draft accepted** — emit the draft *and* row 1's token (a bonus token the pass
  computed for free), promote row 1's hidden into row 0 for the next draft, and
  complete the head's own cache with the token just verified.
* **draft rejected** — `commit_row(0)` rewinds the recurrent state to exactly the first
  row, and only that row's token is emitted.

The position bookkeeping is the part that has to be exactly right, and it is: after
`forward2(pos)` row 0 is the hidden for `pos` and row 1 for `pos + 1`, so the head is
always fed the hidden of the position *before* the token it consumes, matching the
`mtp_step` contract.

## Results

| | plain greedy | speculative |
|---|---|---|
| decode | 24.06 tok/s (41.6 ms/token) | **32.29 tok/s (30.97 ms/token)** |
| drafts accepted | — | 84.6 - 94.1% |
| tokens per pass | 1.0 | **1.85 - 1.94** |

* **+34% decode throughput** in the one clean (unthrottled) interleaved measurement.
* **Output is byte-identical to plain greedy** on three prompts x 64 tokens, and oracle
  parity stays 6/6.  This is not luck: `forward2`'s rows are bit-exact against the
  sequential k=1 passes (round 27 measured `d = 0.0000`), so the acceptance test simply
  cannot diverge from the target distribution.
* Filling the head's cache with the accepted draft (using row 0 before row 1 is
  promoted over it) raised acceptance from 81.1% to 84.6%.

## The offline acceptance harness was under-reporting

`QW_MTP_CHECK=1` has been reporting 56.2% acceptance and that number has been steering
the plan for rounds.  In the real loop the same head accepts **84.6 - 94.1%**.  Whatever
the harness measures (its own per-step ordering, and it re-runs a k=1 forward for every
comparison), it is not the number speculative decoding actually gets.  Corrected in the
state; the head is much better than it looked.

## How close is 40 tok/s

At 1.85 - 1.94 tokens per pass the target needs the pass at **45 - 48 ms**.  A clean k=2
pass is 49 - 55 ms (28.6 ms of that is the linear sweep at 97% of peak, ~20 ms is the
1282-dispatch chain, plus ~6 ms of drafting).  Nothing here is out of reach, but every
remaining millisecond has to come out of the non-linear chain:

* fuse the residual add into the preceding norm (-64..-128 dispatches)
* cut the draft cost, which is paid once per pass
* a third verified row (needs `forward3`; `q4_gemv_k3` already exists)

Under sustained load the machine throttles hard (plain decode measured 53 -> 85 ms/token
in one session), so throughput is only comparable within an interleaved pair; the
acceptance numbers above are thermal-independent.

Gates: fmt clean, clippy `-D warnings` clean, tests pass, parity 6/6, spec output
byte-identical to greedy.
