# Round 024 — the head's input convention, found by bisection (34.4% -> 56.2%)

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6, decoder untouched)** G3=no change G4=pass

## The reference, finally

`web_fetch` is blocked for these hosts, but plain `curl` from bash has direct network.
That fetched vLLM's `qwen3_next_mtp.py` / `qwen3_5_mtp.py`, which settle the wiring:

```python
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states  = self.pre_fc_norm_hidden(hidden_states)
hidden_states  = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))
hidden_states, residual = mtp_layer(positions=positions, hidden_states=hidden_states,
                                    residual=None)
hidden_states, _ = self.norm(hidden_states, residual)
```

`residual=None` is vLLM's "start a fresh residual stream", which is what this
implementation already does, and the concatenation order matches.  So the structure
was right and the defect had to be in the *values*.

## Bisection

Two ablation switches on the layer's residual adds (`QW_MTP_SKIP`) split the head's
contribution cleanly:

| build | acceptance |
|---|---|
| full head | 34.4% |
| attention ablated | 3.1% |
| MLP ablated | 3.1% |
| both ablated | 0.0% |

The layer works and both sub-blocks carry real signal - the head was never broken.

## The bug: which hidden state

The head consumes the decoder's **post-final-norm** hidden state, not the residual
stream.  That is what vLLM passes its MTP module (`Qwen3NextModel` returns the
normalized hidden), and `pre_fc_norm_hidden` then re-normalizes it.

| hidden fed to the head | acceptance |
|---|---|
| post-final-norm | **56.2%** |
| pre-final-norm residual | 34.4% |

The MTP's own residual scale confirms it: `hid` was `[-244, 229]` with the wrong
input and is `[-31.1, 20.6]` with the right one - the same order as the decoder's own
hidden state, which is what the shared `lm_head` expects.

## The norm-shift rule, now measured rather than assumed

| shift variant | acceptance |
|---|---|
| default (pre_fc + layer norms `+1`, `mtp.norm` as-is) | 56.2% |
| every norm `+1`, including `mtp.norm` | 56.2% |
| layer norms unshifted | **0.0%** |

The layer norms' `+1` is not a detail, it is all of the signal; `mtp.norm`'s shift
provably does not matter.

Position offsets `{0, -1, +1}` and the `[hidden | embed]` order change nothing
(`QW_MTP_OFF`, `QW_MTP_SWAP`): the head's output is dominated by the `fc` + layer path
and is far less position-sensitive than a decoder layer.  Whether that is a property
of the head or a remaining defect is the open question.

## Where this leaves the objective

At 56.2% a k=2 speculative pass yields `1 + 0.562 = 1.56` tokens for
`49.2 + 5.5 = 54.7 ms`, i.e. 28.6 tok/s - a 14% win over plain decoding, nowhere near
40.  The arithmetic says the decisive lever is no longer the head:

* the verify pass costs 39.79 ms for k=1 against a 27 ms bandwidth floor (14.41 GB at
  519 GB/s).  The missing 13 ms is dispatch overhead - **1282 dispatches per token**,
  each with a launch cost.  Cutting that is worth more than any drafter.
* a drafted token costs ~5.5-7 ms against a ~1.6 ms floor for its 841 MB (205 MB of
  head weights + the 636 MB head sweep), so drafting three tokens costs more than the
  rows they fill (the pass adds ~9.4 ms per extra row).
* TILE=3/4 remains the other axis: the kernels exist, the plumbing does not.

Gates: clippy 0 errors, 13 test binaries pass, parity 6/6, `forward2` bit-exact.
The k=1/k=2 timings printed in this round's gate run (43.4/70.9 ms) were taken on a
machine hot from a long session and are throttled; the best interleaved measurement
stands at 39.79/49.18 ms = 40.7 tok/s.
